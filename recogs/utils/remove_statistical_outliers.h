#pragma once

#include <cukd/knn.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "utils/cuda_utils.h"
#include "utils/misc_utils.h"

// CUDA implementation of:
// https://github.com/isl-org/Open3D/blob/da4d8fcdf60a59d20b2fd9f86f2e0955c5ba3c45/cpp/open3d/geometry/PointCloud.cpp#L602

namespace recogs
{
namespace detail
{
/// Traits to use glm::vec3 with cudaKDTree
struct cukd_glm_vec3_traits : public cukd::default_data_traits<float3> {
    using point_t = float3;

    static inline __device__ __host__ float3 get_point(const glm::vec3& p)
    {
        // Is this efficient? Anyway, we don't need efficiency here...
        return make_float3(p.x, p.y, p.z);
    }

    static inline __device__ __host__ float get_coord(const glm::vec3& p, int d) { return p[d]; }

    static inline __device__ int get_dim(const glm::vec3&) { return -1; }
    static inline __device__ void set_dim(const glm::vec3&, int) {}
};

template <int K>
__global__ void compute_avg_distances_kernel( //
    const glm::vec3* points,
    size_t num_points,
    float cutoff_radius,
    float* out_avg_distances,
    float* out_avg_distances_sum)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float3 self_float3 = make_float3(points[i].x, points[i].y, points[i].z);
    cukd::FixedCandidateList<K> result(cutoff_radius);
    cukd::stackBased::knn<cukd::FixedCandidateList<K>, glm::vec3, cukd_glm_vec3_traits>(
        result, self_float3, points, num_points);

    float avg_distance = 0.f;
    for (int j = 0; j < K; ++j) { // Assuming the K-wide list is always filled
        float dist2 = result.get_dist2(j);
        avg_distance += std::sqrt(dist2) / float(K);
    }
    out_avg_distances[i] = avg_distance;
    atomicAdd(out_avg_distances_sum, avg_distance);
}

__global__ void compute_var_kernel( //
    size_t num_points,
    const float* avg_distances,
    const float* inout_avg_distances_sum,
    float* out_variance)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float avg_distances_avg = *inout_avg_distances_sum / float(num_points);
    float d = avg_distances[i] - avg_distances_avg;
    atomicAdd(out_variance, d * d);
}

__global__ void filter_pointcloud_kernel( //
    const glm::vec3* points,
    size_t num_points,
    const float* inout_avg_distances_sum,
    float std_ratio,
    const float* variance_sum,
    const float* avg_distances,
    uint32_t* out_num_filtered_points,
    glm::vec3* out_filtered_points)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float avg_distances_avg = *inout_avg_distances_sum / float(num_points);
    float std_dev = std::sqrt(*variance_sum / float(num_points));
    float avg_distance_threshold = avg_distances_avg + std_ratio * std_dev;
    if (avg_distances[i] < avg_distance_threshold) {
        uint32_t out_i = atomicAdd(out_num_filtered_points, 1);
        out_filtered_points[out_i] = points[i];
    } else { // Discard
    }
}
} // namespace detail

template <int K>
thrust::device_vector<glm::vec3> remove_statistical_outliers( //
    thrust::device_vector<glm::vec3>& points,
    float std_ratio,
    float cutoff_radius,
    cudaStream_t stream)
{
    if (points.size() <= K) return points;

    size_t num_points = points.size();

    // Allocate device buffers
    thrust::device_vector<float> avg_distances(num_points);
    thrust::device_vector<float> pointcloud_mean(1, 0.0f);
    thrust::device_vector<float> variance(1, 0.0f);
    thrust::device_vector<uint32_t> num_filtered_points(1, 0);
    thrust::device_vector<glm::vec3> filtered_points(num_points); // Pre-allocated to fit all points

    uint32_t* num_filtered_points_d = thrust::raw_pointer_cast(num_filtered_points.data());

    dim3 num_blocks = div_ceil<size_t>(num_points, 512);
    dim3 block_dim = 512;

    // Build the KDTree for KNN search
    cukd::buildTree<glm::vec3, detail::cukd_glm_vec3_traits>(
        thrust::raw_pointer_cast(points.data()), (int) num_points, nullptr, stream);

    // Compute avg KNN distances per-point,
    // and the global avg KNN distance for the whole pointcloud
    detail::compute_avg_distances_kernel<K><<<num_blocks, block_dim, 0, stream>>>( //
        thrust::raw_pointer_cast(points.data()),
        num_points,
        cutoff_radius,
        thrust::raw_pointer_cast(avg_distances.data()),
        thrust::raw_pointer_cast(pointcloud_mean.data()));
    // Compute variance of avg distance for the whole pointcloud
    detail::compute_var_kernel<<<num_blocks, block_dim, 0, stream>>>( //
        num_points,
        thrust::raw_pointer_cast(avg_distances.data()),
        thrust::raw_pointer_cast(pointcloud_mean.data()),
        thrust::raw_pointer_cast(variance.data()));
    // Compute an avg distance threshold and filter points based on it
    detail::filter_pointcloud_kernel<<<num_blocks, block_dim, 0, stream>>>( //
        thrust::raw_pointer_cast(points.data()),
        num_points,
        thrust::raw_pointer_cast(pointcloud_mean.data()),
        std_ratio,
        thrust::raw_pointer_cast(variance.data()),
        thrust::raw_pointer_cast(avg_distances.data()),
        num_filtered_points_d,
        thrust::raw_pointer_cast(filtered_points.data()));

    // Cap the filtered points to the actual size
    filtered_points.resize(to_host(num_filtered_points_d));

    return filtered_points;
}
} // namespace recogs
