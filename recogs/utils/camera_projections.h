#pragma once

namespace recogs
{
namespace detail
{
template <typename CALLBACK>
__global__ void project_points_kernel(
    const glm::vec3* ws_points, size_t num_points, glm::mat4 view_matrix, glm::mat3 K, CALLBACK callback)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    glm::vec3 vs_point = view_matrix * glm::vec4(ws_points[i], 1);
    if (vs_point.z < 0) return; // Behind the camera
    glm::vec3 ss_point = K * vs_point;
    ss_point /= ss_point.z;
    int2 pixel;
    pixel.x = (int) ss_point.x;
    pixel.y = (int) ss_point.y;
    callback(i, pixel, vs_point.z);
}

/// \param ss_points_d list of screen-space points to unproject; encoded as 16-bit X, Y
template <typename CALLBACK>
__global__ void unproject_points_kernel( //
    const glm::ivec2* ss_points,
    size_t num_points,
    Image4fHWC color_depth,
    glm::mat4 inv_view,
    glm::mat3 inv_K,
    CALLBACK callback)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    uint32_t px = ss_points[i].x;
    uint32_t py = ss_points[i].y;
    uint32_t W = color_depth.width;
    uint32_t H = color_depth.height;
    if (px >= W || py >= H) return;
    glm::vec3 p(px, py, 1.0f);
    p = inv_K * p;
    float view_z = color_depth.value(px, py).w;
    p *= (view_z / p.z);            // View-space
    p = inv_view * glm::vec4(p, 1); // World-space
    callback(i, p);
}
} // namespace detail

template <typename CALLBACK>
void project_points(const thrust::device_vector<glm::vec3>& ws_points,
                    const Camera& camera,
                    CALLBACK callback,
                    cudaStream_t stream)
{
    if (ws_points.empty()) return;

    size_t num_points = ws_points.size();
    dim3 num_blocks = div_ceil<size_t>(num_points, 512);
    dim3 block_dim = 512;
    detail::project_points_kernel<CALLBACK><<<num_blocks, block_dim, 0, stream>>>( //
        thrust::raw_pointer_cast(ws_points.data()),
        ws_points.size(),
        camera.viewmatrix(),
        camera.K(),
        callback);
}

template <typename CALLBACK>
void unproject_points(const thrust::device_vector<glm::ivec2>& ss_points,
                      const Image4fHWC& color_depth,
                      const Camera& camera,
                      CALLBACK callback,
                      cudaStream_t stream)
{
    CHECK_ARG(color_depth.size() == camera.resolution());

    if (ss_points.empty()) return;

    size_t num_points = ss_points.size();
    dim3 num_blocks = div_ceil<size_t>(num_points, 512);
    dim3 block_dim = 512;
    detail::unproject_points_kernel<CALLBACK><<<num_blocks, block_dim, 0, stream>>>( //
        thrust::raw_pointer_cast(ss_points.data()),
        ss_points.size(),
        color_depth,
        camera.inv_view(),
        camera.inv_K(),
        callback);
}
} // namespace recogs
