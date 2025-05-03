#pragma once

#include <cuda/std/limits>

#include <glm/glm.hpp>

#include "cuda_utils.h"

namespace recogs
{
template <glm::length_t LENGTH, typename T>
using Point = glm::vec<LENGTH, T>;

template <glm::length_t LENGTH, typename T>
struct Segment {
    Point<LENGTH, T> p0;
    Point<LENGTH, T> p1;
};

template <glm::length_t LENGTH, typename T>
struct AABB {
    using Point = Point<LENGTH, T>;

    Point min = Point{cuda::std::numeric_limits<T>::max()};
    Point max = Point{cuda::std::numeric_limits<T>::lowest()};

    explicit AABB() = default;
    explicit AABB(const Point& min, const Point& max) : min(min), max(max) {};
    ~AABB() = default;

    /// Check whether the AABB is valid, that is if min <= max.
    __host__ __device__ bool valid() const { return min.x <= max.x && min.y <= max.y; }

    __host__ __device__ glm::vec<LENGTH, T> size() const { return max - min; }

    __host__ __device__ void expand(const Point& point)
    {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    /// Compute the squared distance between the AABB and the given point.
    __host__ __device__ float sq_distance(const Point& p) const
    {
        // Source:
        // https://gamedev.stackexchange.com/a/156877
        float d = 0.0f;
        for (int i = 0; i < LENGTH; i++) {
            // For each axis count any excess distance outside box extents
            float v = p[i];
            if (v < min[i]) d += (min[i] - v) * (min[i] - v);
            if (v > max[i]) d += (v - max[i]) * (v - max[i]);
        }
        return d;
    }

    /// Test if the AABB contains the given point.
    __host__ __device__ bool test_point(const Point& p) const
    {
        return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z;
    }

    /// Test if the AABB intersects with the given sphere.
    __host__ __device__ bool test_sphere(const Point& c, float r) const
    {
        // Source:
        // https://gamedev.stackexchange.com/a/156877
        return sq_distance(c) <= r * r;
    }

    /// Test if the given segment intersects the AABB.
    /// \param segment
    /// \param tmin the t of the first intersection
    /// \param tmax the t of the last intersection
    __host__ __device__ bool test_segment(const Segment<LENGTH, T>& segment, float& tmin, float& tmax) const
    {
        tmin = -FLT_MAX;
        tmax = FLT_MAX;
        // #pragma unroll ?
        for (int d = 0; d < LENGTH; d++) {
            T dir = segment.p1[d] - segment.p0[d];
            if (dir != 0) {
                float t1 = float(min[d] - segment.p0[d]) / float(dir);
                float t2 = float(max[d] - segment.p0[d]) / float(dir);
                if (t1 > t2) swap(t1, t2);
                tmin = glm::max(t1, tmin);
                tmax = glm::min(t2, tmax);
                if (tmin > tmax) return false;
            }
        }
        // Line intersecting but segment is not (fully outside)
        if (tmin > 1.0f || tmax < 0.0f) return false;
        if (tmin < 0.0f) tmin = 0.0f;
        if (tmax > 1.0f) tmax = 1.0f;
        return true;
    }

    __host__ __device__ bool contains(const AABB<LENGTH, T>& aabb) const
    {
        // TODO check
        for (int i = 0; i < LENGTH; i++) {
            if (max[i] < aabb.min[i] || min[i] > aabb.max[i]) return false;
        }
        return true;
    }
};

using Segment2i = Segment<2, int>;

using AABB2i = AABB<2, int>;
using AABB3i = AABB<3, int>;
using AABB3f = AABB<3, float>;

} // namespace recogs