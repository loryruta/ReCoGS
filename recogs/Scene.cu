#include "Scene.h"

using namespace recogs;

Scene::Scene(int num_vertices) : num_vertices(num_vertices)
{
    means.resize(num_vertices * 3);
    shs.resize(num_vertices * 16 * 3);
    opacities.resize(num_vertices);
    scales.resize(num_vertices * 3);
    rotations.resize(num_vertices * 4);

    shs_2.resize(num_vertices * 16 * 3);
}

void Scene::prepare_for_training()
{
    shs_2.resize(num_vertices * 16 * 3);

    dL_dmean2D.resize(num_vertices * 3); // float3? See code...
    dL_dconic.resize(num_vertices * 2 * 2);
    dL_dopacity.resize(num_vertices);
    dL_dcolor.resize(num_vertices * 3);  // glm::vec3*
    dL_dmean3D.resize(num_vertices * 3); // glm::vec3*
    dL_dcov3D.resize(num_vertices * 6);
    dL_dsh.resize(num_vertices * 16 * 3);
    dL_dscale.resize(num_vertices * 3);
    dL_drot.resize(num_vertices * 4);
}

void Scene::zero_grad(cudaStream_t stream)
{
    CHECK_STATE(is_prepared_for_training(), "Scene is not trainable");
    thrust::fill(thrust::cuda::par.on(stream), dL_dmean2D.begin(), dL_dmean2D.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dconic.begin(), dL_dconic.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dopacity.begin(), dL_dopacity.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dmean3D.begin(), dL_dmean3D.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dcov3D.begin(), dL_dcov3D.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dsh.begin(), dL_dsh.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_dscale.begin(), dL_dscale.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), dL_drot.begin(), dL_drot.end(), 0);
}

void Scene::compute_minmax(cudaStream_t stream)
{
    if (has_computed_minmax()) return;
    const glm::vec3* begin = (glm::vec3*) RCGS_TPTR(means);
    const glm::vec3* end = begin + num_vertices;
    auto min_op = [] __device__(const glm::vec3& a, const glm::vec3& b) -> glm::vec3 { return glm::min(a, b); };
    auto max_op = [] __device__(const glm::vec3& a, const glm::vec3& b) -> glm::vec3 { return glm::max(a, b); };
    m_min = thrust::reduce(thrust::cuda::par.on(stream), begin, end, glm::vec3(INFINITY), min_op);
    m_max = thrust::reduce(thrust::cuda::par.on(stream), begin, end, glm::vec3(-INFINITY), max_op);
    // clang-format off
    printf("[INFO ] [Scene] Min/max computed: (%f, %f, %f) -> (%f, %f, %f)\n",
           m_min.x, m_min.y, m_min.z,
           m_max.x, m_max.y, m_max.z);
    // clang-format on
}

glm::vec3 Scene::min() const
{
    CHECK_STATE(has_computed_minmax());
    return m_min;
}

glm::vec3 Scene::max() const
{
    CHECK_STATE(has_computed_minmax());
    return m_max;
}

size_t Scene::num_bytes() const
{
    size_t bytes = means.size() + shs.size() + opacities.size() + scales.size() + rotations.size();
    bytes += shs_2.size();
    bytes += dL_dmean2D.size() + dL_dconic.size() + dL_dopacity.size() + dL_dcolor.size() + dL_dmean3D.size() +
             dL_dcov3D.size() + dL_dsh.size() + dL_dscale.size() + dL_drot.size();
    bytes *= sizeof(float);
    return bytes;
}
