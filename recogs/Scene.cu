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

size_t Scene::num_bytes() const
{
    size_t bytes = means.size() + shs.size() + opacities.size() + scales.size() + rotations.size();
    bytes += shs_2.size();
    bytes += dL_dmean2D.size() + dL_dconic.size() + dL_dopacity.size() + dL_dcolor.size() + dL_dmean3D.size() +
             dL_dcov3D.size() + dL_dsh.size() + dL_dscale.size() + dL_drot.size();
    bytes *= sizeof(float);
    return bytes;
}
