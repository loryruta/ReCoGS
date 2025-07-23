#pragma once

#include <memory>

#include "Camera.h"
#include "utils/image/Image.h"

BEGIN_NAMESPACE

// Forward decl
class Scene;
class GSRasterizer;

struct DepthEstimatorParams {
    const float* background_d;
    const Scene* scene;
    GSRasterizer* gs_rasterizer;
    Image4fHWC* inout_colordepth;
    const Camera* camera;
    float b = 0.07f;
    bool debug = false;
    bool overwrite_depth = false;
    cudaStream_t stream;

    virtual void validate() const;
};

/// Complete list of implementations of depth estimators.
enum class DepthEstimatorType : int { //
    Gaussians = 0,
    PCVNetHV,
    FoundationStereo
};

[[nodiscard]] inline const char* DepthEstimatorType_name(DepthEstimatorType type)
{
    switch (type) {
    case DepthEstimatorType::Gaussians:
        return "Gaussians";
    case DepthEstimatorType::PCVNetHV:
        return "PCVNetHV";
    case DepthEstimatorType::FoundationStereo:
        return "FoundationStereo";
    default:
        throw IllegalArgumentException("Invalid DepthEstimatorType");
    }
}

/// Abstract class for depth estimators:
/// given a view of a 3DGS scene, estimate the depth at that view.
class DepthEstimator
{
public:
    virtual ~DepthEstimator() = default;

    [[nodiscard]] virtual DepthEstimatorType type() const = 0;

    /// Thread-safe function to perform depth estimation.
    /// The output depth is written as the 4-th component of <code>params.im0</code>.
    virtual void estimate(const DepthEstimatorParams& params) = 0; // TODO thread-safe

    // ----------------------------------------------------------------
    /* Singleton */
    // ----------------------------------------------------------------

    static DepthEstimator& get();
    static void set(DepthEstimatorType type);

protected:
    explicit DepthEstimator() = default;
};

/// Simple depth estimator implementation:
/// compute the depth as a weighted sum of gaussians depths along the ray.
class Gaussians_DepthEstimator : public DepthEstimator
{
public:
    explicit Gaussians_DepthEstimator() = default;
    ~Gaussians_DepthEstimator() override = default;

    [[nodiscard]] DepthEstimatorType type() const override { return DepthEstimatorType::Gaussians; }

    void estimate(const DepthEstimatorParams& params) override
    {
        params.validate();
        // Nothing to do, im0 4-th component is already the depth computed that way
    }
};

END_NAMESPACE
