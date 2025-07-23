#pragma once

#include <imgui.h>

#include "depth_estimators/DepthEstimator.h"
#include "depth_estimators/PCVNetHV_DepthEstimator.h"
#include "utils/imgui_utils.h"
#include "video/gl_utils.h"

BEGIN_NAMESPACE

// Forward decl
class MainScreen;

namespace ui
{
/// Window for Gaussians depth estimation settings.
struct GaussiansWindow {
    bool show_depth = false;

    void ui();
};

/// Window for PCVNetHV depth estimation settings.
struct PCVNetHVWindow {
    /// The axis where to perform stereo matching, either horizontal, vertical or both.
    PCVNetHV_DepthEstimatorParams::Axis axis = PCVNetHV_DepthEstimatorParams::Axis::Axis_Both;
    bool estimate = false;

    void ui();
};

/// Window for Foundation Stereo depth estimation settings.
struct FoundationStereoWindow {
    bool estimate = false;

    void ui();
};

/// A window to choose between the depth estimation method to use.
struct DepthEstimatorWindow {
    enum class RenderTransform : int { Color, DepthMap, NormalMap } render_transform;
    float scene_depth_epsilon = 0.f;

    /* UI */
    GaussiansWindow gaussians;
    PCVNetHVWindow pcvnethv;
    FoundationStereoWindow foundation_stereo;

    void ui();
};

} // namespace ui

END_NAMESPACE
