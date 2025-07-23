#include "DepthEstimatorWindow.h"

#include "App.h"

USING_NAMESPACE
using namespace recogs::ui;

void GaussiansWindow::ui()
{
    ImGui::SeparatorText("Gaussians");

    ImGui::Checkbox("Show depth", &show_depth);
}

void PCVNetHVWindow::ui()
{
    ImGui::SeparatorText("PCVNetHV");

    ImGui::Text("Axis:");
    using Axis = PCVNetHV_DepthEstimatorParams::Axis;
    ImGui::RadioButton("Horizontal", (int*) &axis, (int) Axis::Axis_Horizontal);
    ImGui::SameLine();
    ImGui::RadioButton("Vertical", (int*) &axis, (int) Axis::Axis_Vertical);
    ImGui::SameLine();
    ImGui::RadioButton("Horizontal+Vertical", (int*) &axis, (int) Axis::Axis_Both);

    if (ImGui::Button("Estimate")) {
        estimate = true;
    }
}

void FoundationStereoWindow::ui()
{
    ImGui::SeparatorText("Foundation Stereo");

    if (ImGui::Button("Estimate")) {
        estimate = true;
    }
}

void DepthEstimatorWindow::ui()
{
    if (ImGui::Begin("Depth Estimator")) {
        ImGui::SliderFloat("Scene depth epsilon", &scene_depth_epsilon, -0.4f, 0.4f, "%.3f");

        ImGui::Text("Render transform:");
        ImGui::RadioButton("Color", (int*) &render_transform, (int) RenderTransform::Color);
        ImGui::SameLine();
        ImGui::RadioButton("Depthmap", (int*) &render_transform, (int) RenderTransform::DepthMap);
        ImGui::SameLine();
        ImGui::RadioButton("Normal map", (int*) &render_transform, (int) RenderTransform::NormalMap);

        DepthEstimatorType method = DepthEstimator::get().type();
        ImGui::Text("Method:");
        if (ImGui::RadioButton("Gaussians", (int*) &method, (int) DepthEstimatorType::Gaussians)) {
            g_app->enqueue_job([]() { DepthEstimator::set(DepthEstimatorType::Gaussians); });
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("PCVNetHV", (int*) &method, (int) DepthEstimatorType::PCVNetHV)) {
            g_app->enqueue_job([]() { DepthEstimator::set(DepthEstimatorType::PCVNetHV); });
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Foundation Stereo", (int*) &method, (int) DepthEstimatorType::FoundationStereo)) {
            g_app->enqueue_job([]() { DepthEstimator::set(DepthEstimatorType::FoundationStereo); });
        }

        if (method == DepthEstimatorType::Gaussians) {
            gaussians.ui();
        } else if (method == DepthEstimatorType::PCVNetHV) {
            pcvnethv.ui();
        } else if (method == DepthEstimatorType::FoundationStereo) {
            foundation_stereo.ui();
        }
    }
    ImGui::End();
}
