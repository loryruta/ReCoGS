#pragma once

#include <memory>

#include <onnxruntime_cxx_api.h>

#include "utils/image/Image.h"

namespace gs_train
{
// Forward decl
class App;

class PCVNet
{
public:
    using ImageT = Image<3, float, ImageMemoryLayout::CHW>;
    using Image1fCHW = Image<1, float, ImageMemoryLayout::CHW>;

private:
    App& m_app;

    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;

public:
    explicit PCVNet(App& app);
    ~PCVNet() = default;

    /// \param im0 The left image
    /// \param im1 The right image
    /// \param disparity_map the output depth map
    void forward(const ImageT& im0, const ImageT& im1, Image1fCHW& disparity_map);
};

} // namespace gs_train
