#pragma once

#include <imgui.h>

#include "utils/imgui_utils.h"
#include "video/gl_utils.h"

namespace recogs
{
namespace ui
{
class StereoTest
{
private:
    GLuint m_ht_stereo_icon;
    GLuint m_vt_stereo_icon;
    GLuint m_hv_stereo_icon;

public:
    int button_size = 20;

    enum { Capture_NONE = 0, Capture_HORIZONTAL = 1, Capture_VERTICAL = 2, Capture_HV = 3 };
    /// Which type of stereo test has to be captured the next frame.
    int capture = Capture_NONE;
    /// The currently displayed capture.
    int current_capture = Capture_NONE;

    explicit StereoTest();
    ~StereoTest();

    void ui();
};
} // namespace ui

} // namespace recogs
