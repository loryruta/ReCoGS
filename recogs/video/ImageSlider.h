#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <glad/glad.h>

#include "utils/Stopwatch.h"

namespace recogs
{
class ImageSlider
{
private:
    constexpr static double k_slide_accel_elapsed_time = 0.2;

    const int m_N;
    std::vector<GLuint> m_textures;
    int m_start_texture_idx = 0;
    int m_end_texture_idx = -1; // Exclusive; Dynamically updated
    std::optional<Stopwatch> m_slid_left_at;
    std::optional<Stopwatch> m_slid_right_at;

public:
    /// The width of the arrow buttons to scroll the previews.
    float button_width = 50.0f;
    std::function<void(int)> on_image_click;    /// Function called when the user clicks an image of the slider
    std::function<GLuint(int)> provide_texture; /// Function called to generate the texture

    /// \param N the total number of images
    explicit ImageSlider(int N);
    ~ImageSlider() = default;

    /// Retrieve a constant reference of the i-th texture.
    [[nodiscard]] const GLuint& texture(int i) const { return m_textures.at(i); }
    /// Retrieve a reference of the i-th texture.
    [[nodiscard]] GLuint& texture(int i) { return m_textures.at(i); }

    /// Get the index of the first visible texture.
    [[nodiscard]] int start_texture_index() const { return m_start_texture_idx; };
    /// Get the index of the first non-visible texture, after visible ones.
    [[nodiscard]] int end_texture_index() const { return m_end_texture_idx; };

    void ui();

private:
    void init_texture_if_uninit(int i);
};
} // namespace recogs
