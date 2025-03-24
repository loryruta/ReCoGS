#pragma once

#include <functional>
#include <vector>

#include <glad/glad.h>

namespace gs_train
{
class ImageSlider
{
private:
    const int m_N;
    std::vector<GLuint> m_textures;
    int m_start_texture_idx = 0;
    int m_end_texture_idx = -1; // Exclusive; Dynamically updated

public:
    float button_width = 50.0f;
    int image_resolution_xy = 100; /// Slider images are always squares
    std::function<void(int)> on_image_click;

    explicit ImageSlider(int N, int resolution_xy);
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
};
} // namespace gs_train
