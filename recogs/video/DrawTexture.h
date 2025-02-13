#pragma once

#include <glad/glad.h>

#include "gl_utils.h"

namespace gs_train
{
/// Draw a texture to screen
class DrawTexture
{
private:
    Program m_program;
    GLuint m_vao{}; // Required to avoid "Array object is not active" (even if not used)

public:
    explicit DrawTexture();
    DrawTexture(const DrawTexture&) = delete;
    DrawTexture(const DrawTexture&&) = delete;
    ~DrawTexture();

    /// Draw the texture at the given region
    void draw(GLuint texture, int x, int y, int width, int height);
};
} // namespace gslab
