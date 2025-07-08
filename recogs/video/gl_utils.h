#pragma once

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <span>

#include <glad/glad.h>

#include "utils/exceptions.h"
#include "utils/misc_utils.h"
#include "utils/stb_image.h"

namespace recogs
{
/// A RAII wrapper for GL shader
struct Shader {
    GLuint handle;
    std::string name;

    explicit Shader(GLenum type, std::string name = "untitled") : handle(glCreateShader(type)), name(name) {};
    Shader(const Shader&) = delete;
    Shader(Shader&& other) noexcept
    {
        handle = other.handle;
        other.handle = 0;
    }
    ~Shader()
    {
        if (handle) glDeleteShader(handle);
    }

    void source_from_str(const std::string& src_str)
    {
        printf("[DEBUG] [GL/Shader] Attaching source to \"%s\"...\n", name.c_str());
        const char* src_ptr = src_str.c_str();
        glShaderSource(handle, 1, &src_ptr, nullptr);
    }

    std::string get_info_log()
    {
        GLint log_length = 0;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &log_length);

        std::vector<GLchar> log(log_length);
        glGetShaderInfoLog(handle, log_length, nullptr, log.data());
        return {log.begin(), log.end()};
    }

    void compile()
    {
        printf("[DEBUG] [GL/Shader] Compiling \"%s\"...\n", name.c_str());
        glCompileShader(handle);
        GLint status;
        glGetShaderiv(handle, GL_COMPILE_STATUS, &status);
        if (!status) {
            printf("[ERROR] Shader failed to compile: %s\n", get_info_log().c_str());
            CHECK_STATE(status);
        }
    }
};

/// A RAII wrapper for GL program
struct Program {
    GLuint handle;

    explicit Program() { handle = glCreateProgram(); };
    Program(const Program&) = delete;
    Program(Program&& other) noexcept
    {
        handle = other.handle;
        other.handle = 0;
    }
    ~Program()
    {
        if (handle) glDeleteProgram(handle);
    }

    void attach_shader(GLuint shader_handle) { glAttachShader(handle, shader_handle); }
    void attach_shader(const Shader& shader) { glAttachShader(handle, shader.handle); }

    [[nodiscard]] std::string get_info_log() const
    {
        GLint log_length = 0;
        glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &log_length);

        std::vector<GLchar> log(log_length);
        glGetProgramInfoLog(handle, log_length, nullptr, log.data());
        return {log.begin(), log.end()};
    }

    void link()
    {
        GLint status;
        glLinkProgram(handle);
        glGetProgramiv(handle, GL_LINK_STATUS, &status);
        if (!status) {
            printf("[ERROR] Program failed to link: %s\n", get_info_log().c_str());
            CHECK_STATE(status);
        }
    }

    void use() { glUseProgram(handle); }

    GLint get_uniform_location(const char* uniform_name)
    {
        GLint loc = glGetUniformLocation(handle, uniform_name);
        if (loc < 0) {
            printf("[ERROR] Failed to get uniform location: %s\n", uniform_name);
            CHECK_STATE(loc >= 0);
        }
        return loc;
    }
};

inline GLuint load_texture(const std::filesystem::path& filepath, int channels, GLint minmag_filter = GL_LINEAR)
{
    // Load texture from file (using stbi)
    int width, height;
    int channels_in_file;
    uint8_t* texture_data = stbi_load(filepath.c_str(), &width, &height, &channels_in_file, channels);
    CHECK_ARG(texture_data, "Invalid texture file: {}", filepath.string());
    // Create texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    GLint internalformat = channels == 3 ? GL_RGB8 : GL_RGBA8;
    GLint format = channels == 3 ? GL_RGB : GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, width, height, 0, format, GL_UNSIGNED_BYTE, texture_data);
    stbi_image_free(texture_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minmag_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, minmag_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

struct Framebuffer {
    glm::ivec2 resolution;
    GLuint handle;
    /* Texture references */
    std::array<GLuint, 32> color_attachments{};
    GLuint depth_texture = 0;
    /* Renderbuffers */
    GLuint renderbuffer = 0; ///< Owned renderbuffer for depth

    explicit Framebuffer(glm::ivec2 resolution);
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& other) noexcept;
    ~Framebuffer();

    void set_name(const char* name);
    void bind();

    void attach_color(int slot, GLuint texture);
    void attach_depth_texture(GLuint texture);
    void create_and_attach_depth_renderbuffer();

    void blit_from(const Framebuffer& other, GLbitfield mask, GLenum filter = GL_NEAREST);

    void draw_buffers(const std::vector<GLenum>& slots);

    GLenum status();
};
} // namespace recogs
