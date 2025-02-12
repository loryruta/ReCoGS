#pragma once

#include <cstdlib>
#include <string>
#include <vector>

#include <glad/glad.h>

#include "utils/misc_utils.h"

namespace gs_train
{
/// A RAII wrapper for GL shader
struct Shader {
    GLuint handle;

    explicit Shader(GLenum type) : handle(glCreateShader(type)) {};
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
} // namespace gslab