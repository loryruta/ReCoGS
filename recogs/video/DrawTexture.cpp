#include "DrawTexture.h"

using namespace recogs;

namespace
{
const char* k_vert_shader_src = R"(#version 460 core

    out vec2 v_texcoord;

    void main()
    {
        const vec2 k_texcoords[] = vec2[](
            vec2(0.0, 0.0), // 0
            vec2(1.0, 0.0), // 1
            vec2(0.0, 1.0), // 2
            vec2(1.0, 0.0), // 1
            vec2(1.0, 1.0), // 3
            vec2(0.0, 1.0)  // 2
        );
        const vec2 k_vertices[] = vec2[](
            vec2(-1.0, 1.0),  // 0
            vec2(1.0, 1.0),   // 1
            vec2(-1.0, -1.0), // 2
            vec2(1.0, 1.0),   // 1
            vec2(1.0, -1.0),  // 3
            vec2(-1.0, -1.0)  // 2
        );
        gl_Position = vec4(k_vertices[gl_VertexID], 0, 1);
        v_texcoord = k_texcoords[gl_VertexID];
    }
)";

const char* k_frag_shader_src = R"(#version 460 core

    in vec2 v_texcoord;

    uniform sampler2D u_texture; // RGB

    layout(location = 0) out vec4 f_color;

    void main()
    {
        vec3 rgb = texture(u_texture, v_texcoord).rgb;
        f_color = vec4(rgb, 1);
    }
)";
} // namespace

DrawTexture::DrawTexture()
{
    // Create shader program
    Shader vertex_shader(GL_VERTEX_SHADER);
    vertex_shader.source_from_str(k_vert_shader_src);
    vertex_shader.compile();

    Shader fragment_shader(GL_FRAGMENT_SHADER);
    fragment_shader.source_from_str(k_frag_shader_src);
    fragment_shader.compile();

    m_program.attach_shader(vertex_shader);
    m_program.attach_shader(fragment_shader);
    m_program.link();

    // Create vertex array
    glGenVertexArrays(1, &m_vao);
}

DrawTexture::~DrawTexture() { glDeleteVertexArrays(1, &m_vao); }

void DrawTexture::draw(GLuint texture, int x, int y, int width, int height)
{
    glViewport(0, 0, width, height);

    glDisable(GL_DEPTH_TEST);

    m_program.use();

    glBindVertexArray(m_vao);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // TODO restore old viewport
}
