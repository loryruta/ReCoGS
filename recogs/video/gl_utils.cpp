#include "gl_utils.h"

using namespace recogs;

Framebuffer::Framebuffer(glm::ivec2 resolution) : resolution(resolution) { glGenFramebuffers(1, &handle); }

Framebuffer::Framebuffer(Framebuffer&& other) noexcept
    : resolution(other.resolution), handle(other.handle), color_attachments(other.color_attachments),
      depth_texture(other.depth_texture), renderbuffer(other.renderbuffer)
{
    other.handle = 0;
    other.depth_texture = 0;
}

Framebuffer::~Framebuffer()
{
    if (renderbuffer) glDeleteRenderbuffers(1, &renderbuffer);
    if (handle) glDeleteFramebuffers(1, &handle);
}

void Framebuffer::set_name(const char* name) { glObjectLabel(GL_FRAMEBUFFER, handle, -1, name); }

void Framebuffer::bind() { glBindFramebuffer(GL_FRAMEBUFFER, handle); }

void Framebuffer::attach_color(int slot, GLuint texture)
{
    CHECK_ARG(slot < 32);
    glBindFramebuffer(GL_FRAMEBUFFER, handle);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + slot, GL_TEXTURE_2D, texture, 0);
    color_attachments[slot] = texture;
}

void Framebuffer::attach_depth_texture(GLuint texture)
{
    glBindFramebuffer(GL_FRAMEBUFFER, handle);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);
    depth_texture = texture;
}

void Framebuffer::create_and_attach_depth_renderbuffer()
{
    glGenRenderbuffers(1, &renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, resolution.x, resolution.y);

    glBindFramebuffer(GL_FRAMEBUFFER, handle);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
}

void Framebuffer::blit_from(const Framebuffer& other, GLbitfield mask, GLenum filter)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, other.handle);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, handle);
    // clang-format off
    glBlitFramebuffer(0, 0, other.resolution.x, other.resolution.y,
                      0, 0, resolution.x, resolution.y,
                      mask, filter);
    // clang-format on
}

void Framebuffer::draw_buffers(const std::vector<GLenum>& draw_buffers)
{
    glBindFramebuffer(GL_FRAMEBUFFER, handle);
    glDrawBuffers((GLsizei) draw_buffers.size(), draw_buffers.data());
}

GLenum Framebuffer::status()
{
    glBindFramebuffer(GL_FRAMEBUFFER, handle);
    return glCheckFramebufferStatus(GL_FRAMEBUFFER);
}
