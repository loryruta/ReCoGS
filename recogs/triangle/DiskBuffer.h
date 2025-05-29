#pragma once

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace recogs
{
struct Disk {
    glm::vec4 position;
    glm::vec2 scale;
    glm::vec4 rotation;
};

class DiskBuffer
{
private:
    GLuint m_vao = 0;
    GLuint m_vbo = 0;

public:
    std::vector<Disk> disks{};

    DiskBuffer() = default;
    DiskBuffer(const DiskBuffer&) = delete;
    DiskBuffer(DiskBuffer&& other) noexcept : disks(std::move(other.disks)), m_vbo(other.m_vbo) { other.m_vbo = 0; }
    ~DiskBuffer()
    {
        if (m_vao) glDeleteVertexArrays(1, &m_vao), m_vao = 0;
        if (m_vbo) glDeleteBuffers(1, &m_vbo), m_vbo = 0;
    }

    [[nodiscard]] bool empty() const { return disks.empty(); }
    [[nodiscard]] size_t size() const { return disks.size(); }
    Disk& emplace_back() { return disks.emplace_back(); }

    [[nodiscard]] GLuint vao() const { return m_vao; }
    [[nodiscard]] GLuint vbo() const { return m_vbo; }

    void upload()
    {
        if (m_vao) glDeleteVertexArrays(1, &m_vao);
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        if (m_vbo) glDeleteBuffers(1, &m_vbo);
        glGenBuffers(1, &m_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr) (disks.size() * sizeof(Disk)), disks.data(), GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // Position
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Disk), (void*) offsetof(Disk, position));
        glEnableVertexAttribArray(1); // Scale
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Disk), (void*) offsetof(Disk, scale));
        glEnableVertexAttribArray(2); // Rotation
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Disk), (void*) offsetof(Disk, rotation));
        glVertexAttribDivisor(0, 1); // Update vertex attribute every instance
        glVertexAttribDivisor(1, 1); // Update vertex attribute every instance
        glVertexAttribDivisor(2, 1); // Update vertex attribute every instance
    }
};
} // namespace recogs
