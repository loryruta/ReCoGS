#include "GSCamera.h"

#include <glm/gtc/type_ptr.hpp>

using namespace gs_train;

GSCamera::GSCamera()
{
    m_viewmatrix.resize(16);
    m_projmatrix.resize(16);
    m_campos.resize(3);
}

glm::mat4 GSCamera::viewmatrix() const { return glm::inverse(inv_view()); }

glm::mat3 GSCamera::K() const
{
    float hw = float(width) * 0.5f;
    float hh = float(height) * 0.5f;
    glm::mat3 K{};
    K[0][0] = fx * hw;
    K[1][1] = fy * hh;
    K[2][2] = 1.0f;
    K[2][0] = hw;
    K[2][1] = hh;
    return K;
}

glm::mat4 GSCamera::inv_view() const
{
    glm::mat4 world2cam = glm::mat3_cast(rotation);
    world2cam[3] = glm::vec4(position, 1.0f);
    return world2cam;
}

glm::mat3 GSCamera::inv_K() const { return glm::inverse(K()); }

void GSCamera::set_resolution(int width_, int height_)
{
    fx *= float(width_) / float(width);
    fy *= float(height_) / float(height);
    width = width_;
    height = height_;
}

void GSCamera::update()
{
    /* View matrix */
    glm::mat4 world2cam = glm::mat3_cast(rotation);
    world2cam[3] = glm::vec4(position, 1.0f);
    glm::mat4 viewmatrix = glm::inverse(world2cam);

    /* Projection matrix */
    const float znear = 0.01f;
    const float zfar = 1000.0f;
    float top = tan_fovy() * znear;
    float bottom = -top;
    float right = tan_fovx() * znear;
    float left = -right;
    const float zsign = 1.0f;
    glm::mat4 projmatrix{};
    projmatrix[0][0] = 2.0f * znear / (right - left);
    projmatrix[1][1] = 2.0f * znear / (top - bottom);
    projmatrix[2][0] = (right + left) / (right - left);
    projmatrix[2][1] = (top + bottom) / (top - bottom);
    projmatrix[2][3] = zsign;
    projmatrix[2][2] = zsign * zfar / (zfar - znear);
    projmatrix[3][2] = -(zfar * znear) / (zfar - znear);
    glm::mat4 viewproj = projmatrix * viewmatrix;

    thrust::copy(glm::value_ptr(viewmatrix), glm::value_ptr(viewmatrix) + 16, m_viewmatrix.begin());
    thrust::copy(glm::value_ptr(viewproj), glm::value_ptr(viewproj) + 16, m_projmatrix.begin());
    thrust::copy(glm::value_ptr(position), glm::value_ptr(position) + 3, m_campos.begin());
}
