#include "GSCamera.h"

#include <glm/gtc/type_ptr.hpp>

using namespace gs_train;

GSCamera::GSCamera()
{
    m_viewmatrix.resize(16 * sizeof(float));
    m_projmatrix.resize(16 * sizeof(float));
    m_campos.resize(3 * sizeof(float));
}

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

    m_viewmatrix.upload(glm::value_ptr(viewmatrix), 16);
    m_projmatrix.upload(glm::value_ptr(viewproj), 16);
    m_campos.upload(glm::value_ptr(position), 3);
}
