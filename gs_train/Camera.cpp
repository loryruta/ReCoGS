#include "Camera.h"


using namespace gs_train;

void Camera::update() {
    intrinsic_params[0][0] = fx * width * 0.5f;
    intrinsic_params[1][1] = fy * height * 0.5f;
    intrinsic_params[2][2] = 1.0f;
    intrinsic_params[2][0] = width * 0.5f;
    intrinsic_params[2][1] = height * 0.5f;
}

void Camera::from_json(nlohmann::json data) {
    position.x = data["position"][0].get<float>();
    position.y = data["position"][1].get<float>();
    position.z = data["position"][2].get<float>();
    rotation[0][0] = data["rotation"][0][0].get<float>();
    rotation[1][0] = data["rotation"][0][1].get<float>();
    rotation[2][0] = data["rotation"][0][2].get<float>();
    rotation[0][1] = data["rotation"][1][0].get<float>();
    rotation[1][1] = data["rotation"][1][1].get<float>();
    rotation[2][1] = data["rotation"][1][2].get<float>();
    rotation[0][2] = data["rotation"][2][0].get<float>();
    rotation[1][2] = data["rotation"][2][1].get<float>();
    rotation[2][2] = data["rotation"][2][2].get<float>();
    fx = data["fx"].get<float>();
    fy = data["fy"].get<float>();
    width = data["fx"].get<float>();
    height = data["fx"].get<float>();
}
