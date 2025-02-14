#pragma once

#include "utils/DeviceBuffer.h"

namespace gs_train
{
struct Scene {
    int num_vertices;
    DeviceBuffer means{"Scene/means"};
    //DeviceBuffer normals{"Scene/normals"};
    DeviceBuffer shs{"Scene/shs"};
    DeviceBuffer opacities{"Scene/opacities"};
    DeviceBuffer scales{"Scene/scales"};
    DeviceBuffer rotations{"Scene/rotations"};
};
} // namespace gs_train
