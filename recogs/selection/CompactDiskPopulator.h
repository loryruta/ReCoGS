#pragma once

#include "Camera.h"
#include "ClusterIntGrid.h"
#include "disk/DiskBuffer.h"
#include "utils/image/Image.h"

namespace recogs
{
class CompactDiskPopulator
{
public:
    explicit CompactDiskPopulator() = default;
    ~CompactDiskPopulator() = default;

    std::vector<glm::ivec2> _identify_statistical_outliers(const Image1u8& fill_mask,
                                                           const Camera& camera,
                                                           const Image4fHWC& color_depth,
                                                           cudaStream_t stream);

    void _create_disk_at_aabb(const ClusterIntGrid_AABB& aabb,
                              const Camera& camera,
                              int W,
                              const std::vector<float>& depths,
                              const std::vector<float>& normals,
                              Disk& out_disk);

    std::unique_ptr<DiskBuffer> populate(const Image1u8& fill_mask,
                                         const Camera& camera,
                                         const Image4fHWC& color_depth,
                                         const Image4fHWC& normal_map,
                                         cudaStream_t stream);
};

} // namespace recogs