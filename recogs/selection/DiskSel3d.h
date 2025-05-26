#pragma once

#include <memory>

#include "GSCamera.h"
#include "Sel3d.h"
#include "disk/DiskRasterizer.h"
#include "disk/Disks.h"
#include "utils/image/Image.h"

namespace recogs
{
class DiskPcdSel3d : public Sel3d
{
private:
    std::unique_ptr<Disks> m_disks;

public:
    explicit DiskPcdSel3d();
    ~DiskPcdSel3d() = default;

    void append(const Image1u8& fill_mask,
                const GSCamera& camera,
                const Image4fHWC& color_depth,
                cudaStream_t stream) override;

    void clear(const Image1u8& edit_mask,
               const GSCamera& camera,
               const Image4fHWC& color_depth,
               cudaStream_t stream) override;

    void project(const GSCamera& camera,
                 Image4fHWC& color_depth,
                 const FillPixelT& fill_pixel_fn,
                 cudaStream_t stream) override;
};
} // namespace recogs
