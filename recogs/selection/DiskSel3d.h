#pragma once

#include <memory>

#include "GSCamera.h"
#include "Sel3d.h"
#include "triangle/DiskBuffer.h"
#include "utils/image/Image.h"

namespace recogs
{
class DiskPcdSel3d : public Sel3d
{
private:
    DiskBuffer m_disk_buffer;

public:
    explicit DiskPcdSel3d();
    ~DiskPcdSel3d() = default;

    [[nodiscard]] const DiskBuffer& disk_buffer() const { return m_disk_buffer; }

    void
    append(const Image1u8& fill_mask, const GSCamera& camera, const Image4fHWC& depthmap, cudaStream_t stream) override;

    void clear(const Image1u8& clear_mask, const GSCamera& camera, cudaStream_t) override;

    void project(const GSCamera& camera,
                 const Image4fHWC& depthmap,
                 Image1u8& sel3d_mask,
                 cudaStream_t stream) const override;
};
} // namespace recogs
