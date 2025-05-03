#pragma once

#include <filesystem>
#include <memory>

#include "Image.h"
#include "utils/exceptions.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_visit_transform.h"
#include "utils/stb_image.h"

namespace recogs
{
template <int C> // T = float
void image_load(const std::filesystem::path& filepath,
                std::unique_ptr<Image<C, float, ImageMemoryLayout::HWC>>& out_image,
                cudaStream_t stream)
{
    using OutImage = Image<C, float, ImageMemoryLayout::HWC>;

    // Load the image to uint8
    std::unique_ptr<Image<C, uint8_t, ImageMemoryLayout::HWC>> image_u8;
    image_load<C>(filepath, image_u8, stream);
    // Transform the uint8 image to float
    out_image = std::make_unique<OutImage>(OutImage::malloc(image_u8->width, image_u8->height));
    image_visit(
        *out_image,
        [image_u8_ = *image_u8.get()] __device__(OutImage & img, int x, int y) mutable {
            glm::vec<C, float> value = glm::clamp(glm::vec<C, float>(image_u8_.value(x, y)) / 255.0f, 0.0f, 1.0f);
            img.set_value(x, y, value);
            return 0; // TODO temporary
        },
        stream);
    // Wait for the uint8->float cast to be done so we can safely delete image_u8
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <int C> // T = uint8_t
void image_load(const std::filesystem::path& filepath,
                std::unique_ptr<Image<C, uint8_t, ImageMemoryLayout::HWC>>& out_image,
                cudaStream_t stream)
{
    using OutImage = Image<C, uint8_t, ImageMemoryLayout::HWC>;

    int width, height;
    int comps;
    const uint8_t* u8_data = stbi_load(filepath.c_str(), &width, &height, &comps, C);
    uint8_t* u8_data_d;
    CHECK_CUDA(cudaMallocAsync(&u8_data_d, width * height * C * sizeof(uint8_t), stream));
    CHECK_CUDA(
        cudaMemcpyAsync(u8_data_d, u8_data, width * height * C * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    out_image = std::make_unique<OutImage>(OutImage::ref(width, height, u8_data_d));
    out_image->owned = true;
}

template <int C, typename T /* uint8_t | float */>
void image_load_chw(const std::filesystem::path& filepath,
                    std::unique_ptr<Image<C, T, ImageMemoryLayout::CHW>>& out_image_chw,
                    cudaStream_t stream)
{
    using OutImageT = Image<C, T, ImageMemoryLayout::CHW>;

    // Load the image in HWC format (only possible by stbi_image)
    std::unique_ptr<Image<C, T, ImageMemoryLayout::HWC>> image_hwc;
    image_load<C>(filepath, image_hwc, stream); // Synchronous

    // Memory layout transfer HWC -> CHW
    OutImageT image_chw = OutImageT::malloc(image_hwc->width, image_hwc->height);
    image_cast(*image_hwc, image_chw, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    out_image_chw = std::make_unique<OutImageT>(std::move(image_chw));
}
} // namespace recogs
