#pragma once

#include <memory>
#include <filesystem>

#include "Image.h"
#include "utils/exceptions.h"
#include "utils/image/image_cast.h"
#include "utils/stb_image.h"

namespace gs_train
{
template <int C, typename T>
void image_load(const std::filesystem::path& filepath, std::unique_ptr<Image<C, T, ImageMemoryLayout::HWC>>& out_image)
{
    static_assert(C == 3 || C == 4, "Image must be RGB or RGBA");
    CHECK_ARG(std::filesystem::exists(filepath), "Image file not found: %s\n", filepath);

    int width, height;
    int components_in_file;
    void* image_data;
    if constexpr (std::is_same_v<T, float>) {
        image_data = stbi_loadf(filepath.c_str(), &width, &height, &components_in_file, C);
    } else if (std::is_same_v<T, uint8_t>) {
        image_data = stbi_load(filepath.c_str(), &width, &height, &components_in_file, C);
    } else {
        throw IllegalArgumentException("Unsupported image type (either uint8_t or float)");
    }

    /* GPU upload */
    using ImageT = Image<C, T, ImageMemoryLayout::HWC>;
    out_image = std::make_unique<ImageT>(ImageT::malloc(width, height));
    CHECK_CUDA(cudaMemcpy(out_image->data_d(), image_data, width * height * C * sizeof(T), cudaMemcpyHostToDevice));
}

template <int C, typename T>
void image_load_chw(const std::filesystem::path& filepath,
                    std::unique_ptr<Image<C, T, ImageMemoryLayout::CHW>>& out_image_chw)
{
    using OutImageT = Image<C, T, ImageMemoryLayout::CHW>;

    // Load the image in HWC format (only possible by stbi_image)
    std::unique_ptr<Image<C, T, ImageMemoryLayout::HWC>> image_hwc;
    image_load<C, T>(filepath, image_hwc);

    // Memory layout transfer HWC -> CHW
    OutImageT image_chw = OutImageT::malloc(image_hwc->width, image_hwc->height);
    image_cast(*image_hwc, image_chw);
    out_image_chw = std::make_unique<OutImageT>(std::move(image_chw));
}
} // namespace gs_train
