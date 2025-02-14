#pragma once

#include <memory>

#include "Image.h"
#include "utils/exceptions.h"
#include "utils/stb_image.h"

namespace gs_train
{
template <int C, typename T>
void image_load(const char* filepath, std::unique_ptr<Image<C, T, ImageMemoryLayout::HWC>>& out_image)
{
    static_assert(C == 3 || C == 4, "Image must be RGB or RGBA");

    int width, height;
    int components_in_file;
    void* image_data;
    if constexpr (std::is_same_v<T, float>) {
        image_data = stbi_loadf(filepath, &width, &height, &components_in_file, C);
    } else if (std::is_same_v<T, uint8_t>) {
        image_data = stbi_load(filepath, &width, &height, &components_in_file, C);
    } else {
        throw IllegalArgumentException("Unsupported image type (either uint8_t or float)");
    }

    /* GPU upload */
    using ImageT = Image<C, T, ImageMemoryLayout::HWC>;
    out_image = std::make_unique<ImageT>(ImageT::malloc(width, height));
    CHECK_CUDA(cudaMemcpy(out_image->data_d(), image_data, width * height * C * sizeof(T), cudaMemcpyHostToDevice));
}
} // namespace gs_train
