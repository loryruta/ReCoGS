#include <filesystem>

#include "Image.h"
#include "image_visit_transform.h"
#include "utils/exceptions.h"
#include "utils/misc_utils.h"
#include "utils/stb_image_write.h"

namespace gs_train
{
/// Save the image to the provided file as PNG.
/// If the image values are floating-point they must be in [0, 1] range.
///
/// \param image the image to save
/// \param out_filepath the output filepath
template <int C>
void image_save_png(const Image<C, uint8_t, ImageMemoryLayout::HWC>& image, const std::filesystem::path& out_filepath)
{
    static_assert(C == 3 || C == 4, "Unsupported number of channels");

    printf("[DEBUG] [Image/save] Saving image (%d, %d) (comp: %d, type: %s) to: %s\n",
           image.width,
           image.height,
           C,
           typeid(uint8_t).name(),
           out_filepath.c_str());
    std::vector<uint8_t> image_data;
    image.to_host(image_data);
    stbi_write_png(out_filepath.c_str(), (int) image.width, (int) image.height, C, image_data.data(), image.width * C);
}

/// Given a RGB/RGBA floating-point image with values [0, 1] and memory layout CHW,
/// convert it to a 8-bit image with values [0, 255].
template <int C>
void image_save_png(const Image<C, float, ImageMemoryLayout::CHW>& image, const std::filesystem::path& out_filepath)
{
    static_assert(C == 3 || C == 4, "Unsupported number of channels");
    using ImageCfCHW = Image<C, float, ImageMemoryLayout::CHW>;
    using ImageCuHWC = Image<C, uint8_t, ImageMemoryLayout::HWC>;
    // Allocate the u8 image
    ImageCuHWC image_u8 = ImageCuHWC::malloc(image.width, image.height);
    // Convert [0, 1] FP values to [0, 255] values
    image_visit(
        image,
        [image_u8] __device__(const ImageCfCHW& image, uint32_t x, uint32_t y) mutable {
            auto val = image.value(x, y);
            auto val_u8 = typename ImageCuHWC::Value(glm::clamp(val * 255.99f, 0.0f, 255.0f));
            image_u8.set_value(x, y, val_u8);
            return 0; // TODO returning a int (that is ignored), check (image_visit_transform)!
        },
        CU_STREAM_LEGACY);
    image_save_png<C>(image_u8, out_filepath);
}

/// Save the image to the provided file as PFM (Portable FloatMap).
template <int C>
void image_save_pfm(Image<C, float, ImageMemoryLayout::HWC>& image, const std::filesystem::path& out_filepath)
{
    FILE* f = fopen(out_filepath.c_str(), "wb");
    CHECK_STATE(f, "Can't open file: %s", out_filepath);

    if constexpr (C == 1) {
        fputs("Pf\n", f);
    } else if (C == 3 || (C == 4 && image.store_alpha)) {
        fputs("PF\n", f);
    } else {
        throw IllegalArgumentException("Only supporting R or RGB images");
    }

    std::string dim_str = std::to_string(image.width) + " " + std::to_string(image.height) + "\n";
    fputs(dim_str.c_str(), f);
    fputs("-1.0\n", f);

    std::vector<float> image_data;
    image.to_host(image_data);
    fwrite(image_data.data(), sizeof(float), image.height * image.width * C, f);
    fclose(f);
}
} // namespace gs_train
