#include <typeinfo>

#include "Image.h"
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
void image_save_png(const Image<C, uint8_t, ImageMemoryLayout::HWC>& image, const char* out_filepath)
{
    static_assert(C == 1 || C == 3 || C == 4, "Unsupported number of channels");

    printf("[DEBUG] [Image/save] Saving image (%d, %d) (comp: %d, type: %s) to: %s\n",
           image.width,
           image.height,
           C,
           typeid(uint8_t).name(),
           out_filepath);
    std::vector<uint8_t> image_data;
    image.to_host(image_data);
    stbi_write_png(out_filepath, (int) image.width, (int) image.height, C, image_data.data(), image.width * C);
}

/// Save the image to the provided file as PFM (Portable FloatMap).
template <int C>
void image_save_pfm(Image<C, float, ImageMemoryLayout::HWC>& image, const char* out_filepath)
{
    FILE* f = fopen(out_filepath, "wb");
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

    float* image_data = image_download(image);
    fwrite(image_data, sizeof(float), image.height * image.width * C, f);
    fclose(f);

    free(image_data);
}
} // namespace gs_train
