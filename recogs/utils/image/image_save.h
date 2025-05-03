#include <filesystem>

#include "Image.h"
#include "image_visit_transform.h"
#include "utils/exceptions.h"
#include "utils/misc_utils.h"
#include "utils/stb_image_write.h"

namespace recogs
{
/// Save the image to the provided PNG or JPEG file.
/// Device operations are performed on the provided stream, towards which this function eventually synchronizes.
template <int C>
void image_save(const Image<C, uint8_t, ImageMemoryLayout::HWC>& image,
                const std::filesystem::path& out_filepath,
                cudaStream_t stream)
{
    static_assert(C == 3 || C == 4, "Unsupported number of channels");
    std::string ext = out_filepath.extension();
    CHECK_ARG(ext == ".png" || ext == ".jpg" || ext == ".jpeg", "Unrecognized image extension: %s", ext.c_str());

    std::vector<uint8_t> image_data(image.width * image.height * C);
    CHECK_CUDA(cudaMemcpyAsync(image_data.data(),
                               image.data_d(),
                               image.width * image.height * C * sizeof(uint8_t),
                               cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("[DEBUG] [Image/save] Saving image (%d, %d) (comp: %d, type: %s) to: %s\n",
           image.width,
           image.height,
           C,
           typeid(uint8_t).name(),
           out_filepath.c_str());

    if (ext == ".png") {
        stbi_write_png(
            out_filepath.c_str(), (int) image.width, (int) image.height, C, image_data.data(), image.width * C);
    } else if (ext == ".jpg" || ext == ".jpeg") {
        stbi_write_jpg(
            out_filepath.c_str(), (int) image.width, (int) image.height, C, image_data.data(), image.width * C);
    } else {
        assert(false);
    }
}

/// Save a float image to the provided PNG or JPEG file.
/// Device operations are performed on the provided stream, towards which this function eventually synchronizes.
template <int C, ImageMemoryLayout MEMORY_LAYOUT>
void image_save(const Image<C, float, MEMORY_LAYOUT>& image,
                const std::filesystem::path& out_filepath,
                cudaStream_t stream)
{
    static_assert(C == 3 || C == 4, "Unsupported number of channels");
    using ImageCf = Image<C, float, MEMORY_LAYOUT>;
    using ImageCu = Image<C, uint8_t, ImageMemoryLayout::HWC>;
    ImageCu image_u8 = ImageCu::malloc(image.width, image.height);
    image_visit(
        image,
        [image_u8] __device__(const ImageCf& image, uint32_t x, uint32_t y) mutable {
            glm::vec<C, float> val = image.value(x, y);
            glm::vec<C, uint8_t> val_u8 = typename ImageCf::Value(glm::clamp(val * 255.99f, 0.0f, 255.0f));
            image_u8.set_value(x, y, val_u8);
            return 0; // TODO
        },
        stream);
    image_save<C>(image_u8, out_filepath, stream);
}

/// Save a float image to the provided PFM file (Portable FloatMap).
/// Device operations are performed on the provided stream, towards which this function eventually synchronizes.
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
} // namespace recogs
