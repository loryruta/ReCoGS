#include "scene_io.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include <miniply.h>

#include "utils/misc_utils.h"

using namespace recogs;

namespace
{
void require_line(const std::string& actual, const std::string& expected)
{
    if (actual != expected) // TODO unlikely
    {
        fprintf(stderr, "[ERROR] Expected line \"%s\", but got \"%s\"\n", expected.c_str(), actual.c_str());
        exit(1);
    }
}
} // namespace

std::vector<uint32_t> miniply_find_properties_indices(const miniply::PLYElement& element,
                                                      const std::vector<std::string>& property_names)
{
    std::vector<uint32_t> indices;
    indices.reserve(property_names.size());
    for (const std::string& property_name : property_names) {
        uint32_t i = element.find_property(property_name.c_str());
        CHECK_ARG(i != miniply::kInvalidIndex, "Cannot find property: %s", property_name.c_str());
        indices.emplace_back(i);
    }
    return indices;
}

Scene gs_train::read_scene_from_ply(const std::filesystem::path& scene_ply)
{
    miniply::PLYReader reader(scene_ply.c_str());
    CHECK_STATE(reader.valid(), "Can't parse .ply file: %s", scene_ply.c_str());
    CHECK_STATE(reader.has_element() && reader.element_is(miniply::kPLYVertexElement) && reader.load_element());
    const miniply::PLYElement& element = *reader.element();
    int N = (int) element.count;
    printf("[DEBUG] [scene_io] Total count of vertices: %d\n", N);

    std::vector<float> means(N * 3);
    std::vector<float> shs(N * 16 * 3);
    std::vector<float> opacities(N);
    std::vector<float> scales(N * 3);
    std::vector<float> rotations(N * 4);

    auto xyz_indices = miniply_find_properties_indices(element, {"x", "y", "z"});
    auto opacity_indices = miniply_find_properties_indices(element, {"opacity"});
    auto scale_indices = miniply_find_properties_indices(element, {"scale_0", "scale_1", "scale_2"});
    auto rot_indices = miniply_find_properties_indices(element, {"rot_0", "rot_1", "rot_2", "rot_3"});

    // clang-format off
    reader.extract_properties(xyz_indices.data(), xyz_indices.size(), miniply::PLYPropertyType::Float, means.data());
    reader.extract_properties(opacity_indices.data(), opacity_indices.size(), miniply::PLYPropertyType::Float, opacities.data());
    reader.extract_properties(scale_indices.data(), scale_indices.size(), miniply::PLYPropertyType::Float, scales.data());
    reader.extract_properties(rot_indices.data(), rot_indices.size(), miniply::PLYPropertyType::Float, rotations.data());
    // clang-format on

    // R channel: f_dc_0, f_rest_0,  ... f_rest_14 (r_indices)
    // G channel: f_dc_1, f_rest_15, ... f_rest_29 (g_indices)
    // B channel: f_dc_2, f_rest_30, ... f_rest_44 (b_indices)
    std::vector<uint32_t> shcoeff_r_indices;

    size_t num_sh_coeff_components = 3;
    for (const miniply::PLYProperty& property : element.properties) {
        if (property.name.starts_with("f_rest_")) {
            ++num_sh_coeff_components;
        }
    }
    assert(num_sh_coeff_components % 3 == 0);
    size_t num_sh_coeffs = num_sh_coeff_components / 3;
    printf("[DEBUG] [scene_io] Found %zu SH coefficients (%zu total components)\n",
           num_sh_coeffs,
           num_sh_coeff_components);

    for (int sh_coeff = 0; sh_coeff < num_sh_coeffs; ++sh_coeff) {
        if (sh_coeff == 0) {
            // f_dc_
            auto f_dc_indices = miniply_find_properties_indices(element, {"f_dc_0", "f_dc_1", "f_dc_2"});
            printf("[DEBUG] [scene_io] Extracting: f_dc_0, f_dc_1, f_dc_2\n");
            size_t offset = 0;
            size_t stride = num_sh_coeffs * 3 * sizeof(float);
            reader.extract_properties_with_stride(
                f_dc_indices.data(), f_dc_indices.size(), miniply::PLYPropertyType::Float, shs.data() + offset, stride);
        } else {
            // f_rest_
            std::string f_rest_r = "f_rest_" + std::to_string(sh_coeff - 1);
            std::string f_rest_g = "f_rest_" + std::to_string(sh_coeff - 1 + (num_sh_coeffs - 1));
            std::string f_rest_b = "f_rest_" + std::to_string(sh_coeff - 1 + (num_sh_coeffs - 1) * 2);
            printf("[DEBUG] [scene_io] Extracting: %s, %s, %s\n", f_rest_r.c_str(), f_rest_g.c_str(), f_rest_b.c_str());
            auto f_rest_ = miniply_find_properties_indices(element, {f_rest_r, f_rest_g, f_rest_b});
            size_t offset = 3;
            size_t stride = num_sh_coeffs * 3 * sizeof(float);
            reader.extract_properties_with_stride(
                f_rest_.data(), f_rest_.size(), miniply::PLYPropertyType::Float, shs.data() + offset, stride);
        }
    }

    // Upload to GPU
    Scene scene(N);
    thrust::copy(means.begin(), means.end(), scene.means.begin());
    // thrust::copy(normals.begin(), normals.end(), scene.normals.begin());
    thrust::copy(shs.begin(), shs.end(), scene.shs.begin());
    thrust::copy(shs.begin(), shs.end(), scene.shs_2.begin());
    thrust::copy(opacities.begin(), opacities.end(), scene.opacities.begin());
    thrust::copy(scales.begin(), scales.end(), scene.scales.begin());
    thrust::copy(rotations.begin(), rotations.end(), scene.rotations.begin());
    return scene;
}

void gs_train::write_scene_to_ply(const Scene& scene, const std::filesystem::path& scene_ply)
{
    std::ofstream of(scene_ply, std::ios::binary);
    of << "ply" << "\n";
    of << "format binary_little_endian 1.0\n";
    of << "element vertex " << scene.num_vertices << "\n";
    of << "property float x\n";
    of << "property float y\n";
    of << "property float z\n";
    of << "property float nx\n";
    of << "property float ny\n";
    of << "property float nz\n";
    of << "property float f_dc_0\n";
    of << "property float f_dc_1\n";
    of << "property float f_dc_2\n";
    of << "property float f_rest_0\n";
    of << "property float f_rest_1\n";
    of << "property float f_rest_2\n";
    of << "property float f_rest_3\n";
    of << "property float f_rest_4\n";
    of << "property float f_rest_5\n";
    of << "property float f_rest_6\n";
    of << "property float f_rest_7\n";
    of << "property float f_rest_8\n";
    of << "property float f_rest_9\n";
    of << "property float f_rest_10\n";
    of << "property float f_rest_11\n";
    of << "property float f_rest_12\n";
    of << "property float f_rest_13\n";
    of << "property float f_rest_14\n";
    of << "property float f_rest_15\n";
    of << "property float f_rest_16\n";
    of << "property float f_rest_17\n";
    of << "property float f_rest_18\n";
    of << "property float f_rest_19\n";
    of << "property float f_rest_20\n";
    of << "property float f_rest_21\n";
    of << "property float f_rest_22\n";
    of << "property float f_rest_23\n";
    of << "property float f_rest_24\n";
    of << "property float f_rest_25\n";
    of << "property float f_rest_26\n";
    of << "property float f_rest_27\n";
    of << "property float f_rest_28\n";
    of << "property float f_rest_29\n";
    of << "property float f_rest_30\n";
    of << "property float f_rest_31\n";
    of << "property float f_rest_32\n";
    of << "property float f_rest_33\n";
    of << "property float f_rest_34\n";
    of << "property float f_rest_35\n";
    of << "property float f_rest_36\n";
    of << "property float f_rest_37\n";
    of << "property float f_rest_38\n";
    of << "property float f_rest_39\n";
    of << "property float f_rest_40\n";
    of << "property float f_rest_41\n";
    of << "property float f_rest_42\n";
    of << "property float f_rest_43\n";
    of << "property float f_rest_44\n";
    of << "property float opacity\n";
    of << "property float scale_0\n";
    of << "property float scale_1\n";
    of << "property float scale_2\n";
    of << "property float rot_0\n";
    of << "property float rot_1\n";
    of << "property float rot_2\n";
    of << "property float rot_3\n";
    of << "end_header\n";

    // TODO :'(
    //    for (size_t i = 0; i < scene.num_vertices; ++i) {
    //        of.write((const char*) &scene.means[i * 3], 3 * sizeof(float));
    //        of.write((const char*) &scene.normals[i * 3], 3 * sizeof(float));
    //        of.write((const char*) &scene.shs[i * 16 * 3], 16 * 3 * sizeof(float));
    //        of.write((const char*) &scene.opacities[i], sizeof(float));
    //        of.write((const char*) &scene.scales[i * 3], 3 * sizeof(float));
    //        of.write((const char*) &scene.rotations[i * 4], 4 * sizeof(float));
    //    }
}

std::vector<GSCamera> gs_train::read_cameras_from_json(const std::filesystem::path& scene_folder, cudaStream_t stream)
{
    std::filesystem::path cameras_json = scene_folder / "cameras.json";
    if (!std::filesystem::exists(cameras_json)) {
        printf("[WARN ] [scene_io] cameras.json not found\n");
        return {};
    }

    printf("[INFO ] [scene_io] Loading cameras at: %s\n", cameras_json.c_str());
    std::ifstream file(cameras_json);
    nlohmann::json json = nlohmann::json::parse(file);
    std::vector<GSCamera> cameras;
    cameras.reserve(json.size());
    for (const nlohmann::json& camera_json : json) {
        GSCamera& camera = cameras.emplace_back();
        camera.deserialize(camera_json);
        camera.update(stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("[INFO ] [scene_io] %zu cameras loaded\n", cameras.size());
    return cameras;
}
