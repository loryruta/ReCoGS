#include "scene_io.h"

#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <miniply.h>

#include "utils/exceptions.h"
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

Scene recogs::read_scene_from_ply(const std::filesystem::path& scene_ply)
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
    scene.name = scene_ply.stem().string();
    thrust::copy(means.begin(), means.end(), scene.means.begin());
    // thrust::copy(normals.begin(), normals.end(), scene.normals.begin());
    thrust::copy(shs.begin(), shs.end(), scene.shs.begin());
    thrust::copy(shs.begin(), shs.end(), scene.shs_2.begin());
    thrust::copy(opacities.begin(), opacities.end(), scene.opacities.begin());
    thrust::copy(scales.begin(), scales.end(), scene.scales.begin());
    thrust::copy(rotations.begin(), rotations.end(), scene.rotations.begin());
    return scene;
}

void recogs::write_scene_to_ply(const Scene& scene, const std::filesystem::path& scene_ply)
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

void deserialize_camera(const nlohmann::json& json, CameraData& out_camera)
{
    out_camera.width = json["width"];
    out_camera.height = json["height"];
    // Position
    for (int i = 0; i < 3; ++i) {
        out_camera.position[i] = json["position"][i];
    }
    // Rotation
    glm::mat3 rot_mat{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rot_mat[c][r] = json["rotation"][r][c];
        }
    }
    out_camera.rotation = glm::quat_cast(rot_mat); // Rotation matrix -> Quaternion
    out_camera.fx = json["fx"];
    out_camera.fy = json["fy"];
}

void recogs::read_cameras_from_json(const std::filesystem::path& scene_folder, std::vector<CameraData>& out_cameras)
{
    std::filesystem::path cameras_json = scene_folder / "cameras.json";
    if (!std::filesystem::exists(cameras_json)) {
        printf("[WARN ] [scene_io] cameras.json not found\n");
        return;
    }
    printf("[INFO ] [scene_io] Loading cameras at: %s\n", cameras_json.c_str());
    std::ifstream file(cameras_json);
    nlohmann::json json = nlohmann::json::parse(file);
    for (const nlohmann::json& camera_json : json) {
        CameraData& camera_data = out_cameras.emplace_back();
        deserialize_camera(camera_json, camera_data);
    }
    printf("[INFO ] [scene_io] %zu cameras loaded\n", out_cameras.size());
}

ColmapOutputReader::ColmapOutputReader() {}

std::unordered_map<uint32_t, ColmapCamera>
ColmapOutputReader::read_cameras_(const std::filesystem::path& camerasbin_filepath)
{
    std::ifstream is(camerasbin_filepath);
    CHECK_STATE(is.is_open(), "Cannot open file: %s", camerasbin_filepath.c_str());

    uint64_t num_cameras;
    is.read((char*) &num_cameras, sizeof(uint64_t));
    printf("[DEBUG] [ColmapOutputReader] Read %" PRIu64 " cameras...\n", num_cameras);

    std::unordered_map<uint32_t, ColmapCamera> cameras;
    cameras.reserve(num_cameras);
    for (int i = 0; i < num_cameras; ++i) {
        ColmapCamera camera{};

        is.read((char*) &camera.camera_id, sizeof(uint32_t));
        is.read((char*) &camera.model_id, sizeof(uint32_t));
        is.read((char*) &camera.width, sizeof(uint64_t));
        is.read((char*) &camera.height, sizeof(uint64_t));

        printf("[DEBUG] [ColmapOutputReader] Camera %d (Model %d): Width: %" PRIu64 ", Height: %" PRIu64 "\n",
               camera.camera_id,
               camera.model_id,
               camera.width,
               camera.height);

        // Model ID:
        // CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
        // CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
        // CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
        // CameraModel(model_id=3, model_name="RADIAL", num_params=5),
        // CameraModel(model_id=4, model_name="OPENCV", num_params=8),
        // CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
        // CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
        // CameraModel(model_id=7, model_name="FOV", num_params=5),
        // CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
        // CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
        // CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),

        if (camera.model_id == 4) {
            const size_t num_params = 8;
            is.read((char*) &camera.params[0], num_params * sizeof(double));
        } else {
            throw IllegalArgumentException("Model ID not supported"); // TODO model_id
        }

        cameras.emplace(camera.camera_id, camera);
    }
    return cameras;
}

std::vector<ColmapImage> ColmapOutputReader::read_images(const std::filesystem::path& imagesbin_filepath)
{
    std::ifstream is(imagesbin_filepath);
    CHECK_STATE(is.is_open(), "Cannot open file: %s", imagesbin_filepath.c_str());

    uint64_t num_images;
    is.read((char*) &num_images, sizeof(uint64_t));
    printf("[DEBUG] [ColmapOutputReader] Read %" PRIu64 " images...\n", num_images);

    std::vector<ColmapImage> cameras;
    cameras.reserve(num_images);
    for (int i = 0; i < num_images; ++i) {
        ColmapImage& image = cameras.emplace_back();

        is.read((char*) &image.image_id, sizeof(uint32_t));
        is.read((char*) &image.R[0], 4 * sizeof(double));
        is.read((char*) &image.t[0], 3 * sizeof(double));
        is.read((char*) &image.camera_id, sizeof(uint32_t));
        std::getline(is, image.image_name, '\0');
        uint64_t num_points;
        is.read((char*) &num_points, sizeof(uint64_t));
        image.point_xy_ids.resize(num_points);
        is.read((char*) image.point_xy_ids.data(), std::streamsize(num_points * sizeof(image.point_xy_ids[0])));

        printf("[DEBUG] [ColmapOutputReader] Image %d; Camera: %d, Name: %s, Num. keypoints: %zu\n",
               image.image_id,
               image.camera_id,
               image.image_name.c_str(),
               num_points);
    }
    return cameras;
}

std::pair<glm::mat4, float>
ColmapOutputReader::read_dataparser_transforms(const std::filesystem::path& dataparser_transforms_filepath)
{
    std::ifstream file(dataparser_transforms_filepath);
    CHECK_STATE(file.is_open());
    nlohmann::json json = nlohmann::json::parse(file);
    float scale = json["scale"];
    glm::mat4 transform = glm::identity<glm::mat4>();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            transform[c][r] = json["transform"][r][c];
        }
    }
    return {transform, scale};
}

std::vector<GSCamera> ColmapOutputReader::read_cameras(const std::filesystem::path& output_folder)
{
    // Reference:
    // https://colmap.github.io/format.html#binary-file-format
    // https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py

    // NOTE: textual format not supported!

    std::filesystem::path camerasbin_filepath = output_folder / "cameras.bin";
    std::filesystem::path imagesbin_filepath = output_folder / "images.bin";
    std::filesystem::path points3d_filepath = output_folder / "points3D.bin";

    auto colmap_cameras_by_id = read_cameras_(camerasbin_filepath);
    std::vector<ColmapImage> colmap_images = read_images(imagesbin_filepath);

    auto [transform, scale] =
        read_dataparser_transforms("/home/loryruta/Desktop/custom_splats/felicien_1/outputs/.processed_data/splatfacto/"
                                   "2025-03-17_095944/dataparser_transforms.json");

    std::vector<GSCamera> cameras;
    cameras.reserve(colmap_images.size());
    for (const ColmapImage& image : colmap_images) {
        // ---------------------------------------------------------------- Read camera pose
        GSCamera& camera = cameras.emplace_back();

        glm::quat R = glm::quat(float(image.R[0]), float(image.R[1]), float(image.R[2]), float(image.R[3]));
        glm::vec3 t = glm::make_vec3(image.t);

        // Source:
        // https://github.com/nerfstudio-project/nerfstudio/blob/73fe54dda0b743616854fc839889d955522e0e68/nerfstudio/data/dataparsers/colmap_dataparser.py#L160-L170
        glm::mat4 w2c = glm::mat3_cast(R);
        w2c[3] = glm::vec4(t, 1);
        glm::mat4 c2w = glm::inverse(w2c);
        // TODO don't transpose - re-transpose
        c2w[1] *= -1;
        c2w[2] *= -1;
        // if self.config.assume_colmap_world_coordinate_convention:
        // Swap 2nd with 1st row
        std::swap(c2w[0][1], c2w[0][2]);
        std::swap(c2w[1][1], c2w[1][2]);
        std::swap(c2w[2][1], c2w[2][2]);
        std::swap(c2w[3][1], c2w[3][2]);
        c2w[0][2] *= -1;
        c2w[1][2] *= -1;
        c2w[2][2] *= -1;
        c2w[3][2] *= -1;

        // Apply dataparser JSON transforms
        glm::mat4 c2w_ = (transform * glm::mat4(c2w)) * scale;

        camera.rotation = glm::quat_cast(glm::transpose(glm::mat3(c2w_)));
        camera.rotation = glm::normalize(camera.rotation);
        camera.position = c2w_[3];

        // ---------------------------------------------------------------- Read camera intrinsics (shared)
        ColmapCamera& colmap_camera = colmap_cameras_by_id.at(image.camera_id);
        camera.width = (int) colmap_camera.width;
        camera.height = (int) colmap_camera.height;
        // OpenCV camera model params:
        // fx, fy, cx, cy, k1, k2, p1, p2
        // Reference:
        // https://github.com/colmap/colmap/blob/6556b4e28fba070e15894833b31f66de6cf4c6e1/src/colmap/sensor/models.h#L317
        camera.fx = (float) colmap_camera.params[0];
        camera.fy = (float) colmap_camera.params[1];
        // cx, cy derived from width and height
        // Ignoring distortion parameters
    }
    return cameras;
}

std::vector<GSCamera> NerfStudioOutputReader::read_cameras(const std::filesystem::path& output_dir)
{
    std::unordered_map<uint32_t, ColmapCamera> colmap_cameras = m_colmap_reader.read_cameras_(
        "/home/loryruta/Desktop/custom_splats/felicien_1/.processed_data/colmap/sparse/0/cameras.bin");
    ColmapCamera colmap_camera = colmap_cameras.at(1);

    std::filesystem::path transforms_train_filepath =
        "/home/loryruta/Desktop/custom_splats/felicien_1/output/transforms_train.json";
    std::ifstream is(transforms_train_filepath);
    nlohmann::json json = nlohmann::json::parse(is);
    std::vector<GSCamera> cameras{};
    for (nlohmann::json pose_json : json) {
        std::string filepath = pose_json["file_path"];
        printf("[DEBUG] [NerfStudioOutputReader] File path: %s\n", filepath.c_str());
        auto transform_json = pose_json["transform"];
        glm::mat4 transform = glm::identity<glm::mat4>();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                transform[c][r] = transform_json[r][c];
            }
        }
        GSCamera& camera = cameras.emplace_back();
        camera.position = -transform[3];
        camera.rotation = glm::quat_cast(glm::transpose(glm::mat3(transform)));
        camera.width = (int) colmap_camera.width;
        camera.height = (int) colmap_camera.height;
        CHECK_STATE(colmap_camera.model_id == 4, "Only OPENCV camera model is supported");
        camera.fx = (float) colmap_camera.params[0];
        camera.fy = (float) colmap_camera.params[1];
        //        camera.cx = (float) colmap_camera.params[2];
        //        camera.cy = (float) colmap_camera.params[3];
    }
    return cameras;
}
