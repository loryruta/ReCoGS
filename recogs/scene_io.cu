#include "scene_io.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "utils/misc_utils.h"

using namespace gs_train;

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

Scene gs_train::read_scene_from_ply(const std::filesystem::path& scene_ply)
{
    std::ifstream is(scene_ply, std::ios::binary);

    std::string line;
    std::string str;

    int num_vertices = 0;

    // clang-format off
    std::getline(is, line); require_line(line, "ply");
    std::getline(is, line); require_line(line, "format binary_little_endian 1.0");
    std::getline(is, line);
    {
        // Line: element vertex 1026508
        std::stringstream ss(line);
        ss >> str; CHECK_STATE(str == "element");
        ss >> str; CHECK_STATE(str == "vertex");
        ss >> num_vertices;
    }
    std::getline(is, line); require_line(line, "property float x");
    std::getline(is, line); require_line(line, "property float y");
    std::getline(is, line); require_line(line, "property float z");
    std::getline(is, line); require_line(line, "property float nx");
    std::getline(is, line); require_line(line, "property float ny");
    std::getline(is, line); require_line(line, "property float nz");
    std::getline(is, line); require_line(line, "property float f_dc_0");
    std::getline(is, line); require_line(line, "property float f_dc_1");
    std::getline(is, line); require_line(line, "property float f_dc_2");
    std::getline(is, line); require_line(line, "property float f_rest_0");
    std::getline(is, line); require_line(line, "property float f_rest_1");
    std::getline(is, line); require_line(line, "property float f_rest_2");
    std::getline(is, line); require_line(line, "property float f_rest_3");
    std::getline(is, line); require_line(line, "property float f_rest_4");
    std::getline(is, line); require_line(line, "property float f_rest_5");
    std::getline(is, line); require_line(line, "property float f_rest_6");
    std::getline(is, line); require_line(line, "property float f_rest_7");
    std::getline(is, line); require_line(line, "property float f_rest_8");
    std::getline(is, line); require_line(line, "property float f_rest_9");
    std::getline(is, line); require_line(line, "property float f_rest_10");
    std::getline(is, line); require_line(line, "property float f_rest_11");
    std::getline(is, line); require_line(line, "property float f_rest_12");
    std::getline(is, line); require_line(line, "property float f_rest_13");
    std::getline(is, line); require_line(line, "property float f_rest_14");
    std::getline(is, line); require_line(line, "property float f_rest_15");
    std::getline(is, line); require_line(line, "property float f_rest_16");
    std::getline(is, line); require_line(line, "property float f_rest_17");
    std::getline(is, line); require_line(line, "property float f_rest_18");
    std::getline(is, line); require_line(line, "property float f_rest_19");
    std::getline(is, line); require_line(line, "property float f_rest_20");
    std::getline(is, line); require_line(line, "property float f_rest_21");
    std::getline(is, line); require_line(line, "property float f_rest_22");
    std::getline(is, line); require_line(line, "property float f_rest_23");
    std::getline(is, line); require_line(line, "property float f_rest_24");
    std::getline(is, line); require_line(line, "property float f_rest_25");
    std::getline(is, line); require_line(line, "property float f_rest_26");
    std::getline(is, line); require_line(line, "property float f_rest_27");
    std::getline(is, line); require_line(line, "property float f_rest_28");
    std::getline(is, line); require_line(line, "property float f_rest_29");
    std::getline(is, line); require_line(line, "property float f_rest_30");
    std::getline(is, line); require_line(line, "property float f_rest_31");
    std::getline(is, line); require_line(line, "property float f_rest_32");
    std::getline(is, line); require_line(line, "property float f_rest_33");
    std::getline(is, line); require_line(line, "property float f_rest_34");
    std::getline(is, line); require_line(line, "property float f_rest_35");
    std::getline(is, line); require_line(line, "property float f_rest_36");
    std::getline(is, line); require_line(line, "property float f_rest_37");
    std::getline(is, line); require_line(line, "property float f_rest_38");
    std::getline(is, line); require_line(line, "property float f_rest_39");
    std::getline(is, line); require_line(line, "property float f_rest_40");
    std::getline(is, line); require_line(line, "property float f_rest_41");
    std::getline(is, line); require_line(line, "property float f_rest_42");
    std::getline(is, line); require_line(line, "property float f_rest_43");
    std::getline(is, line); require_line(line, "property float f_rest_44");
    std::getline(is, line); require_line(line, "property float opacity");
    std::getline(is, line); require_line(line, "property float scale_0");
    std::getline(is, line); require_line(line, "property float scale_1");
    std::getline(is, line); require_line(line, "property float scale_2");
    std::getline(is, line); require_line(line, "property float rot_0");
    std::getline(is, line); require_line(line, "property float rot_1");
    std::getline(is, line); require_line(line, "property float rot_2");
    std::getline(is, line); require_line(line, "property float rot_3");
    std::getline(is, line); require_line(line, "end_header");
    // clang-format on

    /* Read scene data */
    std::vector<float> means(num_vertices * 3);
    std::vector<float> normals(num_vertices * 3);
    std::vector<float> shs(num_vertices * 16 * 3);
    std::vector<float> opacities(num_vertices);
    std::vector<float> scales(num_vertices * 3);
    std::vector<float> rotations(num_vertices * 4);
    for (size_t i = 0; i < num_vertices; ++i) {
        is.read((char*) (means.data() + i * 3), 3 * sizeof(float));
        is.read((char*) (normals.data() + i * 3), 3 * sizeof(float));
        is.read((char*) (shs.data() + i * 48), 48 * sizeof(float));
        is.read((char*) (opacities.data() + i), sizeof(float));
        is.read((char*) (scales.data() + i * 3), 3 * sizeof(float));
        is.read((char*) (rotations.data() + i * 4), 4 * sizeof(float));
    }

    /* Upload to GPU */
    Scene scene{};
    scene.num_vertices = num_vertices;
    scene.means.fit_data(means.data(), num_vertices * 3);
    // scene.normals.fit_data(normals.data(), num_vertices * 3);
    scene.shs.fit_data(shs.data(), num_vertices * 16 * 3);
    scene.opacities.fit_data(opacities.data(), num_vertices);
    scene.scales.fit_data(scales.data(), num_vertices * 3);
    scene.rotations.fit_data(rotations.data(), num_vertices * 4);
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
