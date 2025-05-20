#include "SVOScreen.h"

#include "App.h"

using namespace recogs;

SVOScreen::SVOScreen(App& app, std::shared_ptr<SVO> svo) : m_app(app)
{
    if (svo) {
        m_svo = std::move(svo);
    } else {
        m_svo = create_simple_svo();
    }
    m_camera_controller = std::make_unique<GSCameraController>(m_app.window(), m_camera);
}

void SVOScreen::resize(int width, int height) {}

void SVOScreen::update(float dt)
{
    if (m_camera_controller) {
        bool updated = m_camera_controller->update(dt);
        if (updated) m_camera.update(m_app.stream());
        // m_camera.log_info();
    }
}

void SVOScreen::render(Image4fHWC& out_color_depth)
{
    if (m_svo) {
        m_app.svo_renderer().render(*m_svo, m_camera, out_color_depth, m_app.stream());
    }
}

void SVOScreen::ui() { m_camera_controller->ui(); }

std::shared_ptr<SVO> SVOScreen::create_simple_svo()
{
    std::shared_ptr<SVO> svo = std::make_shared<SVO>();
    svo->min = {0, 0, 0};
    svo->max = {10, 10, 10};

    std::vector<SVONode> svo_nodes;
    auto add_node = [&](uint32_t children_mask, uint32_t children_offset, bool leaf = false) {
        SVONode& node = svo_nodes.emplace_back();
        node.children_mask = children_mask;
        node.first_child_offset = children_offset | (leaf ? 0 : 0x80000000);
    };
    add_node(0b00000001, 1);
    add_node(0b10000001, 2);
    add_node(0b10011001, 4);
    add_node(0b10000110, 8);
    add_node(0b00000000, 545, true);
    add_node(0b00000000, 636, true);
    add_node(0b00000000, 678, true);
    add_node(0b00000000, 777, true);
    add_node(0b00000000, 555, true);
    add_node(0b00000000, 444, true);
    add_node(0b00000000, 333, true);
    svo->nodes.resize(svo_nodes.size());
    thrust::copy(svo_nodes.begin(), svo_nodes.end(), svo->nodes.begin());
    return svo;
}
