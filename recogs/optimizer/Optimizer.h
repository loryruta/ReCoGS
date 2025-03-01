#pragma once

#include <atomic>

#include "GSCamera.h"
#include "GSRasterizer.h"

namespace gs_train
{
// Forward decl
class App;

class Optimizer
{
private:
    App& m_app;
    glm::ivec2 m_resolution{500, 500};        /// Resolution used for training (lower -> more performance)
    std::vector<GSCamera> m_training_cameras; /// Training cameras (cameras.json in scene's folder)

    std::atomic<bool> m_running = false;

public:
    explicit Optimizer(App& app);
    ~Optimizer();

    void start();
    void signal_stop();

private:
    void load_training_cameras();
};
} // namespace gs_train