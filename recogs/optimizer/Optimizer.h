#pragma once

#include <atomic>

#include "Camera.h"
#include "GSRasterizer.h"

namespace recogs
{
// Forward decl
class App;

class Optimizer
{
private:
    App& m_app;
    glm::ivec2 m_resolution{500, 500}; /// Resolution used for training (lower -> more performance)

    std::atomic<bool> m_running = false;

public:
    explicit Optimizer(App& app);
    ~Optimizer();

    void start();
    void signal_stop();
};
} // namespace recogs