#pragma once

#include <cstdio>
#include <cstdlib>

#define GSE_CHECK_STATE(condition, ...) gs_train::check_state(condition, #condition, __FILE__, __LINE__)

namespace gs_train
{
inline void check_state(bool condition, char const* condition_str, char const* file, int line)
{
    if (!condition) {
        fprintf(stderr, "[ERROR] State check failed: %s (%s:%d)\n", condition_str, file, line);
        exit(1);
    }
}
} // namespace gs_train
