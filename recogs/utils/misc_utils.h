#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#define CHECK_STATE(condition, ...) recogs::check_state(!!(condition), #condition, __FILE__, __LINE__)
#define CHECK_ARG CHECK_STATE
#define RCGS_LIKELY(x) __builtin_expect(!!(x), 1)
#define RCGS_UNLIKELY(x) __builtin_expect(!!(x), 0)

namespace recogs
{
inline void check_state(bool condition, char const* condition_str, char const* file, int line)
{
    if (RCGS_UNLIKELY(!condition)) {
        fprintf(stderr, "[ERROR] State check failed: %s (%s:%d)\n", condition_str, file, line);
        exit(1);
    }
}

template <typename INT>
#ifdef __CUDACC__
__forceinline__ __host__ __device__ INT div_ceil(INT a, INT b)
#else
INT div_ceil(INT a, INT b)
#endif
{
    if (a == 0) return 0;
    return 1 + ((a - 1) / b); // if a != 0
}

inline size_t get_filesize(const std::filesystem::path& filepath)
{
    FILE* f = fopen(filepath.c_str(), "r");
    fseek(f, 0L, SEEK_END);
    return ftell(f);
}
} // namespace recogs
