#pragma once

#define GSE_CHECK_CUDA(_error) gs_train::checkCudaError(_error, __FILE_NAME__, __LINE__)

namespace gs_train
{
inline void checkCudaError(cudaError_t error, char const* file, int line)
{
    if (error != cudaSuccess) {
        printf("[ERROR] CUDA error: %s (%s:%d)\n", cudaGetErrorString(error), file, line);
        exit(1);
    }
}
} // namespace gs_train
