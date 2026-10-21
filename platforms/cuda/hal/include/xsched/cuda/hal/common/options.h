#pragma once

#include <string>

namespace xsched::cuda
{

enum CudaLv3Implementation
{
    kCudaLv3ImplementationTrap,
    kCudaLv3ImplementationTsg,
};

bool GetCudaSingleStreamPerProcessEnabled();
CudaLv3Implementation GetCudaLv3Implementation();

/// @brief Whether XSCHED_CUDA_NON_DEACTIVATABLE_KERNELS is configured.
bool HasCudaNonDeactivatableKernels();
/// @brief Whether the given kernel symbol is listed as non-deactivatable.
bool IsCudaKernelNonDeactivatable(const std::string &name);

} // namespace xsched::cuda
