#pragma once

#include <string>

namespace xsched::cuda
{

enum CudaLv3Implementation
{
    kCudaLv3ImplementationTrap,
    kCudaLv3ImplementationTsg,
    kCudaLv3ImplementationGate,
};

bool GetCudaSingleStreamPerProcessEnabled();
CudaLv3Implementation GetCudaLv3Implementation();

/// @brief Whether XSCHED_CUDA_NON_DEACTIVATABLE_KERNELS is configured.
bool HasCudaNonDeactivatableKernels();
/// @brief Whether the given kernel symbol is listed as non-deactivatable.
bool IsCudaKernelNonDeactivatable(const std::string &name);
/// @brief Explicit permission to replay interrupted CTAs under TRAP Level-3.
bool IsCudaKernelTrapReplaySafe(const std::string &name);

} // namespace xsched::cuda
