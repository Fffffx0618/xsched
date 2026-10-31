#include <cctype>
#include <unordered_set>

#include "xsched/utils/log.h"
#include "xsched/utils/env.h"
#include "xsched/protocol/def.h"
#include "xsched/cuda/hal/common/options.h"

bool xsched::cuda::GetCudaSingleStreamPerProcessEnabled()
{
    static const bool sspp = GetEnvOption(XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS_ENV_NAME, false);
    return sspp;
}

xsched::cuda::CudaLv3Implementation xsched::cuda::GetCudaLv3Implementation()
{
    static const CudaLv3Implementation impl = []()->CudaLv3Implementation {
        const std::string str = GetEnv(XSCHED_CUDA_LV3_IMPL_ENV_NAME);
        if (str.empty() || str == "TRAP") return kCudaLv3ImplementationTrap;
        if (str == "TSG") return kCudaLv3ImplementationTsg;
        if (str == "GATE") return kCudaLv3ImplementationGate;
        XWARN("unknown CUDA Level-3 implementation: %s, use default (TRAP)", str.c_str());
        return kCudaLv3ImplementationTrap;
    }();
    return impl;
}

static const std::unordered_set<std::string> &GetNonDeactivatableKernels()
{
    static const std::unordered_set<std::string> kernels = []() {
        std::unordered_set<std::string> set;
        const std::string str = GetEnv(XSCHED_CUDA_NON_DEACTIVATABLE_KERNELS_ENV_NAME);
        size_t start = 0;
        while (start <= str.size()) {
            size_t end = str.find(',', start);
            if (end == std::string::npos) end = str.size();
            size_t b = start, e = end;
            while (b < e && std::isspace(static_cast<unsigned char>(str[b]))) ++b;
            while (e > b && std::isspace(static_cast<unsigned char>(str[e - 1]))) --e;
            if (e > b) set.emplace(str.substr(b, e - b));
            if (end == str.size()) break;
            start = end + 1;
        }
        return set;
    }();
    return kernels;
}

bool xsched::cuda::HasCudaNonDeactivatableKernels()
{
    return !GetNonDeactivatableKernels().empty();
}

bool xsched::cuda::IsCudaKernelNonDeactivatable(const std::string &name)
{
    return GetNonDeactivatableKernels().count(name) > 0;
}

bool xsched::cuda::IsCudaKernelTrapReplaySafe(const std::string &name)
{
    static const auto safe = []() {
        std::unordered_set<std::string> result;
        const std::string value = GetEnv("XSCHED_CUDA_TRAP_REPLAY_SAFE_KERNELS");
        for (size_t begin = 0; begin <= value.size();) {
            size_t end = value.find(',', begin);
            if (end == std::string::npos) end = value.size();
            size_t left = begin, right = end;
            while (left < right && std::isspace(static_cast<unsigned char>(value[left]))) ++left;
            while (right > left && std::isspace(static_cast<unsigned char>(value[right - 1]))) --right;
            if (right > left) result.emplace(value.substr(left, right - left));
            if (end == value.size()) break;
            begin = end + 1;
        }
        return result;
    }();
    return safe.count(name) != 0;
}
