#include <cstdio>
#include <memory>

#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/utils/xassert.h"

using namespace xsched::cuda;

// Keep the fixture independent of the CUDA toolkit used to build the tests.
static const char ptx[] = R"(
.version 6.0
.target sm_30
.address_size 64
.visible .entry write_value(.param .u64 output, .param .u32 value) {
    .reg .b64 %rd;
    .reg .b32 %r;
    ld.param.u64 %rd, [output];
    ld.param.u32 %r, [value];
    st.global.u32 [%rd], %r;
    ret;
}
)";

static void CheckHandle(CUfunction input, CUfunction expected, CUstream stream)
{
    CUdeviceptr output;
    CUDA_ASSERT(Driver::MemAlloc_v2(&output, sizeof(int)));
    for (bool deep_copy : {false, true}) {
        int value = deep_copy ? 42 : 17;
        void *params[] = {&output, &value};
        auto command = std::make_shared<CudaKernelLaunchCommand>(
            input, 1, 1, 1, 1, 1, 1, 0, params, nullptr, deep_copy);
        XASSERT(command->kFunc == input, "launch handle was changed");
        XASSERT(command->kFuncHandle == expected, "incorrect metadata handle");
        if (deep_copy) value = -1; // The command must retain the copied argument.
        CUDA_ASSERT(command->LaunchWrapper(stream));
        CUDA_ASSERT(Driver::StreamSynchronize(stream));
        int actual = 0;
        CUDA_ASSERT(Driver::MemcpyDtoH_v2(&actual, output, sizeof(actual)));
        XASSERT(actual == (deep_copy ? 42 : 17), "incorrect kernel output: %d", actual);
    }
    CUDA_ASSERT(Driver::MemFree_v2(output));
}

int main()
{
    CUDA_ASSERT(Driver::Init(0));
    CUdevice device;
    CUDA_ASSERT(Driver::DeviceGet(&device, 0));
    CUcontext context;
    CUDA_ASSERT(Driver::CtxCreate_v2(&context, 0, device));
    CUstream stream;
    CUDA_ASSERT(Driver::StreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CUmodule module;
    CUDA_ASSERT(Driver::ModuleLoadData(&module, ptx));
    CUfunction function;
    CUDA_ASSERT(Driver::ModuleGetFunction(&function, module, "write_value"));
    CheckHandle(function, function, stream);
    std::puts("CUfunction: PASS (borrowed and copied arguments)");

    int version;
    CUDA_ASSERT(Driver::DriverGetVersion(&version));
    if (version >= 12000) {
        CUlibrary library;
        CUDA_ASSERT(Driver::LibraryLoadData(&library, ptx, nullptr, nullptr, 0,
                                           nullptr, nullptr, 0));
        CUkernel kernel;
        CUDA_ASSERT(Driver::LibraryGetKernel(&kernel, library, "write_value"));
        CUfunction resolved;
        CUDA_ASSERT(Driver::KernelGetFunction(&resolved, kernel));
        CheckHandle((CUfunction)kernel, resolved, stream);
        std::puts("CUkernel: PASS (borrowed and copied arguments)");
        CUDA_ASSERT(Driver::LibraryUnload(library));
    }

    CUDA_ASSERT(Driver::ModuleUnload(module));
    CUDA_ASSERT(Driver::StreamDestroy_v2(stream));
    CUDA_ASSERT(Driver::CtxDestroy_v2(context));
}
