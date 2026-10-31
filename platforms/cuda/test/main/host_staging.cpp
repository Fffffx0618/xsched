#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

#include "xsched/cuda/hal.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/utils/xassert.h"
#include "xsched/xqueue.h"

using namespace xsched::cuda;

int main()
{
    CUDA_ASSERT(Driver::Init(0));
    CUdevice device;
    CUDA_ASSERT(Driver::DeviceGet(&device, 0));
    CUcontext context;
    CUDA_ASSERT(Driver::CtxCreate_v2(&context, 0, device));
    constexpr size_t kWords = 4096;
    constexpr size_t kCopies = 64;
    CUstream streams[2];
    CUdeviceptr output[2];
    HwQueueHandle hardware[2];
    XQueueHandle handles[2];
    for (int j = 0; j < 2; ++j) {
        CUDA_ASSERT(Driver::StreamCreate(&streams[j], CU_STREAM_NON_BLOCKING));
        CUDA_ASSERT(Driver::MemAlloc_v2(&output[j], kCopies * kWords * sizeof(int)));
        XASSERT(::CudaQueueCreate(&hardware[j], streams[j]) == kXSchedSuccess, "create factory queue");
        XASSERT(XQueueCreate(&handles[j], hardware[j], 1, 0) == kXSchedSuccess, "create xqueue");
        XASSERT(XQueueSetLaunchConfig(handles[j], 16, 8) == kXSchedSuccess, "configure xqueue");
    }
    for (int round = 0; round < 8; ++round) {
        std::thread threads[2];
        for (int j = 0; j < 2; ++j) {
            threads[j] = std::thread([&, j]() {
                CUDA_ASSERT(Driver::CtxSetCurrent(context));
                std::vector<int> source(kWords);
                auto queue = xsched::preempt::XQueueManager::Get(handles[j]);
                for (size_t i = 0; i < kCopies; ++i) {
                    std::fill(source.begin(), source.end(), round * 1000 + j * 100 + i);
                    queue->Submit(std::make_shared<CudaMemcpyHtoDV2Command>(
                        output[j] + i * kWords * sizeof(int), source.data(), kWords * sizeof(int)));
                    // The submission must preserve its own snapshot, including
                    // when a staging buffer is reused in a later round.
                    std::fill(source.begin(), source.end(), -1);
                }
                XASSERT(XQueueWaitAll(handles[j]) == kXSchedSuccess, "wait for copies");
            });
        }
        for (auto &thread : threads) thread.join();
        for (int j = 0; j < 2; ++j) {
            std::vector<int> actual(kWords * kCopies);
            CUDA_ASSERT(Driver::MemcpyDtoH_v2(actual.data(), output[j], actual.size() * sizeof(int)));
            for (size_t i = 0; i < kCopies; ++i) {
                for (size_t k = 0; k < kWords; ++k) {
                    XASSERT(actual[i * kWords + k] == round * 1000 + j * 100 + (int)i,
                            "staging snapshot was overwritten: round=%d queue=%d copy=%zu", round, j, i);
                }
            }
        }
    }
    for (int j = 0; j < 2; ++j) {
        XASSERT(XQueueDestroy(handles[j]) == kXSchedSuccess, "destroy xqueue");
        XASSERT(HwQueueDestroy(hardware[j]) == kXSchedSuccess, "destroy hardware queue");
        CUDA_ASSERT(Driver::MemFree_v2(output[j]));
        CUDA_ASSERT(Driver::StreamDestroy_v2(streams[j]));
    }
    CUDA_ASSERT(Driver::CtxDestroy_v2(context));
    std::puts("host staging: PASS (two queues, immediate source mutation, eight reuse rounds)");
}
