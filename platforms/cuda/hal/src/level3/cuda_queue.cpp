#include "xsched/utils/env.h"
#include <mutex>
#include <cuxtra/cuxtra.h>
#include "xsched/cuda/hal/common/options.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level3/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaQueueLv3Trap::CudaQueueLv3Trap(CUstream stream): CudaQueueLv2(stream)
{
    interrupt_context_ = InterruptContext::Instance(context_);
}

void CudaQueueLv3Trap::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    // wait until the preempt flag is set
    CUDA_ASSERT(Driver::StreamSynchronize(instrument_manager_->OpStream()));
    // FIXME: what if multiple threads call Interrupt()?
    interrupt_context_->Interrupt();
}

void CudaQueueLv3Trap::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
}

void CudaQueueLv3Trap::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (level >= kPreemptLevelDeactivate) WarnMultiStreamPreemption();
    if (level == kPreemptLevelInterrupt) {
        static std::once_flag replay_warning;
        std::call_once(replay_warning, []() {
            XWARN("TRAP Level-3 replays interrupted CTAs: only kernels explicitly listed "
                  "in XSCHED_CUDA_TRAP_REPLAY_SAFE_KERNELS may be interrupted inside a CTA; "
                  "unlisted kernels retain Level-2 deactivation. Do not list kernels "
                  "with non-idempotent side effects");
        });
        instrument_manager_->NotifyTrapInstrumented();
        interrupt_context_->InstrumentTrapHandler();
    }
    level_ = level;
}

void CudaQueueLv3Trap::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    this->CudaQueueLv2::OnHwCommandSubmit(hw_cmd);
    if (level_ < kPreemptLevelInterrupt) return;

    if (std::dynamic_pointer_cast<CudaGraphCommand>(hw_cmd) != nullptr) {
        // do nothing here, will automatically fallback to wait-based preemption
        static bool warned = false;
        if (!warned) {
            warned = true;
            XWARN("CUDA graph cannot support trap-based level-3 preemption, "
                  "falling back to level-1");
        }
        return;
    }
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    // Non-deactivatable kernels are launched through the original path with
    // killable == 0, so the trap handler leaves them alone.
    if (kernel != nullptr &&
        kernel->GetProps(preempt::kCommandPropertyDeactivatable)) {
        const char *name = nullptr;
        cuXtraGetFunctionName(context_, kernel->kFuncHandle, &name);
        kernel->killable = name != nullptr && IsCudaKernelTrapReplaySafe(name);
    }
}

CudaQueueLv3Tsg::CudaQueueLv3Tsg(CUstream stream): CudaQueueLv2(stream)
{
    tsg_context_ = TsgContext::Instance(context_);
}

void CudaQueueLv3Tsg::Interrupt()
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Interrupt() not supported on level-%d", level_);
    tsg_context_->Interrupt();
}

void CudaQueueLv3Tsg::Restore(const CommandLog &)
{
    XASSERT(level_ >= kPreemptLevelInterrupt, "Restore() not supported on level-%d", level_);
    tsg_context_->Restore();
}

void CudaQueueLv3Tsg::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    if (level >= kPreemptLevelDeactivate) WarnMultiStreamPreemption();
    level_ = level;
}

CudaQueueLv3Gate::CudaQueueLv3Gate(CUstream stream): CudaQueueLv1(stream)
{
    CUDA_ASSERT(Driver::CtxSetCurrent(context_));
    int lp, hp;
    CUDA_ASSERT(Driver::CtxGetStreamPriorityRange(&lp, &hp));
    CUDA_ASSERT(Driver::StreamCreateWithPriority(&op_stream_, CU_STREAM_NON_BLOCKING, hp));
    CUDA_ASSERT(Driver::MemAlloc_v2(&flag_, sizeof(uint32_t)));
    CUDA_ASSERT(Driver::MemsetD32_v2(flag_, 1, 1));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
}

CudaQueueLv3Gate::~CudaQueueLv3Gate()
{
    if (context_ != nullptr) Driver::CtxSetCurrent(context_);
    if (flag_ != 0) Driver::MemFree_v2(flag_);
    if (op_stream_ != nullptr) Driver::StreamDestroy_v2(op_stream_);
}

void CudaQueueLv3Gate::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    // Enqueue a gate before each kernel: the app stream stops here when the
    // host writes 0 to the flag, without a context switch or CTA replay.
    if (std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd) != nullptr) {
        CUDA_ASSERT(Driver::StreamWaitValue32(kStream, flag_, 1, CU_STREAM_WAIT_VALUE_EQ));
    }
    CudaQueueLv1::Launch(hw_cmd);
}

void CudaQueueLv3Gate::Interrupt()
{
    CUDA_ASSERT(Driver::StreamWriteValue32(op_stream_, flag_, 0, 0));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
}

void CudaQueueLv3Gate::Restore(const preempt::CommandLog &)
{
    CUDA_ASSERT(Driver::StreamWriteValue32(op_stream_, flag_, 1, 0));
    CUDA_ASSERT(Driver::StreamSynchronize(op_stream_));
}

void CudaQueueLv3Gate::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelInterrupt, "unsupported level: %d", level);
    level_ = level;
}
