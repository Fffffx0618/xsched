#pragma once

#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/level3/interrupt.h"
#include "xsched/cuda/hal/level3/tsg.h"

namespace xsched::cuda
{

class CudaQueueLv3Trap : public CudaQueueLv2
{
public:
    CudaQueueLv3Trap(CUstream stream);
    virtual ~CudaQueueLv3Trap() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;
    virtual void OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

protected:
    std::shared_ptr<InterruptContext> interrupt_context_ = nullptr;
};

class CudaQueueLv3Tsg : public CudaQueueLv2
{
public:
    CudaQueueLv3Tsg(CUstream stream);
    virtual ~CudaQueueLv3Tsg() = default;

    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }
    virtual bool          SupportsInterruptFastPath() override { return true; }

private:
    std::shared_ptr<TsgContext> tsg_context_ = nullptr;
};

class CudaQueueLv3Gate : public CudaQueueLv1
{
public:
    CudaQueueLv3Gate(CUstream stream);
    virtual ~CudaQueueLv3Gate();

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Interrupt() override;
    virtual void Restore(const preempt::CommandLog &log) override;
    virtual void OnPreemptLevelChange(XPreemptLevel level) override;

    virtual bool          SupportDynamicLevel()  override { return true; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }

private:
    CUstream op_stream_ = nullptr;
    CUdeviceptr flag_ = 0;
    int64_t launch_count_ = 0;
};

} // namespace xsched::cuda
