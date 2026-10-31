#pragma once

#include "xsched/types.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/handle.h"
#include "xsched/cuda/hal/common/cuda_command.h"

namespace xsched::cuda
{

class CudaQueueLv1 : public preempt::HwQueue
{
public:
    CudaQueueLv1(CUstream stream);
    virtual ~CudaQueueLv1() = default;

    virtual void Launch(std::shared_ptr<preempt::HwCommand> hw_cmd) override;
    virtual void Synchronize() override;
    virtual void OnXQueueCreate() override;
    static CUresult DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel, CUstream stream);

    unsigned int          GetStreamFlags()       const    { return stream_flags_; }
    virtual XDevice       GetDevice()            override { return xdevice_; }
    virtual HwQueueHandle GetHandle()            override { return GetHwQueueHandle(kStream); }
    virtual bool          SupportDynamicLevel()  override { return false; }
    virtual XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelBlock; }
    virtual bool QueryIdleSupported() override { return true; }
    virtual bool QueryIdle() override;

protected:
    const CUstream kStream;
    unsigned int   stream_flags_ = 0;
    CUdevice       cudevice_ = 0;
    XDevice        xdevice_;
    CUcontext      context_ = nullptr;
    XPreemptLevel  level_ = kPreemptLevelUnknown;
    std::shared_ptr<HostBufferPool> host_buffer_pool_;
};

} // namespace xsched::cuda
