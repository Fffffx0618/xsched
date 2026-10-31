#pragma once

#include <memory>
#include <mutex>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

class TsgContext
{
public:
    TsgContext(CUcontext ctx);
    ~TsgContext() = default;
    static std::shared_ptr<TsgContext> Instance(CUcontext ctx);

    void Interrupt();
    void Restore();

private:
    const CUcontext kCtx;
    size_t timeslice_ = 0;
    std::mutex state_mtx_;
    size_t interrupt_count_;
};

} // namespace xsched::cuda
