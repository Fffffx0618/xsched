#pragma once

#include <map>
#include <mutex>
#include <memory>

#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/level2/mm.h"
#include "xsched/cuda/hal/level3/trap.h"

namespace xsched::cuda
{

class InterruptContext
{
public:
    InterruptContext(CUcontext ctx);
    ~InterruptContext() = default;
    static std::shared_ptr<InterruptContext> Instance(CUcontext ctx);

    void Interrupt();
    void DumpTrapHandler();
    void InstrumentTrapHandler();

private:
    const CUcontext kCtx;
    const CUstream kOpStream;
    
    size_t trap_handler_size_ = 0;
    CUdeviceptr trap_handler_dev_ = 0;
    std::shared_ptr<TarpHandler> trap_handler_ = nullptr;
    std::unique_ptr<InstrMemAllocator> instrument_mem_ = nullptr;
    // Trap handlers belong to CUDA contexts, not to the whole process.
    std::mutex instrument_mtx_;
    bool instrumented_ = false;
    std::mutex interrupt_mtx_;
};

} // namespace xsched::cuda
