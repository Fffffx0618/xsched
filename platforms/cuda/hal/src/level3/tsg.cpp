#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/level3/tsg.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

#ifndef XSCHED_TSG_DIAG
#define XSCHED_TSG_DIAG 0
#endif
#if XSCHED_TSG_DIAG
#define XSCHED_DIAG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define XSCHED_DIAG(...) ((void)0)
#endif


using namespace xsched::cuda;

TsgContext::TsgContext(CUcontext ctx): kCtx(ctx), interrupt_count_(0)
{
    // check current context
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx,
            "create TsgContext failed: current context (%p) does not match context (%p)",
            current_ctx, ctx);
    timeslice_ = cuXtraGetTimeslice(kCtx);
}

void TsgContext::Interrupt()
{
    using namespace std::chrono;
    [[maybe_unused]] auto t0 = steady_clock::now();
    std::lock_guard<std::mutex> lock(state_mtx_);
    if (interrupt_count_ == 0) cuXtraSetTimeslice(kCtx, 0);
    ++interrupt_count_;
    [[maybe_unused]] auto us = duration<double, std::micro>(steady_clock::now()-t0).count();
    XSCHED_DIAG("TSG_DIAG interrupt ctx=%p count=%zu dur_us=%.3f\n",
                 (void*)kCtx, (size_t)interrupt_count_, us);
}

void TsgContext::Restore()
{
    using namespace std::chrono;
    [[maybe_unused]] auto t0 = steady_clock::now();
    std::lock_guard<std::mutex> lock(state_mtx_);
    XASSERT(interrupt_count_ > 0, "unbalanced TSG restore for context %p", kCtx);
    if (interrupt_count_ == 1) cuXtraSetTimeslice(kCtx, timeslice_);
    --interrupt_count_;
    [[maybe_unused]] auto us = duration<double, std::micro>(steady_clock::now()-t0).count();
    XSCHED_DIAG("TSG_DIAG restore ctx=%p count=%zu dur_us=%.3f\n",
                 (void*)kCtx, (size_t)interrupt_count_, us);
}

std::shared_ptr<TsgContext> TsgContext::Instance(CUcontext ctx)
{
    static std::mutex ctx_mtx;
    static std::map<CUcontext, std::shared_ptr<TsgContext>> ctx_map;

    std::lock_guard<std::mutex> lock(ctx_mtx);
    auto it = ctx_map.find(ctx);
    if (it != ctx_map.end()) return it->second;

    auto tsg_ctx = std::make_shared<TsgContext>(ctx);
    ctx_map[ctx] = tsg_ctx;
    return tsg_ctx;
}
