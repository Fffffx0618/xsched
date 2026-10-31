#include "xsched/cuda/hal/common/host_buffer_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

std::shared_ptr<HostBufferPool> HostBufferPool::Instance(CUcontext context)
{
    static std::mutex mutex;
    static std::map<CUcontext, std::weak_ptr<HostBufferPool>> pools;
    std::lock_guard<std::mutex> lock(mutex);
    auto pool = pools[context].lock();
    if (!pool) {
        pool = std::make_shared<HostBufferPool>(context);
        pools[context] = pool;
    }
    return pool;
}

void *HostBufferPool::Acquire(size_t size)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = buffers_.find(size);
        if (it != buffers_.end() && !it->second.empty()) {
            void *buffer = it->second.back();
            it->second.pop_back();
            cached_bytes_ -= size;
            --cached_count_;
            if (it->second.empty()) buffers_.erase(it);
            return buffer;
        }
    }
    void *buffer = nullptr;
    CUDA_ASSERT(Driver::MemAllocHost_v2(&buffer, size));
    return buffer;
}

void HostBufferPool::Release(void *buffer, size_t size, bool reusable)
{
    constexpr size_t kMaxBytes = 64UL << 20;
    if (reusable && size <= kMaxBytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cached_count_ < 64 && cached_bytes_ <= kMaxBytes - size) {
            buffers_[size].push_back(buffer);
            cached_bytes_ += size;
            ++cached_count_;
            return;
        }
    }
    CUDA_ASSERT(Driver::MemFreeHost(buffer));
}

HostBufferPool::~HostBufferPool()
{
    CUcontext current = nullptr;
    if (Driver::CtxGetCurrent(&current) == CUDA_ERROR_DEINITIALIZED) return;
    const bool push = current != context_;
    if (push) CUDA_ASSERT(Driver::CtxPushCurrent_v2(context_));
    for (auto &entry : buffers_) {
        for (void *buffer : entry.second) CUDA_ASSERT(Driver::MemFreeHost(buffer));
    }
    if (push) CUDA_ASSERT(Driver::CtxPopCurrent_v2(&current));
}
