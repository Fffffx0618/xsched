#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

// Kept alive by queues and commands, rather than a process-wide owning map.
// Cached buffers are exact-sized and bounded to 64 MiB / 64 allocations.
class HostBufferPool
{
public:
    static std::shared_ptr<HostBufferPool> Instance(CUcontext context);
    explicit HostBufferPool(CUcontext context): context_(context) {}
    ~HostBufferPool();
    void *Acquire(size_t size);
    void Release(void *buffer, size_t size, bool reusable);

private:
    CUcontext context_;
    std::mutex mutex_;
    std::map<size_t, std::vector<void *>> buffers_;
    size_t cached_bytes_ = 0;
    size_t cached_count_ = 0;
};

} // namespace xsched::cuda
