#include "xsched/utils/xassert.h"
#include "xsched/preempt/sched/agent.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/preempt/xqueue/async_xqueue.h"

#include <chrono>
#include <cstdio>
#include <thread>

#ifndef XSCHED_TSG_DIAG
#define XSCHED_TSG_DIAG 0
#endif
#if XSCHED_TSG_DIAG
#define XSCHED_DIAG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define XSCHED_DIAG(...) ((void)0)
#endif


using namespace xsched::utils;

[[maybe_unused]] static inline double DiagNowUs() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}
using namespace xsched::sched;
using namespace xsched::preempt;

AsyncXQueue::AsyncXQueue(std::shared_ptr<HwQueue> hwq, XPreemptLevel level,
                         int64_t threshold, int64_t batch_size)
    : XQueue(kXQueueImplTypeAsync,
             kQueueFeatureAsyncSubmit      |
             (hwq->SupportDynamicLevel()   ? kQueueFeatureDynamicLevel : 0)  |
             kQueueFeatureDynamicThreshold | kQueueFeatureDynamicBatchSize   |
             kQueueFeatureSyncSuspend      | kQueueFeatureResumeDropCommands ,
             hwq)
    , level_(level)
    , cmd_buf_(std::make_shared<CommandBuffer>(kHandle))
    , launch_worker_(std::make_shared<LaunchWorker>(hwq, cmd_buf_, level, threshold, batch_size))
{
    XASSERT(level_ > kPreemptLevelUnknown && level_ < kPreemptLevelMax,
            "invalid preempt level: %d", level_);

    auto wait_cmd = cmd_buf_->EnqueueXQueueWaitAllCommand();
    wait_cmd->Wait();
    kHwQueue->OnPreemptLevelChange(level_);
    SchedAgent::SendEvent(std::make_shared<XQueueCreateEvent>(kHandle, kDevice, level,
                                                              threshold, batch_size));
    if (level_ >= kPreemptLevelInterrupt) StartWatchdog();
}

AsyncXQueue::~AsyncXQueue()
{
    // If the XQueue is terminated, it should not be preempted.
    terminated_.store(true);
    this->Resume(true);
    StopWatchdog();

    auto destroy_command = std::make_shared<XQueueDestroyCommand>();
    cmd_buf_->Enqueue(destroy_command);
    destroy_command->Wait();
    SchedAgent::SendEvent(std::make_shared<XQueueDestroyEvent>(kHandle));
}

void AsyncXQueue::Submit(std::shared_ptr<HwCommand> hw_cmd)
{
    hw_cmd->OnSubmit(shared_from_this());
    hw_cmd->SetIdx(next_hw_cmd_idx_.fetch_add(1));
    kHwQueue->OnHwCommandSubmit(hw_cmd);
    cmd_buf_->Enqueue(hw_cmd);
    if (hw_cmd->GetProps(kCommandPropertyBlockingSubmit)) hw_cmd->WaitUntil(kCommandStateInFlight);
}

std::shared_ptr<XQueueWaitAllCommand> AsyncXQueue::SubmitWaitAll()
{
    return cmd_buf_->EnqueueXQueueWaitAllCommand();
}

void AsyncXQueue::WaitAll()
{
    auto sync_command = cmd_buf_->EnqueueXQueueWaitAllCommand();
    sync_command->Wait();
}

void AsyncXQueue::Wait(std::shared_ptr<HwCommand> hw_cmd)
{
    if (!hw_cmd->Synchronizable()) {
        // If it is not synchronizable, it can only be synced
        // by waiting its state turns to kCommandStateCompleted.
        hw_cmd->WaitUntil(kCommandStateCompleted);
        return;
    }

    // Check if it is already completed.
    if (hw_cmd->GetState() >= kCommandStateCompleted) return;

    // If this HwCommand is synchronizable, then use launch worker to sync it.
    hw_cmd->WaitUntil(kCommandStateInFlight);
    launch_worker_->SyncCmd(hw_cmd);
}

XQueueState AsyncXQueue::Query()
{
    return cmd_buf_->GetXQueueState();
}

int64_t AsyncXQueue::GetHwCommandCount()
{
    return next_hw_cmd_idx_.load() - 1;
}

void AsyncXQueue::Suspend(int64_t flags)
{
    // If the XQueue is already terminated, it should not be suspended.
    if (terminated_.load()) return;

    // Will not suspend if it is already suspended.
    bool expected = false;
    if (!suspended_.compare_exchange_strong(expected, true)) return;

    if (flags & kQueueSuspendFlagWaitIdle) {
        // Wait for all HwCommands to be completed,
        // and the XQueue becomes idle before suspending.
        cmd_buf_->WaitForNextIdle();
    } else if (flags & kQueueSuspendFlagWaitAll) {
        // Wait for all HwCommands to be completed before suspending.
        this->WaitAll();
    }

    launch_worker_->Pause();
    size_t inflight = launch_worker_->InflightCount();
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Deactivate();
    if (level_ >= kPreemptLevelInterrupt && !kHwQueue->SupportsInterruptFastPath()) {
        // Gate backend: Interrupt() pauses the app stream at the next kernel
        // boundary; no context switch and no watchdog.
        kHwQueue->Interrupt();
        interrupt_applied_ = true;
        XSCHED_DIAG("TSG_FASTPATH xq=%llu gate=1 interrupt=1\n",
                    (unsigned long long)GetHandle());
    } else if (level_ >= kPreemptLevelInterrupt) {
        // Fast path: stop queued work at the kernel boundary (Lv2), then let the
        // watchdog escalate to a whole-context TSG switch only if the running
        // kernel has not finished by the deadline.  Do NOT block here: blocking
        // the scheduler delays the foreground and turns the fast path into a
        // tail-latency source.
        // Escalation crossover: kernels shorter than this are handled by the
        // kernel-boundary stop; longer kernels use the TSG context switch.
        // 5 ms is the measured crossover from the long-CTA experiment (5.5 ms
        // CTA: TSG wins; 73--86 us CTA: Lv2 wins).  The env override exists only
        // for diagnostics, not for per-workload tuning.
        // Fixed mechanism constant: the measured crossover between the
        // kernel-boundary (Lv2) wait and the TSG context switch.  It is not a
        // per-workload tuning knob and is not exposed as an env parameter.
        constexpr int64_t kTsgEscalationUs = 5000;
        int64_t drain_us = kTsgEscalationUs;
        // If there is nothing in flight, the queue is idle without asking the
        // driver; this removes a cuStreamQuery from the common idle case.
        bool idle = (inflight == 0);
        if (!idle && kHwQueue->QueryIdleSupported()) idle = kHwQueue->QueryIdle();
        if (idle) {
            CancelWatchdog();
            std::lock_guard<std::mutex> lock(watch_mtx_);
            interrupt_applied_ = false;
            XSCHED_DIAG("TSG_FASTPATH xq=%llu idle=1 interrupt=0\n",
                         (unsigned long long)GetHandle());
        } else {
            ScheduleInterrupt(drain_us);
            XSCHED_DIAG("TSG_FASTPATH xq=%llu idle=0 scheduled=1 drain_us=%lld\n",
                         (unsigned long long)GetHandle(), (long long)drain_us);
        }
    }
    if (flags & kQueueSuspendFlagSyncHwQueue) kHwQueue->Synchronize();
    XSCHED_DIAG("XSCHED_DIAG suspend_end xq=%llu level=%d dur_us=%.3f\n",
                 (unsigned long long)GetHandle(), (int)level_, DiagNowUs()-diag_t0);
}

void AsyncXQueue::Resume(int64_t flags)
{
    // Will not resume if it is not suspended.
    bool expected = true;
    if (!suspended_.compare_exchange_strong(expected, false)) return;

    if (level_ == kPreemptLevelBlock) {
        // Should not clear the command log because when the XQueue is suspended without
        // synchronization, the HwCommand may be still running when resumes.
        launch_worker_->Resume();
        XSCHED_DIAG("XSCHED_DIAG resume_end xq=%llu level=%d dur_us=%.3f\n",
                     (unsigned long long)GetHandle(), (int)level_, DiagNowUs()-diag_t0);
        return;
    }

    bool interrupt_applied = false;
    {
        std::lock_guard<std::mutex> lock(watch_mtx_);
        watch_pending_ = false;
        ++watch_gen_;
        watch_cv_.notify_all();
        interrupt_applied = interrupt_applied_;
        interrupt_applied_ = false;
    }

    // synchronization should be done in HwQueue::Reactivate.
    if (!(flags & kQueueResumeFlagDropCommands)) {
        // Regular restore, reactivate and then resume if not dropping commands.
        const CommandLog &log = launch_worker_->GetCommandLog();
        if (level_ >= kPreemptLevelInterrupt && interrupt_applied) kHwQueue->Restore(log);
        if (level_ >= kPreemptLevelDeactivate) kHwQueue->Reactivate(log);
        launch_worker_->Resume();
        XSCHED_DIAG("XSCHED_DIAG resume_end xq=%llu level=%d dur_us=%.3f\n",
                     (unsigned long long)GetHandle(), (int)level_, DiagNowUs()-diag_t0);
        return;
    }

    // Restore and reactivate the hal queue with an empty command log.
    CommandLog empty_log = {};
    if (level_ >= kPreemptLevelInterrupt && interrupt_applied) kHwQueue->Restore(empty_log);
    if (level_ >= kPreemptLevelDeactivate) kHwQueue->Reactivate(empty_log);
    // Drop all HwCommands in the command buffer.
    cmd_buf_->DropAll();

    // drop_idx is the last HwCommand idx that needs to be dropped.
    // Here drop_idx is the idx of the last HwCommand submitted.
    int64_t drop_idx = next_hw_cmd_idx_.load() - 1;
    launch_worker_->ResumeAndDrop(drop_idx);
    XSCHED_DIAG("XSCHED_DIAG resume_end xq=%llu level=%d dur_us=%.3f\n",
                 (unsigned long long)GetHandle(), (int)level_, DiagNowUs()-diag_t0);
}

void AsyncXQueue::StartWatchdog()
{
    watch_thread_ = std::thread([this]() { this->WatchdogLoop(); });
}

void AsyncXQueue::StopWatchdog()
{
    {
        std::lock_guard<std::mutex> lock(watch_mtx_);
        watch_stop_ = true;
        watch_pending_ = false;
        ++watch_gen_;
        watch_cv_.notify_all();
    }
    if (watch_thread_.joinable()) watch_thread_.join();
}

void AsyncXQueue::ScheduleInterrupt(int64_t drain_us)
{
    std::lock_guard<std::mutex> lock(watch_mtx_);
    watch_deadline_ = std::chrono::steady_clock::now() +
                      std::chrono::microseconds(drain_us > 0 ? drain_us : 0);
    ++watch_gen_;
    watch_pending_ = true;
    watch_cv_.notify_all();
}

void AsyncXQueue::CancelWatchdog()
{
    std::lock_guard<std::mutex> lock(watch_mtx_);
    watch_pending_ = false;
    ++watch_gen_;
    watch_cv_.notify_all();
}

void AsyncXQueue::WatchdogLoop()
{
    std::unique_lock<std::mutex> lock(watch_mtx_);
    while (!watch_stop_) {
        watch_cv_.wait(lock, [&]() { return watch_stop_ || watch_pending_; });
        if (watch_stop_) return;

        uint64_t gen = watch_gen_;
        auto deadline = watch_deadline_;
        while (watch_pending_ && gen == watch_gen_ && !watch_stop_) {
            if (watch_cv_.wait_until(lock, deadline) == std::cv_status::timeout) break;
        }
        if (watch_stop_) return;
        if (!watch_pending_ || gen != watch_gen_) continue;

        bool idle = kHwQueue->QueryIdleSupported() && kHwQueue->QueryIdle();
        if (!idle) {
            kHwQueue->Interrupt();
            interrupt_applied_ = true;
            XSCHED_DIAG("TSG_WATCHDOG xq=%llu fire idle=0 interrupt=1\n",
                         (unsigned long long)GetHandle());
        } else {
            XSCHED_DIAG("TSG_WATCHDOG xq=%llu fire idle=1 interrupt=0\n",
                         (unsigned long long)GetHandle());
        }
        watch_pending_ = false;
    }
}

void AsyncXQueue::SetPreemptLevel(XPreemptLevel level)
{
    XASSERT(level > kPreemptLevelUnknown && level < kPreemptLevelMax,
            "invalid preempt level: %d", level);
    // Dynamic level support of this AsyncXQueue depends on kHwQueue->SupportDynamicLevel().
    if (!this->GetFeatures(kQueueFeatureDynamicLevel)) {
        XASSERT(level == level_, "AsyncXQueue does not support dynamic level");
        return;
    }
    if (level > kHwQueue->GetMaxSupportedLevel()) {
        XWARN("preempt level %d is not supported by the HwQueue, "
              "max supported level is %d", level, kHwQueue->GetMaxSupportedLevel());
        return;
    }
    // Not thread-safe (with Submit & Wait).
    // TODO: change to thread-safe, because the level change may be triggered by scheduler or xcli.
    this->WaitAll();
    kHwQueue->OnPreemptLevelChange(level);
    launch_worker_->SetPreemptLevel(level);
    level_ = level;
    SchedAgent::SendEvent(std::make_shared<XQueueConfigUpdateEvent>(kHandle, kDevice, level_,
        launch_worker_->GetThreshold(), launch_worker_->GetBatchSize()));
}

void AsyncXQueue::SetLaunchConfig(int64_t threshold, int64_t batch_size)
{
    if (threshold <= 0 && batch_size <= 0) return;

    // Check command threshold.
    if (threshold <= 0) {
        threshold = launch_worker_->GetThreshold();
    } else {
        XASSERT(this->GetFeatures(kQueueFeatureDynamicThreshold),
                "AsyncXQueue does not support dynamic threshold");
    }

    // Check command batch size.
    if (batch_size <= 0) {
        batch_size = launch_worker_->GetBatchSize();
    } else {
        XASSERT(this->GetFeatures(kQueueFeatureDynamicBatchSize),
                "AsyncXQueue does not support dynamic batch size");
    }

    // Check threshold and batch size are valid.
    XASSERT(threshold >= batch_size,
            "command threshold (" FMT_64D ") must not be smaller than "
            "command batch size (" FMT_64D ")", threshold, batch_size);

    // Not thread-safe.
    // TODO: change to thread-safe
    this->WaitAll();
    launch_worker_->SetLaunchConfig(threshold, batch_size);
    SchedAgent::SendEvent(std::make_shared<XQueueConfigUpdateEvent>(kHandle, kDevice, level_,
        launch_worker_->GetThreshold(), launch_worker_->GetBatchSize()));
}
