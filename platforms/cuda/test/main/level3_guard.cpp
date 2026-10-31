#include <cstdio>
#include <cstdlib>
#include <vector>
#include "xsched/cuda/hal/arch/sm86.h"
#include "xsched/cuda/hal/common/options.h"
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace xsched::cuda;
int main()
{
#if defined(_WIN32)
    std::puts("SKIP: this process-isolated regression uses fork");
    return 0;
#else
    for (int mode = 0; mode < 5; ++mode) {
        pid_t child = fork();
        if (child < 0) return 2;
        if (child == 0) {
            if (mode < 2) {
                TarpHandlerSM86 handler;
                std::vector<unsigned long long> words(mode ? 128 : 2, 0);
                std::vector<char> injected(handler.GetInjectSize());
                handler.Instrument(words.data(), 0x100000, words.size()*8,
                                   injected.data(), 0x200000);
                _exit(0); // unsupported layouts must fail before patching.
            }
            setenv("XSCHED_CUDA_LV3_IMPL", mode == 4 ? "TSG" : "TRAP", 1);
            unsetenv("XSCHED_CUDA_TRAP_REPLAY_SAFE_KERNELS");
            if (mode != 2) setenv("XSCHED_CUDA_TRAP_REPLAY_SAFE_KERNELS", " safe_kernel , other ", 1);
            bool safe = IsCudaKernelTrapReplaySafe("safe_kernel");
            bool unknown = IsCudaKernelTrapReplaySafe("atomic_kernel");
            bool backend = GetCudaLv3Implementation() ==
                (mode == 4 ? kCudaLv3ImplementationTsg : kCudaLv3ImplementationTrap);
            _exit(safe == (mode != 2) && !unknown && backend ? 0 : 2);
        }
        int status;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != (mode < 2 ? EXIT_FAILURE : EXIT_SUCCESS)) {
            std::fprintf(stderr, "FAIL level3 guard case %d\n", mode);
            return 1;
        }
    }
    std::puts("PASS: unsupported handler layouts, default deny, exact replay-safe symbols, TRAP/TSG selection");
    return 0;
#endif
}
