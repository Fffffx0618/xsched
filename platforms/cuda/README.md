# CUDA Platform Support for XSched

## Supported preemption level

<table>
  <tr>
    <th align="center">Platform</th>
    <th align="center">XPU</th>
    <th align="center">Shim</th>
    <th align="center">Level-1</th>
    <th align="center">Level-2</th>
    <th align="center">Level-3</th>
  </tr>
  <tr>
    <td align="center" rowspan="4">CUDA</td>
    <td align="center">NVIDIA Ampere GPUs (sm86)</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center">🚧</td>
    <td align="center">🚧</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Volta GPUs (sm70)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Kepler GPUs (sm35)</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">Other NVIDIA GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
</table>

## Multi-stream workloads

Level-2/3 preemption with multiple application streams has known interaction
issues. For multi-stream workloads, set
`XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS=ON`, which maps all application streams
to one internal CUDA stream. Without it, the HAL logs a warning. See
[the protocol environment variables](../../protocol/README.md) for details.

## Non-deactivatable kernels

Cooperative launches (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`) are automatically
marked non-deactivatable. Other kernels that cannot be safely deactivated or
re-executed, such as persistent kernels, can be listed in
`XSCHED_CUDA_NON_DEACTIVATABLE_KERNELS` (comma-separated kernel symbol names).
Such kernels are launched without a guardian, so Level-2 waits for them
instead of preempting them.

## SM86 Level-3 TRAP constraints

`XSCHED_CUDA_LV3_IMPL=TRAP` selects the trap backend (also the default).
The trap handler must be instrumented separately for each CUDA context.
Initialization and trap triggering are serialized within that context.

The SM86 trap backend terminates interrupted CTAs and replays their bodies
when the queue resumes. It does not checkpoint registers or roll back global
memory. Intra-CTA interruption is therefore **opt-in**: list exact CUDA kernel
symbols in `XSCHED_CUDA_TRAP_REPLAY_SAFE_KERNELS` only after establishing that
partial execution followed by replay preserves results. Unlisted kernels
retain Level-2 CTA-boundary deactivation, even when the queue's level is 3.
A kernel with non-idempotent effects, including atomic updates before
the interruption point or in-place accumulation, must not be listed as replay-safe.
To disable CTA-boundary deactivation as well, list it in
`XSCHED_CUDA_NON_DEACTIVATABLE_KERNELS`. With the explicit command API, omit
`kCommandPropertyDeactivatable` for these kernels. This preserves correctness
by letting the kernel finish; it does not provide interruption for that kernel.
CUDA Graph commands similarly fall back to waiting for completion.

`XSCHED_CUDA_LV3_IMPL=TSG` changes a whole context's scheduling time slice.
Its interrupt references are shared by queues in that context. It must not be
treated as an independent per-stream stop, or as proof that in-flight CTAs
have completed. Use separate contexts when evaluating its scheduling effect.

## Usage

Build the CUDA platform adapter:

```bash
make cuda
```

For transparent scheduling, see the [Linux transparent scheduling example](../../examples/Linux/1_transparent_sched/README.md).
