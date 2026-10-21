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

## Usage

Build the CUDA platform adapter:

```bash
make cuda
```

For transparent scheduling, see the [Linux transparent scheduling example](../../examples/Linux/1_transparent_sched/README.md).
