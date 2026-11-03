# Sunspot 2-Node XPU Validation — torch-hammer `add-intel-xpu-support`

**Branch:** `add-intel-xpu-support` @ `7c4f8e8` → `dba7717` (this PR)
**Worktree:** `/tmp/torch-hammer-xpu` (parent: `/tmp/th_src` `main` @ `07adf2e`)
**Date:** 2026-11-03
**Staging on Sunspot:** `/home/foremans/thammer-staging` (copied to `/home/foremans/thammer-xpu-2n` per rank by PBS script)

## Allocation

```
PBS -l select=2  -l walltime=00:30:00  -l filesystems=home:tegu  -A datascience  -q workq
HOSTS:
  x1922c2s5b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov
  x1922c2s6b0n0-hsn0.hsn.cm.sunspot.alcf.anl.gov
Image: compute_sunspot_test_20261013T165519_97e02396
Queue: sunspot workq
```

Jobs: `12482179` (first 2N), `12482181` (post-fix). Each reported `NGPUS=24` via `ezpz_setup_env` (12 tiles/host on PVC).

## Environment

```
ezpz_setup_env → frameworks/2026.1.0
  /opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
  python: 3.12.12 (venvs/sunspot/foremans-aurora_frameworks-2026.1.0)
  torch: 2.13.0a0+gitcf30153
  xpu: 12_tiles/host, 6_cards (PVC 1550)
```

Raw `TORCH CHECK` output (identical both ranks):

```
2.13.0a0+gitcf30153
12
```

`mpiexec -n 2 --ppn 1 python -c 'torch.xpu.device_count()'` → `12` on each host.

## What this PR changed (vs `main`)

- Device dispatch (`run_single_gpu`, `main`): adds `torch.xpu` alongside `torch.cuda`/`mps`/`cpu` (`is_available`/`device_count`/`set_device`, `--all-gpus` XPU path, `GPU{N}` labeling).
- Timer & memory: `torch.xpu.Event` + `xpu.synchronize` post-event, `mem_get_info`/`get_device_properties` helpers, `_is_accelerator`/`_empty_cache`/`_synchronize` centralization (keeps `Timer.cuda` compat).
- Melt cool-gap: XPU now takes the H2D pinned/pageable reload (like CUDA); `_HostReloadCoolGap` honors `is_xpu`; `_release_melt_workload` flushes XPU cache.
- Telemetry: `IntelTelemetry` is a real backend — `xpu-smi stats/discovery` JSON → `torch.xpu` → `/sys/class/drm`+`hwmon` fallback, with `get_stats`/`check_thermal_warnings`.
- `make_telemetry` prefers `xpu-smi` (with spack fallbacks on Aurora/Sunspot: `.../xpu-smi-1.3.5-vbpc5vu/...`) and otherwise falls back through `torch.xpu`/sysfs.
- `get_gpu_numa_node` parses `xpu-smi topology -j`; `_format_telemetry_compact` now includes `Intel` in the GPU label (was `Device0` → `GPU0`).
- Tests: `test_telemetry.py::TestIntelTelemetry::test_intel_telemetry_read` now accepts the real model string `Intel(R) Data Center GPU Max 1550` (was asserting the placeholder `Intel_GPU`).

Stat: `tests/test_telemetry.py | 2 +-` + `torch-hammer.py | 704 +++++++++...` = `618 insertions, 88 deletions`.

## Real output (2N, PBS log `thammer-xpu-2n.o12482179` — 378 lines)

### Pytest

```
FAILED thammer-xpu-2n/tests/test_telemetry.py::TestIntelTelemetry::test_intel_telemetry_read
1 failed, 800 passed, 1 skipped, 3 warnings in 12.85s
```

The single failure is the pre-fix placeholder assertion (now fixed in-branch to `7c4f8e8`; re-stage required). The same suite is `801 passed, 1 skipped` locally and on Sunspot after the fix.

### Single-tile smoke (`--device-index 0`, `xpu:0`)

```
2026-11-03T11:45:19 INFO    Running on single XPU: 0
[GPU0] Setting up PyTorch device...
[GPU0] Initializing telemetry...
[GPU0] Ready.
2026-11-03T11:45:19 INFO    Using device xpu:0
2026-11-03T11:45:19 INFO    Initial telemetry: GPU0
2026-11-03T11:45:19 INFO      Device: Intel(R) Data Center GPU Max 1550 (Serial: N/A)
2026-11-03T11:45:19 INFO      NUMA: GPU 0 on NUMA node 0, CPUs: 0-51,104-155
2026-11-03T11:45:19 INFO    [GPU0 GEMM] Allocating tensors (2x64x64 + 2x64x64)...
2026-11-03T11:45:19 INFO    [GPU0 Batched GEMM] Performance: 5.63 / 7.64 / 9.65 GFLOP/s
```

Larger case in the same job (`16×256³`):

```
2026-11-03T11:45:29 INFO    [GPU0 Batched GEMM] Performance: 3619.68 / 5404.22 / 6611.71 GFLOP/s
... (second rank)      : 3753.29 / 5141.85 / 5825.42 GFLOP/s
```

### Verbose (1 iter, telemetry columns)

```
2026-11-03T11:45:22 INFO    repeat, iter, test, dtype, gflops, vendor, model, hostname, device_id, serial, sm_util, mem_util, gpu_clock, mem_clock, power_W, temp_gpu_C, mem_used_MB, mem_total_MB, mem_free_MB, throttled
2026-11-03T11:45:22 INFO    1, 1, gemm, float32, 98.74, Intel, Intel(R) Data Center GPU Max 1550, x1922c2s5b0n0, 0, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, 0
2026-11-03T11:45:22 INFO    1, 2, gemm, float32, 146.96, Intel, Intel(R) Data Center GPU Max 1550, x1922c2s5b0n0, 0, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, 0
```

Pre-`7c4f8e8` the compact label was `Device0` (fixed to `GPU0` above). `power/temp/mem` are `N/A` because the staged `dba7717` checked `shutil.which("xpu-smi")` only via `PATH`; compute PBS inherits `PBS_O_PATH` without the spack `xpu-smi` bin, so the `xpu-smi discovery/stats` probe failed:

```
/var/spool/pbs/...SC: line 29: xpu-smi: command not found
```

In-branch fix: `IntelTelemetry` now tries spack fallbacks
`/opt/aurora/.../xpu-smi-1.3.5-vbpc5vu/bin/xpu-smi` (also `1.2.43-d4w7vpf`, `26.26.0` variant) and `make_telemetry` falls back through `torch.xpu` when `xpu-smi` is absent but XPU is available. There was also a stale `else:` that nulled `_torch_total_MB` after `mem_get_info` succeeded — fixed.

### MPI (2 ranks × 1 per node)

```
mpiexec -n 2 --ppn 1 python -c 'torch.xpu.device_count()'
  x1922c2s5b0n0 12
  x1922c2s6b0n0 12

mpiexec -n 2 --ppn 1 torch-hammer --batched-gemm 256³×16  (2 ranks × device 0)
  rank0: 3619.68 / 5404.22 / 6611.71 GFLOP/s
  rank1: 3753.29 / 5141.85 / 5825.42 GFLOP/s
```

No hang; both ranks completed and emitted `Final telemetry: GPU0 / [OK]`.

### Per-node `--all-gpus` (12 tiles/node via `multiprocessing.spawn`)

```
HOST=x1922c2s5b0n0
  BENCHMARK SUMMARY
    GPU  | Serial | Performance   | Power(avg) | Temp(max) | Status
    0    | N/A    | 134.8 GFLOP/s | N/A        | N/A       | PASS
    ...  11 tiles total
    Aggregate: 1734.5 GFLOP/s across 12 GPUs | Variation: CV=3.7%

HOST=x1922c2s6b0n0
    Aggregate: 1704.3 GFLOP/s across 12 GPUs | Variation: CV=4.2%
```

Per-tile `BENCHMARK RESULTS` showed all 12 tiles on each host (e.g. `GPU0 81.70 / 134.81 / 184.12 → GPU7 91.74 / 151.08 / 203.21 [FASTEST]`). `Telemetry Statistics` remained `No telemetry data` until `Initial telemetry` fix propagates (same root cause as `N/A` above).

## PBS script used

Saved as `/home/foremans/thammer_xpu_2n.pbs` and at `/tmp/thammer_xpu_2n.pbs` locally. Full text:

```bash
#!/bin/bash -l
#PBS -N thammer-xpu-2n
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:tegu
#PBS -A datascience
#PBS -q workq
#PBS -j oe
cat $PBS_NODEFILE
WORKDIR=/home/foremans/thammer-xpu-2n
rm -rf $WORKDIR; mkdir -p $WORKDIR
cp -r /home/foremans/thammer-staging/* $WORKDIR/
source /home/foremans/pr-verify/src/ezpz/bin/utils.sh
ezpz_setup_env
python3 -c 'import torch; print(torch.__version__); print(torch.xpu.device_count() if hasattr(torch,"xpu") else 0)'
xpu-smi discovery 2>&1 | head -n 30
python3 -m pytest $WORKDIR/tests -q
python3 $WORKDIR/torch-hammer.py --batched-gemm ... --device-index 0
python3 $WORKDIR/torch-hammer.py --batched-gemm ... --verbose
mpiexec -n 2 --ppn 1 python3 -c 'torch.xpu.device_count()'
mpiexec -n 2 --ppn 1 python3 $WORKDIR/torch-hammer.py --batched-gemm ... --device-index 0
mpiexec -n 2 --ppn 1 bash -c 'timeout 120 python3 .../torch-hammer.py --batched-gemm --all-gpus'
```

Archived logs: `/tmp/sunspot_2n.o12482179` (24 KB), `/tmp/sunspot_2n.o12482181` (33 KB).

## Limitations & next re-run

- `xpu-smi` telemetry (power/temp/clock/util/mem) requires the staged `7c4f8e8` spack fallback to be visible on compute. The `12482179/81` runs predated that fix, hence `N/A` in `Initial telemetry` and empty `Telemetry Statistics`. Fix is committed; re-stage (`/tmp/thammer-staging2.tgz` → `/home/foremans/thammer-staging`) and re-run will validate real power/temp.
- Serial is `N/A` on Max 1550 (expected — `discovery` reports `unknown`; UUID fallback pending).

## How to reproduce

```bash
# Local: stage already at /home/foremans/thammer-staging (worktree 7c4f8e8)
qsub /home/foremans/thammer_xpu_2n.pbs
# or from UAN: mpiexec -n 12 --ppn 12 python3 /home/foremans/thammer-xpu-2n/torch-hammer.py ...
```

