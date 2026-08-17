# CUDA snapshotting: findings

Testing `cuda-checkpoint` (NVIDIA's GPU state suspend/resume tool) and CRIU
(Linux process checkpoint/restore) to speed up cold starts for a small model
inference process, on a Lambda A10G instance (Ubuntu 22.04, driver
580.173.02, criu 3.16.1).

## What works: bare-process cuda-checkpoint suspend/resume (PR #1, merged)

`cuda-checkpoint --toggle` locks/unlocks a *still-alive* process's GPU state
(copies GPU memory into host RAM and back), combined with `SIGSTOP`/`SIGCONT`
to freeze the process itself. No CRIU, no container, no process kill.

Result: cold start 6.54s vs suspend+resume 0.33s (~20x). Output identical
before/after. This is the validated, working benchmark
(`benchmark.sh`).

## What's broken: Docker's own `checkpoint`/`start --checkpoint`

Two separate containerd bugs, hit on trivial non-GPU containers before CUDA
ever entered the picture:
- `docker start --checkpoint --checkpoint-dir <path>`: "custom checkpointdir
  is not supported" (create supports a custom dir, restore doesn't).
- Even using the default checkpoint location: `commit failed: content
  sha256:... already exists` (containerd content-store conflict on restore).

Not pursued further — this is Docker's own (lightly-maintained, experimental)
checkpoint feature failing independent of anything we're doing.

## What's blocked: container + manual CRIU dump/restore (PR #2, open, documented)

Bypassing Docker's checkpoint feature and driving `cuda-checkpoint` + `criu`
directly against a containerized process (`--pid=host` so criu doesn't have
to reconstruct a PID namespace).

`nvidia-container-toolkit` bind-mounts ~50 *individual host files* into the
container (driver .so files, `nvidia-smi`, firmware blobs, etc.) to give it
GPU access without a CUDA-toolkit base image — on top of Docker's own
`/etc/hosts`, `/etc/hostname`, `/etc/resolv.conf` bind mounts. Each of these
needs a `criu --external mnt[...]:tag` declaration (`gen_external_mounts.py`
auto-generates the full list by parsing `/proc/<pid>/mountinfo`).

Even with all ~54 mounts correctly declared external, `criu dump` fails:

```
Error (criu/mount.c:654): mnt: 639:./etc/hosts doesn't have a proper root mount
```

Verbose logging showed criu's mount-sharing/propagation-group detection
treats all ~50 same-device individual file bind mounts as one ambiguous
group, anchored at the `/etc/hosts` mount, and can't resolve a root mount for
that anchor — even though it's declared external. This is criu's own
mount-tree analysis choking on the *pattern* of many same-device single-file
bind mounts, not a missing flag. Reproduced consistently. Not pursued
further without patching/upgrading criu or changing the GPU-access mechanism
(e.g. CDI mode).

## In progress: bare-process (no container) CRIU dump/restore

Same idea as PR #2, but on the plain host process (no Docker), which sidesteps
the mount-tree problem entirely since there's no container mount namespace —
`libcuda.so` is just a normal file at its normal path, no bind mounts.

Issues hit and fixed, in order:
1. **`Connected TCP socket, consider using --tcp-established`** — a leftover
   HTTP connection pool from the earlier Hugging Face Hub request. Fixed by
   adding `--tcp-established` to both `criu dump` and `criu restore`.
2. **`Current gid <N> intersects with pid (255) in images`** on restore —
   the dumped process had inherited its session ID from the SSH shell that
   launched it, which collided with IDs criu tries to establish on restore.
   Fixed by launching `infer.py` under `setsid` from the start, so it gets
   its own independent session at dump time.
3. **Segfault on restore** (current blocker): `criu dump` now succeeds
   cleanly (~1.8s, 3.1GB checkpoint including the GPU state
   `cuda-checkpoint` copied into host RAM). But `criu restore` crashes the
   process with SIGSEGV the instant its injected "restorer" code hands
   control back to real execution — happens after reconstructing 63 threads'
   worth of state (PyTorch spins up many threads: thread pools, CUDA event
   polling, etc.).

   Two live hypotheses, not yet distinguished:
   - CUDA-specific: some low-level GPU driver memory mapping (not the
     weight data itself, which `cuda-checkpoint` already handles) isn't
     reconstructed correctly by criu.
   - General criu fragility with heavily multithreaded processes,
     independent of CUDA.

   Currently isolating which by building and testing NVIDIA's own reference
   example (`cuda-checkpoint` repo's `src/counter.cu` + `example.sh`, a
   minimal CUDA program, officially documented to work with this exact
   `cuda-checkpoint` + `criu dump`/`criu restore` sequence). If NVIDIA's
   own minimal example restores cleanly on this same instance, the problem
   is specific to PyTorch's thread complexity, not the environment. If it
   also segfaults, it's environment-level (criu/driver version
   incompatibility) and not something fixable via configuration here.

## Other gotchas hit along the way (unrelated to the core investigation)

- `pkill -f infer.py` matches its own command-line argv (the string
  "infer.py" appears in the pkill invocation itself) and can kill the
  calling shell/SSH session. Use the bracket trick: `pkill -f '[i]nfer.py'`.
- `python:3.11-slim` has no C compiler; PyTorch's `triton` backend needs one
  at runtime to JIT-compile certain kernels. Needed `build-essential`.
- `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`
  dict in newer `transformers`, not a raw tensor — needed
  `return_dict=True` and `**inputs` unpacking into `generate()`.
- Old system `jinja2` (3.0.3) is below what `apply_chat_template` requires
  (>=3.1.0).
