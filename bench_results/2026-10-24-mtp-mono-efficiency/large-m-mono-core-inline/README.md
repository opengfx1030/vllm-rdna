# Disposable inline-barrier core build

`build_disposable_core.py` creates a separate `_rocm_C.abi3.so` by replaying
the configured Ninja compile and link commands for `mono_k2_rdna2.hip.o`. It
applies exactly one source edit to the generated HIP input:

```diff
-__device__ bool grid_barrier(
+__device__ __forceinline__ bool grid_barrier(
```

The script defaults to plan-only. It tokenizes commands with `shlex`, accepts
only simple CMake/Ninja wrappers, and invokes compile/link argv directly with
`subprocess.run` (no shell). The compile object, source input, dependency
outputs, link object, and shared-library output are redirected into a new
scratch directory. Other command arguments remain as emitted by Ninja. The
link dependency file is also redirected. Because the generated HIP file is
staged elsewhere, the compile argv adds only `-iquote <original source dir>`
to preserve its quoted-header lookup. The manifest records command hashes,
the source and transitive quoted-header hashes, baseline/object hashes, hashes
for explicit other object/archive inputs, and the final binary hash.

Example on the build host:

```bash
python3 build_disposable_core.py \
  --build-dir /home/chenco_adm/vllm-rdna-mtp-mono/build/temp.linux-x86_64-cpython-312 \
  --scratch-dir /home/chenco_adm/w4a8_runs/mono-inline-core-build-unique
```

Inspect the printed command vectors and `build_manifest.json`. Only after
review, pass `--execute` to compile and link. The scratch path must be new or
empty. The script does not install/copy the result over vLLM's core extension.
It blanks HIP, ROCR, and CUDA device visibility for compiler/linker subprocesses.

After building, `verify_disposable_core.py` checks source/object/header and
other link-input hashes, verifies the one-annotation source delta, and checks
that the installed extension and tracked source still match caller-supplied
pre-build hashes. It also requires a reviewed normalization map
(`{"replacements":[{"generated":"...","tracked":"...","count":N}]}`)
to prove generated `.hip` maps exactly back to tracked `.cu`; it does not infer
or accept source differences.
