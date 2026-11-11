# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate for RDNA-specific behavior in shared (non-ROCm-only) code paths.

Hot paths traced by Dynamo should read the result once at module import
(``_ON_RDNA = on_rdna_family()``) instead of calling these per forward.
"""

_ON_RDNA_FAMILY: bool | None = None
_ON_RDNA2: bool | None = None
_ON_RDNA3: bool | None = None
_RDNA3_HIP_FAMILIES: frozenset[str] | None = None

# Kernel families of the fork's RDNA HIP ports that can be switched on for a
# gfx1100 (RDNA3) device with VLLM_RDNA3_HIP_OPS. The kernels are wave32 /
# v_dot2_f32_f16 code that also compiles for gfx1100 (the fat build carries it)
# but has no recorded gfx1100 run, so the default there is the Triton / torch /
# upstream route; each family is switched on after its per-op test passes.
#   hc     hyper-connection mix kernels (decode and prefill)
#   gdn    gated-delta-net decode / prefill / spec-verify and causal conv1d
#   qsa    Qwen sparse attention decode kernels and QSA metadata builder
#   se     fused shared-expert decode kernels
#   dense  int8-shadow and fp16 skinny GEMV for the dense linears
#   ar     rdna_ar push all-reduce
#   moeu   fp16 (unquantized) MoE HIP kernels: the MTP drafter's experts
#   v1     V1 worker / sampler / speculative-decode HIP ops, mrope, vision
#          pos-embed and the other on_rdna2_hip() sites
RDNA3_HIP_FAMILIES = frozenset({"hc", "gdn", "qsa", "se", "dense", "ar", "v1", "moeu"})

# One op per family that must be registered in ``_rocm_C`` for the family to work.
# They sit in the gfx1030-only registration block of csrc/rocm/torch_bindings.cpp: the
# fat ``gfx1030;gfx1100`` build carries them, a gfx1100-only build does not. hc, se
# and dense have no entry (their call sites probe the ops they use); ar probes in
# rdna_all_reduce.
_FAMILY_PROBE_OP = {
    "gdn": "gdn_decode_rdna2",
    "qsa": "qsa_mqa_paged_rdna2",
    "moeu": "moe_gate_up_silu_unquantized_rdna2",
    "v1": "apply_write_rdna2",
}


def on_rdna_family() -> bool:
    """ROCm on RDNA2 (gfx10x) or RDNA3/4 (gfx11/gfx12, excluding CDNA gfx1250).

    Never imports ``vllm.platforms.rocm`` on other platforms.
    """
    global _ON_RDNA_FAMILY
    if _ON_RDNA_FAMILY is None:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx10x, on_rdna

            _ON_RDNA_FAMILY = on_gfx10x() or on_rdna()
        else:
            _ON_RDNA_FAMILY = False
    return _ON_RDNA_FAMILY


def on_rdna2() -> bool:
    """ROCm on RDNA2 (gfx10x)."""
    global _ON_RDNA2
    if _ON_RDNA2 is None:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx10x

            _ON_RDNA2 = on_gfx10x()
        else:
            _ON_RDNA2 = False
    return _ON_RDNA2


def on_rdna3() -> bool:
    """ROCm on a gfx1100 (RDNA3, e.g. Radeon PRO W7800) device."""
    global _ON_RDNA3
    if _ON_RDNA3 is None:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx1100

            _ON_RDNA3 = on_gfx1100()
        else:
            _ON_RDNA3 = False
    return _ON_RDNA3


def _family_built(family: str) -> bool:
    """True when this build registers the probe op of ``family`` (or has none)."""
    import torch

    op = _FAMILY_PROBE_OP.get(family)
    return op is None or hasattr(getattr(torch.ops, "_rocm_C", None), op)


def _rdna3_hip_families() -> frozenset[str]:
    """Families named in ``VLLM_RDNA3_HIP_OPS`` ("all" = every family). Read once.

    A named family whose ops the build lacks is an error (the call sites would
    fail with a missing-op AttributeError at the first launch); "all" keeps the
    families the build carries and warns about the rest.
    """
    global _RDNA3_HIP_FAMILIES
    if _RDNA3_HIP_FAMILIES is None:
        from vllm.envs import VLLM_RDNA3_HIP_OPS

        names = {n.strip().lower() for n in VLLM_RDNA3_HIP_OPS.split(",") if n.strip()}
        names.discard("none")
        unknown = names - RDNA3_HIP_FAMILIES - {"all"}
        if unknown:
            raise ValueError(
                f"VLLM_RDNA3_HIP_OPS: unknown families {sorted(unknown)}; "
                f"known: {sorted(RDNA3_HIP_FAMILIES)} (or 'all' / 'none')"
            )
        missing = sorted(f for f in names - {"all"} if not _family_built(f))
        if missing:
            raise RuntimeError(
                "VLLM_RDNA3_HIP_OPS names families whose ops are not in this "
                "_rocm_C build: "
                + ", ".join(f"{f} ({_FAMILY_PROBE_OP[f]})" for f in missing)
                + ". Build the gfx1030 HIP sources too "
                "(PYTORCH_ROCM_ARCH='gfx1030;gfx1100')."
            )
        selected = RDNA3_HIP_FAMILIES if "all" in names else frozenset(names)
        usable = frozenset(f for f in selected if _family_built(f))
        if usable != selected:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "VLLM_RDNA3_HIP_OPS=all: families %s are not in this _rocm_C build "
                "and stay off",
                sorted(selected - usable),
            )
        _RDNA3_HIP_FAMILIES = usable
    return _RDNA3_HIP_FAMILIES


def on_rdna3_hip(family: str) -> bool:
    """True iff ``family`` of the RDNA HIP kernels is opted in on a gfx1100 device.

    Always False on every other device, so a call site that keeps its gfx10x
    test and adds ``or on_rdna3_hip("hc")`` behaves exactly as before there.
    """
    if family not in RDNA3_HIP_FAMILIES:
        raise ValueError(f"unknown RDNA3 HIP family {family!r}")
    return on_rdna3() and family in _rdna3_hip_families()


def on_rdna2_hip(family: str | None = "v1") -> bool:
    """True iff the RDNA2 HIP kernel set is enabled on this process.

    The gfx1030 HIP ports are on by default (``VLLM_RDNA_ALL_HIP=1``); set the
    flag to ``0`` to run the Triton kernels gfx10x ran before the port. On a
    gfx1100 device the call is true only when ``family`` (default ``v1``: the V1
    worker / sampler / speculative-decode ops) is opted in with
    ``VLLM_RDNA3_HIP_OPS``; ``family=None`` keeps the gfx10x-only meaning for
    callers that need it (e.g. the Triton JIT warmup, which must stay on for
    gfx1100).
    """
    from vllm.envs import VLLM_RDNA_ALL_HIP

    if on_rdna2() and VLLM_RDNA_ALL_HIP:
        return True
    return family is not None and on_rdna3_hip(family)
