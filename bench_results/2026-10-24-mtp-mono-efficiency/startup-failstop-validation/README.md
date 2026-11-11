# MonoKernel startup fail-stop coverage

Target K1/K2 and the optional MTP drafter emit NaN outputs after a sticky
barrier failure. The startup check previously required
`VLLM_RAISE_ON_LOGIT_NANS=1` only for the optional drafter. The ordinary target
route could therefore run without the output guard. This is a coverage gap;
it does not establish the cause of any particular reported repetition incident.

The existing startup requirement now covers target K1/K2 too. Native MTP2
without `--monokernel` is unaffected. The validated serving recipe already sets
the flag, so this change adds no work to its serving path. The logit check acts
after sampling and before generated tokens are published; it is not a check
before sampling. Configurations without persistent K1/K2 or the optional
drafter do not acquire the requirement.

Eight focused tests passed in the frozen remote environment, with all GPU
visibility masks empty and CUDA never initialized. The patched module was
loaded from a separate scratch directory into the test process; the serving
source and core were untouched. `cpu-tests.json` records the patched source,
test source and unchanged serving source SHA256. The active serving A/B uses
the old startup Python file with the flag already enabled, so it does not
measure the new refusal behavior. The eight startup tests do.

To reproduce against a checkout containing this patch, use its existing test
environment with GPUs hidden:

```bash
HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  python -m pytest tests/models/qwen4_exp/test_amd_mono_rdna2.py \
  -q -p no:cacheprovider -k timeout_nan_failstop
```

All relevant pre-commit hooks passed for the two changed Python files.
