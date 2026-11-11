# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dry-run both qualification arms without a GPU.

Stubs the server, probes and kernel log so the harness control flow is
exercised end to end: the FP16 reference arm, then the INT4 candidate arm
against that reference (diagnostic-only and strict paths).
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["HIP_VISIBLE_DEVICES"] = "2,3,4,5"

spec = importlib.util.spec_from_file_location("qs", HERE / "qualify_serving.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)

FAKE_BOOT = "test-boot-0000"
BENCH_KEYS = (
    "failed",
    "completed",
    "input_lens",
    "output_lens",
    "ttfts",
    "latencies",
    "itls",
    "duration",
    "total_input_tokens",
    "total_output_tokens",
    "p99_ttft_ms",
)


class FakeServer:
    pid = 4242

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def install_stubs(kernel_text):
    qs.boot_id = lambda: FAKE_BOOT
    qs.manifest = lambda: {"tree": "stub"}
    # The kernel log must grow between the "before" and "after" reads, otherwise
    # the delta the harness computes is empty.
    kernel_calls = {"n": 0}

    def fake_kernel_log():
        kernel_calls["n"] += 1
        if kernel_calls["n"] == 1:
            return "boot line\n"
        return "boot line\n" + kernel_text

    qs.kernel_log = fake_kernel_log
    qs.check_gpu_owners = lambda *a, **k: None

    def fake_popen(command, stdout=None, **kwargs):
        if stdout is not None:
            stdout.write(
                "RDNA_ATTN MTP graph Captur TunableOp lookup enabled tuning off\n"
            )
            if any("int4_fp16_block" in str(x) for x in command):
                stdout.write("kv_cache_dtype int4_fp16_block\n")
            stdout.flush()
        return FakeServer()

    qs.subprocess.Popen = fake_popen
    qs.os.killpg = lambda *a, **k: None

    def fake_request(url, body=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "q27d"}]}
        return True

    qs.request = fake_request

    def fake_run(command, out, env):
        command = [str(x) for x in command]
        out.write_text("stub\n")
        if "bench" in command and "serve" in command:
            name = command[command.index("--result-filename") + 1]
            count = int(command[command.index("--num-prompts") + 1])
            out_path = Path(command[command.index("--result-dir") + 1]) / name
            result = {
                "failed": 0,
                "completed": count,
                "input_lens": [1024] * count,
                "output_lens": [512] * count,
                "ttfts": [1.0] * count,
                "latencies": [10.0] * count,
                "itls": [[0.03] * 10] * count,
                "duration": 5.0,
                "total_input_tokens": 1024 * count,
                "total_output_tokens": 512 * count,
                "p99_ttft_ms": 1000.0,
            }
            out_path.write_text(json.dumps(result))
        elif "ppl_probe" in " ".join(command):
            out.write_text("PPL 4.9400 over 6 documents\n")

    qs.run = fake_run

    def fake_normal_eos(base, model, output):
        Path(output).write_text(json.dumps({"status": "PASS", "requests": 48}))

    qs.normal_eos = fake_normal_eos


def build_args(output, cache_dtype, reference=None, diagnostic=False):
    argv = [
        "qualify_serving.py",
        "--cache-dtype",
        cache_dtype,
        "--output",
        str(output),
        "--model",
        str(MODEL),
        "--recipe",
        "27b-awq",
        "--layout",
        "LBHNC",
        "--kv-bytes",
        "8000000000",
        "--hw-queues",
        "2",
        "--port",
        "18134",
        "--run",
    ]
    if reference:
        argv += ["--reference", str(reference)]
    if diagnostic:
        argv += ["--diagnostic-reference"]
    return argv


MODEL = Path("/tmp/pr134-dryrun-model")


def make_model():
    MODEL.mkdir(parents=True, exist_ok=True)
    (MODEL / "config.json").write_text(json.dumps({"hidden_size": 1}))
    return MODEL


def run_arm(argv):
    sys.argv = argv
    try:
        qs.main()
        return None
    except AssertionError as error:
        return str(error)
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"


def main():
    make_model()
    root = Path(tempfile.mkdtemp())
    clean_kernel = "boot line\nnothing wrong here\n"
    corrected_kernel = (
        "boot line\n"
        "[100.5] {1}[Hardware Error]: It has been corrected by h/w and requires "
        "no further action\n"
        "[100.5] {1}[Hardware Error]: event severity: corrected\n"
        "[100.5] {1}[Hardware Error]:  Error 0, type: corrected\n"
        "[100.6] pcieport 0000:40:03.1: aer_status: 0x00000040\n"
        "[100.6] pcieport 0000:40:03.1:    [ 6] BadTLP\n"
    )

    # ---- Arm 1: FP16 reference, clean kernel log -> strict-pass ----
    install_stubs(clean_kernel)
    ref = root / "fp16-clean"
    error = run_arm(build_args(ref, "auto"))
    assert error is None, f"FP16 arm failed: {error}"
    status = json.loads((ref / "status.json").read_text())
    qualification = json.loads((ref / "qualification.json").read_text())
    print("FP16 clean:", status, qualification["qualification_mode"])
    assert status == {"status": "PASS", "cleanup": "PASS"}, status
    assert qualification["qualification_mode"] == "strict-pass"

    # ---- Arm 2: FP16 reference, corrected-link noise -> raw FAIL, diagnostic-only ----
    install_stubs(corrected_kernel)
    ref2 = root / "fp16-corrected"
    error = run_arm(build_args(ref2, "auto"))
    print("FP16 corrected raised (raw gate must fail):", (error or "")[:80])
    assert error is not None and "GPU/PCIe fault" in error, error
    status2 = json.loads((ref2 / "status.json").read_text())
    qualification2 = json.loads((ref2 / "qualification.json").read_text())
    print(
        "FP16 corrected:",
        status2,
        qualification2["qualification_mode"],
        "corrected_groups=",
        qualification2["kernel_events"]["corrected_event_groups"],
    )
    assert status2 == {"status": "FAIL", "cleanup": "PASS"}, status2
    assert qualification2["qualification_mode"] == "diagnostic-only"
    assert qualification2["kernel_events"]["corrected_event_groups"] == 1
    assert qualification2["hardware_attribution"] == "contaminated-corrected-link-noise"

    # ---- Arm 3: INT4 candidate vs corrected reference, no flag -> refused ----
    install_stubs(clean_kernel)
    cand = root / "int4-strict-refused"
    error = run_arm(build_args(cand, "int4_fp16_block", reference=ref2))
    print("INT4 vs corrected ref without flag:", (error or "")[:110])
    assert error is not None and "--diagnostic-reference" in error, error

    # ---- Arm 4: INT4 candidate with --diagnostic-reference -> runs ----
    install_stubs(corrected_kernel)
    cand2 = root / "int4-diagnostic"
    error = run_arm(
        build_args(cand2, "int4_fp16_block", reference=ref2, diagnostic=True)
    )
    assert error is not None and "GPU/PCIe fault" in error, error
    status3 = json.loads((cand2 / "status.json").read_text())
    qualification3 = json.loads((cand2 / "qualification.json").read_text())
    run3 = json.loads((cand2 / "run.json").read_text())
    print(
        "INT4 diagnostic:",
        status3,
        qualification3["qualification_mode"],
        "reference_mode=",
        qualification3["reference_mode"],
    )
    assert status3 == {"status": "FAIL", "cleanup": "PASS"}, status3
    assert qualification3["qualification_mode"] == "diagnostic-only"
    assert qualification3["reference_mode"] == "diagnostic-only", qualification3
    assert run3["reference_mode"] == "diagnostic-only", run3
    assert (cand2 / "ppl-paired.json").exists(), "paired PPL artifact missing"
    paired = json.loads((cand2 / "ppl-paired.json").read_text())
    print("paired PPL:", paired)
    assert paired["ratio"] == 1.0

    # ---- Arm 5: INT4 candidate against a strict reference -> no flag needed ----
    install_stubs(clean_kernel)
    cand3 = root / "int4-strict"
    error = run_arm(build_args(cand3, "int4_fp16_block", reference=ref))
    assert error is None, f"INT4 strict arm failed: {error}"
    qualification4 = json.loads((cand3 / "qualification.json").read_text())
    print(
        "INT4 strict:",
        json.loads((cand3 / "status.json").read_text()),
        qualification4["qualification_mode"],
        "reference_mode=",
        qualification4["reference_mode"],
    )
    assert qualification4["qualification_mode"] == "strict-pass"
    assert qualification4["reference_mode"] == "strict-pass"

    # ---- Arm 6: fatal kernel event -> unusable, never diagnostic-only ----
    fatal_kernel = (
        "boot line\n[200.1] amdgpu 0000:47:00.0: GPU reset: GPU stopped processing\n"
    )
    install_stubs(fatal_kernel)
    reffatal = root / "fp16-fatal"
    error = run_arm(build_args(reffatal, "auto"))
    statusf = json.loads((reffatal / "status.json").read_text())
    qualf = json.loads((reffatal / "qualification.json").read_text())
    print("FP16 fatal:", statusf, qualf["qualification_mode"])
    assert qualf["qualification_mode"] == "fail", qualf
    assert qualf["kernel_events"]["fatal_events"], qualf

    install_stubs(clean_kernel)
    cand4 = root / "int4-vs-fatal"
    error = run_arm(
        build_args(cand4, "int4_fp16_block", reference=reffatal, diagnostic=True)
    )
    print("INT4 vs fatal ref (diagnostic flag):", (error or "")[:110])
    assert error is not None and "fatal device events" in error, error

    shutil.rmtree(root, ignore_errors=True)
    print("\nALL DRY-RUN ARM TESTS PASS")


if __name__ == "__main__":
    main()
