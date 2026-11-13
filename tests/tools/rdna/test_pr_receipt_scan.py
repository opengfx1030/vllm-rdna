# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Added-line scan of tools/rdna/pr_receipt_scan.py on synthetic trees.

Everything runs on strings and temporary directories: no GPU, no torch, no
network.
"""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan = _load("pr_receipt_scan", "tools/rdna/pr_receipt_scan.py")


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _all_lines(text: str) -> list[int]:
    return list(range(1, len(scan.split_lines(text)) + 1))


def c_hits(text: str, added: list[int] | None = None) -> list[tuple[int, str]]:
    """(line, pattern) of every finding with every line added by default."""
    lines = _all_lines(text) if added is None else added
    return [(n, name) for n, name, _ in scan.scan_c(text, lines)]


def c_patterns(text: str) -> list[str]:
    return [name for _, name in c_hits(text)]


# ------------------------------------------------------------- added lines


def test_added_lines_insert_replace_and_head_numbers():
    base = ["a", "b", "c", "d"]
    head = ["a", "NEW", "b", "C2", "d", "tail"]
    assert scan.added_lines(base, head) == [2, 4, 6]


def test_added_lines_new_file_deleted_file_and_equal_files():
    assert scan.added_lines([], ["x", "y"]) == [1, 2]
    assert scan.added_lines(["x", "y"], []) == []
    assert scan.added_lines(["x", "y"], ["x", "y"]) == []


def test_added_lines_moved_block_counts_by_matching_blocks():
    base = ["one", "two", "three"]
    head = ["one", "two", "inserted", "three"]
    assert scan.added_lines(base, head) == [3]


def test_split_lines_keeps_inner_blank_lines_and_drops_trailing_newline():
    assert scan.split_lines("a\n\nb\n") == ["a", "", "b"]
    assert scan.split_lines("") == []
    assert scan.split_lines("a") == ["a"]


# ----------------------------------------------------------------- python


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("stream.synchronize()", ["synchronize"]),
        ("event.synchronize ()", ["synchronize"]),
        ("torch.accelerator.synchronize()", ["synchronize"]),
        ("synchronize()", ["synchronize"]),
        ("x = t.item()", ["item"]),
        ("x = t.item ()", ["item"]),
        ("x = t.cpu()", ["cpu"]),
        ("x = t.tolist()", ["tolist"]),
        ('x = t.to("cpu")', ["to-cpu"]),
        ("x = t.to('cpu')", ["to-cpu"]),
        ('x = t.to(device="cpu")', ["to-cpu"]),
        ('x = t.to( device = "cpu" )', ["to-cpu"]),
        ('x = t.to(torch.device("cpu"))', ["to-cpu"]),
        ('x = t.to(device=torch.device( "cpu" ))', ["to-cpu"]),
        ("y = x.copy_(z, non_blocking=False)", ["blocking-copy"]),
        ("y = x.to(d, non_blocking = False)", ["blocking-copy"]),
        ("v = t.cpu().tolist()", ["cpu", "tolist"]),
    ],
)
def test_python_patterns_positive(line, expected):
    assert scan.python_patterns(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "x = t.item_x()",
        "x = t.item",
        "x = items(t)",
        "my_synchronize_x()",
        "resynchronize(x)",
        "x = synchronized(y)",
        "x = t.cpu_count()",
        "x = t.cpu",
        "x = t.tolist",
        "x = t.tolist_all()",
        'x = t.to("cuda")',
        'x = t.to(device="cuda:0")',
        'x = t.to("cpu_x")',
        "y = x.copy_(z, non_blocking=True)",
        "",
        "   ",
    ],
)
def test_python_patterns_negative(line):
    assert scan.python_patterns(line) == []


def test_python_comment_only_line_is_skipped():
    assert scan.python_patterns("# stream.synchronize()") == []
    assert scan.python_patterns("    # x = t.item()") == []


def test_python_trailing_comment_is_dropped():
    assert scan.python_patterns("x = 1  # t.item() was removed") == []
    assert scan.python_patterns("x = t.item()  # why") == ["item"]


def test_python_hash_inside_string_is_not_a_comment():
    assert scan.python_patterns('s = "#"; x = t.item()') == ["item"]
    assert scan.python_patterns("s = '#'; x = t.cpu()") == ["cpu"]
    assert scan.python_patterns('s = """#"""; x = t.tolist()') == ["tolist"]
    assert scan.python_patterns('s = "a \\" # b"; x = t.item()') == ["item"]


def test_strip_py_comment():
    assert scan.strip_py_comment("a = 1  # c") == "a = 1  "
    assert scan.strip_py_comment('a = "#"') == 'a = "#"'
    assert scan.strip_py_comment("a = ''  # c") == "a = ''  "


def test_scan_python_reports_only_added_lines_with_stripped_text():
    lines = ["a = t.item()", "    b = t.cpu()  ", "c = 1"]
    assert scan.scan_python(lines, [2, 3]) == [(2, "cpu", "b = t.cpu()")]


def test_scan_python_text_is_clipped_to_160_chars():
    line = "x = t.item()" + " " + "y" * 300
    [(_, _, text)] = scan.scan_python([line], [1])
    assert len(text) == 160
    assert text.startswith("x = t.item()")


# ------------------------------------------------------------ csrc host-sync


@pytest.mark.parametrize(
    "line",
    [
        "hipStreamSynchronize(stream);",
        "cudaStreamSynchronize (stream);",
        "hipDeviceSynchronize();",
        "cudaDeviceSynchronize();",
        "hipEventSynchronize(ev);",
        "hipMemcpy(dst, src, n, hipMemcpyDeviceToHost);",
        "cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);",
    ],
)
def test_host_sync_positive(line):
    assert c_hits(line + "\n") == [(1, "host-sync")]


@pytest.mark.parametrize(
    "line",
    [
        "hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, stream);",
        "cudaMemcpyAsync(dst, src, n, kind, stream);",
        "hipStreamSynchronizeX(stream);",
        "my_hipDeviceSynchronize(x);",
        "// hipStreamSynchronize(stream);",
        "/* hipDeviceSynchronize(); */",
        "int x = 1;  // hipDeviceSynchronize();",
        "int x = 1; /* hipMemcpy(a, b, n, k); */",
        'printf("hipDeviceSynchronize(");',
        "hipMemcpyKind kind = hipMemcpyDeviceToHost;",
    ],
)
def test_host_sync_negative(line):
    assert c_hits(line + "\n") == []


def test_host_sync_inside_block_comment_over_lines_is_skipped():
    text = "/*\n hipDeviceSynchronize();\n*/\nhipDeviceSynchronize();\n"
    assert c_hits(text) == [(4, "host-sync")]


def test_host_sync_only_on_added_lines():
    text = "hipDeviceSynchronize();\nhipStreamSynchronize(s);\n"
    assert c_hits(text, added=[2]) == [(2, "host-sync")]


# ----------------------------------------------------------------- launches


def test_launch_with_two_params_is_null_stream():
    text = "k<<<grid, block>>>(a, b);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_launch_with_three_params_is_null_stream():
    text = "k<<<grid, block, shmem>>>(a);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_launch_with_one_param_is_null_stream():
    text = "k<<<grid>>>(a);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_launch_with_stream_and_check_is_clean():
    text = "k<<<grid, block, 0, stream>>>(a);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == []


@pytest.mark.parametrize("fourth", ["0", "nullptr", "NULL", "hipStreamDefault", " 0 "])
def test_launch_with_null_fourth_param(fourth):
    text = f"k<<<grid, block, shmem, {fourth}>>>(a);\nC10_HIP_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_launch_with_current_stream_call_is_clean():
    text = (
        "k<<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(a);\n"
        "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    )
    assert c_hits(text) == []


def test_split_params_dim3_and_calls():
    assert scan.split_params("dim3(a, b), dim3(c), 0, stream") == [
        "dim3(a, b)",
        "dim3(c)",
        "0",
        "stream",
    ]
    assert scan.split_params(
        "grid, block, shmem, at::cuda::getCurrentCUDAStream()"
    ) == ["grid", "block", "shmem", "at::cuda::getCurrentCUDAStream()"]


def test_split_params_template_and_paren_containing_params():
    assert scan.split_params("std::min<int>(n, 4), 256, 0, stream") == [
        "std::min<int>(n, 4)",
        "256",
        "0",
        "stream",
    ]
    assert scan.split_params("g, b, 0, pick<int, 3>(x, y)") == [
        "g",
        "b",
        "0",
        "pick<int, 3>(x, y)",
    ]
    assert scan.split_params("g, b, f(a, (b, c)), s") == ["g", "b", "f(a, (b, c))", "s"]


def test_split_params_operators_do_not_open_template_levels():
    assert scan.split_params("n<<2, b, 0, s") == ["n<<2", "b", "0", "s"]
    assert scan.split_params("a<=b, c, d, e") == ["a<=b", "c", "d", "e"]
    assert scan.split_params("p->x, c, d, e") == ["p->x", "c", "d", "e"]
    assert scan.split_params("a > b ? x : y, c") == ["a > b ? x : y", "c"]


def test_split_params_nested_templates_and_empty():
    assert scan.split_params("A<B<int>>(x, y), b") == ["A<B<int>>(x, y)", "b"]
    assert scan.split_params("") == []
    assert scan.split_params("   ") == []


def test_launch_with_template_params_and_stream_is_clean():
    text = (
        "k<<<dim3(a, b), std::min<int>(n, 4), 0, stream>>>(x);\n"
        "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    )
    assert c_hits(text) == []


def test_launch_with_template_params_and_null_stream():
    text = (
        "k<<<dim3(a, b), std::min<int>(n, 4), 0, 0>>>(x);\n"
        "HIP_CHECK(hipGetLastError());\n"
    )
    assert c_hits(text) == [(1, "null-stream")]


def test_launch_template_kernel_name_and_multiline_config():
    text = textwrap.dedent(
        """\
        my_kernel<float, 4><<<
            dim3(grid_x,
                 grid_y),
            dim3(block),
            shmem,
            stream>>>(a,
                      b);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        """
    )
    assert c_hits(text) == []
    [(line, name, shown)] = [
        hit for hit in scan.scan_c(text.replace("stream>>>", "0>>>"), _all_lines(text))
    ]
    assert (line, name) == (1, "null-stream")
    assert shown.startswith("my_kernel<float, 4><<< dim3(grid_x, grid_y), dim3(block)")
    assert shown.endswith("b);")


def test_launch_finding_text_is_one_collapsed_line():
    text = "k<<<g,\n     b>>>(a,\n   c);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    [(line, name, shown)] = scan.scan_c(text, [1])
    assert (line, name, shown) == (1, "null-stream", "k<<<g, b>>>(a, c);")


def test_launch_finding_text_drops_macro_line_continuations():
    text = "#define GO(a) \\\n  k<<<g, \\\n    b>>>(a)\\\n  ;\n"
    [(line, name, shown)] = scan.scan_c(text, [2])[:1]
    assert (line, name) == (2, "null-stream")
    assert shown == "k<<<g, b>>>(a) ;"


def test_launch_check_after_it_and_missing_check():
    ok = "k<<<g, b, 0, s>>>(a);\nint x = 1;\nC10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(ok) == []
    missing = "k<<<g, b, 0, s>>>(a);\nint x = 1;\n"
    assert c_hits(missing) == [(1, "no-launch-check")]


@pytest.mark.parametrize(
    "check",
    [
        "C10_CUDA_KERNEL_LAUNCH_CHECK();",
        "C10_HIP_KERNEL_LAUNCH_CHECK();",
        "hipGetLastError();",
        "cudaGetLastError();",
        "C10_CUDA_CHECK(cudaGetLastError());",
        "HIP_CHECK(hipPeekAtLastError());",
        "CUDA_CHECK(x);",
    ],
)
def test_every_listed_launch_check_counts(check):
    assert c_hits(f"k<<<g, b, 0, s>>>(a);\n{check}\n") == []


def test_launch_check_on_the_same_line_counts():
    assert c_hits("k<<<g, b, 0, s>>>(a); C10_CUDA_KERNEL_LAUNCH_CHECK();\n") == []


def test_launch_check_window_is_twelve_lines():
    filler = "int pad = 0;\n"
    text = "k<<<g, b, 0, s>>>(a);\n" + filler * 11 + "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == []  # the check is 12 lines after the launch
    text = "k<<<g, b, 0, s>>>(a);\n" + filler * 12 + "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "no-launch-check")]  # 13 lines after


def test_launch_check_in_a_comment_does_not_count():
    text = "k<<<g, b, 0, s>>>(a);\n// C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    assert c_hits(text) == [(1, "no-launch-check")]


def test_launch_in_a_comment_is_ignored():
    text = "// k<<<g, b>>>(a);\n/* k<<<g, b>>>(a);\n   k<<<g, b>>>(a); */\n"
    assert c_hits(text) == []


def test_launch_in_a_string_is_ignored():
    assert c_hits('const char* s = "k<<<g, b>>>(a);";\n') == []


def test_comment_markers_inside_strings_do_not_hide_code():
    text = 'const char* u = "http://x"; k<<<g, b>>>(a);\n'
    assert c_patterns(text) == ["null-stream", "no-launch-check"]


def test_unchanged_launch_is_not_reported():
    text = "k<<<g, b>>>(a);\nint x = 1;\nint y = 2;\n"
    assert c_hits(text, added=[3]) == []


def test_launch_touched_by_one_added_line_is_reported_at_its_start_line():
    text = "int pre = 0;\nk<<<g,\n   b>>>(a,\n   c);\nint post = 1;\n"
    assert c_hits(text, added=[4]) == [(2, "null-stream"), (2, "no-launch-check")]
    assert c_hits(text, added=[3]) == [(2, "null-stream"), (2, "no-launch-check")]
    assert c_hits(text, added=[5]) == []
    assert c_hits(text, added=[1]) == []


def test_launch_reported_once_even_if_several_lines_are_added():
    text = "k<<<g,\n   b>>>(a,\n   c);\n"
    assert c_hits(text, added=[1, 2, 3]) == [
        (1, "null-stream"),
        (1, "no-launch-check"),
    ]


def test_two_launches_are_reported_separately():
    text = (
        "k<<<g, b, 0, s>>>(a);\n"
        "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
        "k2<<<g, b>>>(a);\n"
        "C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
    )
    assert c_hits(text) == [(3, "null-stream")]


def test_statement_end_ignores_semicolons_inside_parentheses():
    text = "k<<<g, b, 0, s>>>(for_each([&]() { f(); }));\nint z = 0;\n"
    assert c_hits(text, added=[2]) == []
    assert c_hits(text, added=[1]) == [(1, "no-launch-check")]


def test_launch_without_semicolon_ends_at_its_closing_line():
    text = "k<<<g, b, 0, s>>>(a)\n" + "\n" * 50 + "int x = 0;\n"
    assert c_hits(text, added=[1]) == [(1, "no-launch-check")]
    assert c_hits(text, added=[30]) == []


# ------------------------------------------------------- hipLaunchKernelGGL


def test_ggl_with_stream_and_check_is_clean():
    text = (
        "hipLaunchKernelGGL(k, grid, block, 0, stream, a, b);\n"
        "HIP_CHECK(hipGetLastError());\n"
    )
    assert c_hits(text) == []


@pytest.mark.parametrize("stream", ["0", "nullptr", "NULL"])
def test_ggl_null_stream(stream):
    text = f"hipLaunchKernelGGL(k, g, b, 0, {stream}, a);\nhipGetLastError();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_ggl_with_too_few_arguments_is_null_stream():
    text = "hipLaunchKernelGGL(k, g, b, 0);\nhipGetLastError();\n"
    assert c_hits(text) == [(1, "null-stream")]


def test_ggl_without_launch_check():
    text = "hipLaunchKernelGGL(k, g, b, 0, s, a);\n"
    assert c_hits(text) == [(1, "no-launch-check")]


def test_ggl_multiline_with_template_kernel_name():
    text = (
        "hipLaunchKernelGGL(HIP_KERNEL_NAME(k<int, 4>),\n"
        "                   dim3(a, b), dim3(c), 0, 0,\n"
        "                   x, y);\n"
        "hipGetLastError();\n"
    )
    assert c_hits(text, added=[2]) == [(1, "null-stream")]


def test_ggl_not_added_is_not_reported():
    text = "hipLaunchKernelGGL(k, g, b, 0, 0, a);\nint x;\n"
    assert c_hits(text, added=[2]) == []


# ---------------------------------------------------------------------- step


@pytest.mark.parametrize(
    "path",
    [
        "vllm/model_executor/layers/x.py",
        "vllm/attention/backends/x.py",
        "vllm/v1/attention/backends/x.py",
        "vllm/v1/sample/sampler.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/spec_decode/eagle.py",
        "vllm/v1/executor/abstract.py",
        "vllm/distributed/parallel_state.py",
        "vllm/forward_context.py",
        "vllm/compilation/backends.py",
        "csrc/rocm/skinny_gemms.cu",
        "csrc/x.h",
    ],
)
def test_step_paths(path):
    assert scan.is_step_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "vllm/entrypoints/openai/api_server.py",
        "vllm/v1/engine/core.py",
        "vllm/utils/__init__.py",
        "vllm/config.py",
        "vllm/model_executor_x.py",
    ],
)
def test_non_step_paths(path):
    assert not scan.is_step_path(path)


# ------------------------------------------------------------- tree scan


def test_scan_trees_python_and_csrc_findings_and_sorting(tmp_path):
    base = _write_tree(
        tmp_path / "base",
        {
            "vllm/model_executor/a.py": "x = 1\n",
            "vllm/entrypoints/b.py": "y = 1\n",
            "csrc/k.cu": "int a;\n",
        },
    )
    head = _write_tree(
        tmp_path / "head",
        {
            "vllm/model_executor/a.py": "x = 1\nz = t.item()\nw = t.cpu()\n",
            "vllm/entrypoints/b.py": "y = 1\nq = t.tolist()\n",
            "csrc/k.cu": "int a;\nk<<<g, b>>>(a);\nhipDeviceSynchronize();\n",
        },
    )
    result = scan.scan_trees(base, head)
    keys = [(f.path, f.line, f.pattern, f.step, f.kind) for f in result.findings]
    assert keys == [
        ("csrc/k.cu", 2, "no-launch-check", True, "csrc"),
        ("csrc/k.cu", 2, "null-stream", True, "csrc"),
        ("csrc/k.cu", 3, "host-sync", True, "csrc"),
        ("vllm/model_executor/a.py", 2, "item", True, "python"),
        ("vllm/model_executor/a.py", 3, "cpu", True, "python"),
        ("vllm/entrypoints/b.py", 2, "tolist", False, "python"),
    ]
    assert result.files_changed == {"vllm": 2, "csrc": 1}


def test_scan_trees_sorts_by_not_step_path_line_pattern(tmp_path):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(
        tmp_path / "head",
        {
            "vllm/entrypoints/z.py": "a.item()\n",
            "vllm/model_executor/m.py": "b.item()\nc.cpu()\n",
            "vllm/attention/a.py": "d.tolist()\n",
        },
    )
    result = scan.scan_trees(base, head)
    assert [(f.path, f.line, f.pattern) for f in result.findings] == [
        ("vllm/attention/a.py", 1, "tolist"),
        ("vllm/model_executor/m.py", 1, "item"),
        ("vllm/model_executor/m.py", 2, "cpu"),
        ("vllm/entrypoints/z.py", 1, "item"),
    ]


def test_scan_trees_pattern_sort_within_a_line(tmp_path):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(tmp_path / "head", {"vllm/x.py": "a.cpu().tolist()\n"})
    result = scan.scan_trees(base, head)
    assert [f.pattern for f in result.findings] == ["cpu", "tolist"]


def test_scan_trees_new_and_deleted_files(tmp_path):
    base = _write_tree(
        tmp_path / "base",
        {"vllm/gone.py": "a.item()\n", "csrc/gone.cu": "hipDeviceSynchronize();\n"},
    )
    head = _write_tree(
        tmp_path / "head",
        {"vllm/new.py": "x = 1\ny = a.item()\n", "csrc/new.cu": "int a;\n"},
    )
    result = scan.scan_trees(base, head)
    assert [(f.path, f.line) for f in result.findings] == [("vllm/new.py", 2)]
    assert result.files_changed == {"vllm": 2, "csrc": 2}


def test_scan_trees_equal_files_are_skipped_and_not_counted(tmp_path):
    files = {"vllm/a.py": "x = a.item()\n", "csrc/a.cu": "hipDeviceSynchronize();\n"}
    base = _write_tree(tmp_path / "base", files)
    head = _write_tree(tmp_path / "head", files)
    result = scan.scan_trees(base, head)
    assert result.findings == []
    assert result.files_changed == {"vllm": 0, "csrc": 0}


def test_scan_trees_same_size_different_bytes_is_compared(tmp_path):
    base = _write_tree(tmp_path / "base", {"vllm/a.py": "x = a.itex()\n"})
    head = _write_tree(tmp_path / "head", {"vllm/a.py": "x = a.item()\n"})
    result = scan.scan_trees(base, head)
    assert [(f.line, f.pattern) for f in result.findings] == [(1, "item")]


def test_scan_trees_ignores_non_source_files_and_skipped_directories(tmp_path):
    base = _write_tree(tmp_path / "base", {"vllm/a.py": "x = 1\n"})
    head = _write_tree(
        tmp_path / "head",
        {
            "vllm/a.py": "x = 1\n",
            "vllm/__pycache__/a.py": "x = a.item()\n",
            "vllm/.hidden/b.py": "x = a.item()\n",
            "vllm/notes.txt": "x = a.item()\n",
            "vllm/lib.so": "x = a.item()\n",
            "csrc/.git/c.cu": "hipDeviceSynchronize();\n",
            "csrc/readme.md": "hipDeviceSynchronize();\n",
            "csrc/gen/ops.hip": "hipDeviceSynchronize();\n",
            "tests/t.py": "x = a.item()\n",
            "other/o.py": "x = a.item()\n",
        },
    )
    result = scan.scan_trees(base, head)
    assert [f.path for f in result.findings] == ["csrc/gen/ops.hip"]
    assert result.files_changed == {"vllm": 0, "csrc": 1}


@pytest.mark.parametrize(
    "suffix", [".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hip"]
)
def test_scan_trees_csrc_suffixes(tmp_path, suffix):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(
        tmp_path / "head", {f"csrc/x{suffix}": "hipDeviceSynchronize();\n"}
    )
    assert [f.path for f in scan.scan_trees(base, head).findings] == [f"csrc/x{suffix}"]


def test_scan_trees_python_outside_vllm_and_c_suffix_in_vllm_are_ignored(tmp_path):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(
        tmp_path / "head",
        {"csrc/x.py": "a.item()\n", "vllm/x.cu": "hipDeviceSynchronize();\n"},
    )
    assert scan.scan_trees(base, head).findings == []


def test_scan_trees_reads_invalid_utf8_with_replacement(tmp_path):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(tmp_path / "head", {})
    (head / "vllm").mkdir()
    (head / "vllm" / "x.py").write_bytes(b"s = '\xff'\nv = a.item()\n")
    result = scan.scan_trees(base, head)
    assert [(f.line, f.pattern) for f in result.findings] == [(2, "item")]


def test_scan_trees_head_line_numbers_after_an_insert(tmp_path):
    base = _write_tree(tmp_path / "base", {"vllm/a.py": "a = 1\nb = 2\nc = 3\n"})
    head = _write_tree(
        tmp_path / "head", {"vllm/a.py": "a = 1\nnew = t.item()\nb = 2\nc = 3\n"}
    )
    result = scan.scan_trees(base, head)
    assert [(f.line, f.text) for f in result.findings] == [(2, "new = t.item()")]


def test_scan_trees_old_matching_lines_are_not_reported(tmp_path):
    base = _write_tree(tmp_path / "base", {"vllm/a.py": "v = t.item()\nw = 1\n"})
    head = _write_tree(tmp_path / "head", {"vllm/a.py": "v = t.item()\nw = 2\n"})
    assert scan.scan_trees(base, head).findings == []


# ---------------------------------------------------------------- reports


def _result(findings, changed=None):
    return scan.ScanResult(
        "B",
        "H",
        changed or {"vllm": 1, "csrc": 2},
        sorted(findings, key=lambda f: f.sort_key()),
    )


def _finding(path="vllm/a.py", line=1, pattern="item", text="x.item()", step=False):
    kind = "csrc" if path.startswith("csrc/") else "python"
    return scan.Finding(path, line, kind, pattern, text, step)


def test_markdown_empty_result_is_one_sentence():
    md = scan.render_markdown(_result([], {"vllm": 3, "csrc": 4}))
    assert md == (
        "No added line matches the host-sync or launch patterns "
        "(3 vllm/ and 4 csrc/ files changed).\n"
    )


def test_markdown_non_empty_summary_and_table():
    findings = [
        _finding("csrc/k.cu", 7, "null-stream", "k<<<g, b>>>(a);", True),
        _finding("csrc/k.cu", 7, "no-launch-check", "k<<<g, b>>>(a);", True),
        _finding("vllm/utils.py", 5, "item", "x.item()", False),
        _finding("vllm/utils.py", 9, "item", "y.item()", False),
    ]
    md = scan.render_markdown(_result(findings, {"vllm": 1, "csrc": 1}))
    lines = md.splitlines()
    assert lines[0] == (
        "3 added lines to review (1 in the per-step path): "
        "item x 2, no-launch-check x 1, null-stream x 1 "
        "(1 vllm/ and 1 csrc/ files changed)."
    )
    assert lines[1] == ""
    assert lines[2] == "| where | pattern | code |"
    assert lines[3] == "| --- | --- | --- |"
    rows = lines[4:]
    assert len(rows) == 4
    assert rows[0].startswith("| `csrc/k.cu:7` |")
    assert rows[-1] == "| `vllm/utils.py:9` | item | `y.item()` |"
    assert not md.startswith("#")
    assert md.endswith("\n")


def test_markdown_per_step_rows_come_first():
    findings = [
        _finding("vllm/a_util.py", 1, "cpu", "a.cpu()", False),
        _finding("vllm/model_executor/z.py", 1, "item", "z.item()", True),
    ]
    rows = scan.render_markdown(_result(findings)).splitlines()[4:]
    assert "vllm/model_executor/z.py:1" in rows[0]
    assert "vllm/a_util.py:1" in rows[1]


def test_markdown_truncation():
    findings = [_finding(line=n) for n in range(1, 6)]
    md = scan.render_markdown(_result(findings), max_lines=2)
    lines = md.splitlines()
    rows = [ln for ln in lines if ln.startswith("| `")]
    assert len(rows) == 2
    assert lines[-1] == "3 more in the JSON file."
    assert "5 added lines to review" in lines[0]
    assert "more in" not in scan.render_markdown(_result(findings), max_lines=5)


def test_markdown_max_lines_zero_lists_no_rows():
    md = scan.render_markdown(_result([_finding()]), max_lines=0)
    assert [ln for ln in md.splitlines() if ln.startswith("| `")] == []
    assert md.splitlines()[-1] == "1 more in the JSON file."


def test_markdown_escapes_pipes_and_backticks():
    finding = _finding(text="a = t.item() | b `c` ``d``")
    row = scan.render_markdown(_result([finding])).splitlines()[4]
    assert "\\|" in row
    # a code span fence longer than any backtick run inside keeps it one span
    assert row.endswith("| ``` a = t.item() \\| b `c` ``d`` ``` |")
    assert "\n" not in row


def test_code_span():
    assert scan.code_span("a") == "`a`"
    assert scan.code_span("a | b") == "`a \\| b`"
    assert scan.code_span("a `b` c") == "``a `b` c``"
    assert scan.code_span("`x`") == "`` `x` ``"
    assert scan.code_span("a\tb  c") == "`a b c`"


def test_json_schema_and_counts():
    findings = [
        _finding("csrc/k.cu", 7, "null-stream", "k<<<g, b>>>(a);", True),
        _finding("csrc/k.cu", 7, "no-launch-check", "k<<<g, b>>>(a);", True),
        _finding("vllm/utils.py", 5, "item", "x.item()", False),
    ]
    doc = scan.to_json(_result(findings, {"vllm": 1, "csrc": 1}), max_lines=2)
    assert list(doc) == [
        "base",
        "head",
        "files_changed",
        "findings",
        "counts",
        "step_count",
        "truncated",
    ]
    assert doc["base"] == "B" and doc["head"] == "H"
    assert doc["files_changed"] == {"vllm": 1, "csrc": 1}
    assert doc["counts"] == {"no-launch-check": 1, "null-stream": 1, "item": 1}
    assert doc["step_count"] == 1
    assert doc["truncated"] == 1
    assert len(doc["findings"]) == 3
    assert list(doc["findings"][0]) == [
        "path",
        "line",
        "kind",
        "pattern",
        "text",
        "step",
    ]
    assert [f["step"] for f in doc["findings"]] == [True, True, False]
    assert doc["findings"][0]["kind"] == "csrc"
    assert doc["findings"][2]["kind"] == "python"


# -------------------------------------------------------------------- CLI


def test_cli_stdout_equals_md_file_and_json_is_complete(tmp_path, capsys):
    base = _write_tree(tmp_path / "base", {"vllm/a.py": "x = 1\n"})
    head = _write_tree(
        tmp_path / "head",
        {"vllm/a.py": "x = 1\n" + "".join(f"v{n} = t.item()\n" for n in range(5))},
    )
    md_path = tmp_path / "out.md"
    json_path = tmp_path / "out.json"
    code = scan.main(
        [
            str(base),
            str(head),
            "--json",
            str(json_path),
            "--md",
            str(md_path),
            "--max-lines",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out == md_path.read_text()
    assert out.splitlines()[-1] == "3 more in the JSON file."
    doc = json.loads(json_path.read_text())
    assert len(doc["findings"]) == 5
    assert doc["truncated"] == 3
    assert doc["base"] == str(base) and doc["head"] == str(head)
    assert doc["files_changed"] == {"vllm": 1, "csrc": 0}
    assert doc["findings"][0]["path"] == "vllm/a.py"


def test_cli_identical_trees_print_the_no_match_sentence(tmp_path, capsys):
    files = {"vllm/a.py": "x = t.item()\n", "csrc/a.cu": "k<<<g, b>>>(a);\n"}
    base = _write_tree(tmp_path / "base", files)
    head = _write_tree(tmp_path / "head", files)
    assert scan.main([str(base), str(head)]) == 0
    assert capsys.readouterr().out == (
        "No added line matches the host-sync or launch patterns "
        "(0 vllm/ and 0 csrc/ files changed).\n"
    )


def test_cli_findings_still_exit_zero(tmp_path, capsys):
    base = _write_tree(tmp_path / "base", {})
    head = _write_tree(tmp_path / "head", {"vllm/a.py": "x = t.item()\n"})
    assert scan.main([str(base), str(head)]) == 0
    assert "item x 1" in capsys.readouterr().out


def test_cli_missing_directory_exits_two(tmp_path, capsys):
    head = _write_tree(tmp_path / "head", {})
    assert scan.main([str(tmp_path / "nope"), str(head)]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert scan.main([str(head), str(tmp_path / "nope")]) == 2


def test_cli_file_instead_of_directory_exits_two(tmp_path):
    head = _write_tree(tmp_path / "head", {"f": "x"})
    assert scan.main([str(head / "f"), str(head)]) == 2


def test_cli_bad_arguments_exit_two(tmp_path):
    head = _write_tree(tmp_path / "head", {})
    with pytest.raises(SystemExit) as exc:
        scan.main([str(head)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        scan.main([str(head), str(head), "--max-lines", "-1"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        scan.main([str(head), str(head), "--max-lines", "x"])
    assert exc.value.code == 2


def test_cli_unwritable_output_exits_two(tmp_path, capsys):
    head = _write_tree(tmp_path / "head", {})
    code = scan.main([str(head), str(head), "--md", str(tmp_path / "no" / "dir.md")])
    assert code == 2
    assert capsys.readouterr().out == ""
