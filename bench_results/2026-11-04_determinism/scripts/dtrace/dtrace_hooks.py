# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Determinism tracer: per-row fingerprints of every module output.

Loaded by ``sitecustomize.py`` in every process when ``DTRACE_DIR`` is set.
After ``GPUModelRunner.load_model`` (V2 runner) it registers forward hooks on
every submodule of the target model (eager mode only: compiled graphs skip
module hooks), wraps the hyper-connection methods, the TP collectives and
``compute_logits``. Each target forward with ``DTRACE_TOKENS`` rows is one
record: for every module call (in execution order) the exact per-row hash of
each output tensor (and of the tensor inputs). Records go to
``$DTRACE_DIR/r<rank>_f<fwd>.pt``.

Files re-read at the start of every traced forward:
  dump_names.txt  one module name per line: its outputs are saved under
                  ``$DTRACE_DIR/dump/``; ``name@rows`` saves only the rows in
                  DTRACE_DUMP_ROWS (comma list) of [tokens, ...] outputs.
  perturb.txt     perturbations of a module's first output, applied in place
                  before it is hashed (same on every rank):
                    name elem ROW COL ULPS   add ULPS to the fp16 magnitude
                    name noise FRAC SEED     +-1 ulp on a random FRAC of the
                                             finite nonzero elements; the
                                             pattern depends on SEED, the
                                             forward index and the name
Env:
  DTRACE_DIR       output directory (required)
  DTRACE_TOKENS    trace only forwards with this many tokens (default 2048);
                   0 = every forward with at least DTRACE_MIN_TOKENS tokens
  DTRACE_MIN_TOKENS  see DTRACE_TOKENS=0 (default 256)
  DTRACE_MAX_FWD   stop after this many traced forwards (default 400)
  DTRACE_INPUTS    1 = also hash the tensor inputs of each call (default 1)
  DTRACE_DUMP_ROWS rows kept by ``name@rows`` dumps
  DTRACE_DUMP_RANKS ranks that save dumps (comma list; default all)
  DTRACE_NO_COLLECTIVES=1  do not wrap the TP collectives
"""

import fnmatch
import importlib.abc
import os
import sys
import time
import zlib

TARGET_MODULE = "vllm.v1.worker.gpu.model_runner"
_installed = False


def _log(msg):
    print(f"[dtrace pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def install_import_hook():
    global _installed
    if _installed:
        return
    _installed = True
    if TARGET_MODULE in sys.modules:
        _patch_runner_module(sys.modules[TARGET_MODULE])
        return
    sys.meta_path.insert(0, _Finder())


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET_MODULE:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is None:
                continue
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                try:
                    _patch_runner_module(module)
                except Exception as e:
                    _log(f"patch failed: {e!r}")

            loader.exec_module = exec_module
            return spec
        return None


def _patch_runner_module(mod):
    cls = mod.GPUModelRunner
    if getattr(cls.load_model, "_dtrace", False):
        return
    orig = cls.load_model

    def load_model(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            Tracer.install(self.model)
        except Exception as e:
            import traceback

            _log(f"install failed: {e!r}\n{traceback.format_exc()}")
        return r

    load_model._dtrace = True
    cls.load_model = load_model
    _log("patched GPUModelRunner.load_model")


class Tracer:
    inst = None

    def __init__(self, model):
        import torch
        import torch.distributed as dist

        self.torch = torch
        self.model = model
        self.dir = os.environ["DTRACE_DIR"]
        os.makedirs(os.path.join(self.dir, "dump"), exist_ok=True)
        self.tokens = int(os.environ.get("DTRACE_TOKENS", "2048"))
        self.min_tokens = int(os.environ.get("DTRACE_MIN_TOKENS", "256"))
        self.cur_n = -1
        self.n_logged = 0
        self.max_fwd = int(os.environ.get("DTRACE_MAX_FWD", "400"))
        self.hash_inputs = os.environ.get("DTRACE_INPUTS", "1") == "1"
        rows = os.environ.get("DTRACE_DUMP_ROWS", "")
        self.dump_rows = [int(r) for r in rows.split(",") if r.strip()]
        ranks = os.environ.get("DTRACE_DUMP_RANKS", "")
        self.dump_ranks = {int(r) for r in ranks.split(",") if r.strip()}
        self.dump_file = os.path.join(self.dir, "dump_names.txt")
        self.perturb_file = os.path.join(self.dir, "perturb.txt")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.active = False
        self.fwd = -1
        self.rec = []
        self.ncall = 0
        self.dump = {}
        self.perturb = {}
        self.applied = []
        self.weights = {}
        self.names = []
        self.t0 = 0.0

    # ------------------------------------------------------------------ setup
    @classmethod
    def install(cls, model):
        t = cls(model)
        cls.inst = t
        n_hooks = 0
        for name, mod in model.named_modules():
            if mod is model:
                mod.register_forward_pre_hook(t._root_pre, with_kwargs=True)
                mod.register_forward_hook(t._root_post, with_kwargs=True)
                continue
            mod.register_forward_hook(t._mk_post(name), with_kwargs=True)
            t.names.append(name)
            n_hooks += 1
            if type(mod).__name__ == "GatedResidual":
                for meth in ("mix", "combine_and_mix", "combine"):
                    if hasattr(mod, meth):
                        t._wrap_method(mod, meth, f"{name}.{meth}")
        if hasattr(model, "compute_logits"):
            t._wrap_method(model, "compute_logits", "compute_logits")
        try:
            if os.environ.get("DTRACE_NO_COLLECTIVES") == "1":
                raise RuntimeError("collectives not traced (DTRACE_NO_COLLECTIVES=1)")
            from vllm.distributed.parallel_state import GroupCoordinator

            for meth in ("all_reduce", "all_gather", "reduce_scatter"):
                if hasattr(GroupCoordinator, meth) and not getattr(
                    getattr(GroupCoordinator, meth), "_dtrace", False
                ):
                    t._wrap_class_method(GroupCoordinator, meth)
        except Exception as e:
            _log(f"collective wrap failed: {e!r}")
        _log(
            f"installed on rank {t.rank}: {n_hooks} module hooks, "
            f"tokens={t.tokens} dir={t.dir}"
        )

    def _wrap_method(self, obj, meth, name):
        orig = getattr(obj, meth)
        tracer = self

        def wrapped(*a, **k):
            out = orig(*a, **k)
            if tracer.active:
                tracer._call(name, a, k, out, all_inputs=True)
            return out

        setattr(obj, meth, wrapped)
        self.names.append(name)

    def _wrap_class_method(self, cls, meth):
        orig = getattr(cls, meth)
        tracer = self

        def wrapped(group, *a, **k):
            out = orig(group, *a, **k)
            if tracer.active:
                gname = getattr(group, "unique_name", "?")
                tracer._call(f"{meth}#{gname}", a, k, out, all_inputs=False)
            return out

        wrapped._dtrace = True
        setattr(cls, meth, wrapped)

    # ----------------------------------------------------------------- hooks
    def _root_pre(self, module, args, kwargs):
        torch = self.torch
        pos = kwargs.get("positions")
        if pos is None and len(args) > 1:
            pos = args[1]
        n = pos.shape[-1] if isinstance(pos, torch.Tensor) else -1
        want = n == self.tokens if self.tokens > 0 else n >= self.min_tokens
        if n >= 64 and self.n_logged < 400:
            self.n_logged += 1
            _log(f"root forward n={n} traced={want and self.fwd + 1 < self.max_fwd}")
        if want and self.fwd + 1 < self.max_fwd:
            self.active = True
            self.cur_n = n
            self.fwd += 1
            self.rec = []
            self.ncall = 0
            self.applied = []
            self.t0 = time.time()
            self._load_control_files()
            ids = kwargs.get("input_ids", args[0] if args else None)
            for key, val in (
                ("input_ids", ids),
                ("positions", pos),
                ("inputs_embeds", kwargs.get("inputs_embeds")),
                ("query_start_loc", kwargs.get("query_start_loc")),
                ("ngram_context", kwargs.get("ngram_context")),
            ):
                if isinstance(val, torch.Tensor):
                    self._record(f"root.{key}", "in", 0, val)
                    self._dump_tensor(f"root.{key}", "in", 0, val, mode="")
        else:
            self.active = False

    def _root_post(self, module, args, kwargs, output):
        if not self.active:
            return
        self._call("root", (), {}, output, all_inputs=False, hash_in=False)
        self._flush()

    def _mk_post(self, name):
        def hook(module, args, kwargs, output):
            if self.active:
                if name in self.perturb:
                    self._perturb(name, output)
                self._call(name, args, kwargs, output, all_inputs=True)

        return hook

    def _call(self, name, args, kwargs, output, all_inputs, hash_in=True):
        self.ncall += 1
        if self.hash_inputs and hash_in:
            ins = self._tensors(args) + self._tensors(kwargs)
            if not all_inputs:
                ins = ins[:1]
            for i, x in enumerate(ins):
                self._record(name, "in", i, x)
        for i, o in enumerate(self._tensors(output)):
            self._record(name, "out", i, o)
            if name in self.dump:
                self._dump_tensor(name, "out", i, o, mode=self.dump[name])
        if name == "compute_logits":
            self._flush()

    # ----------------------------------------------------------- perturbation
    def _perturb(self, name, output):
        torch = self.torch
        outs = self._tensors(output)
        if not outs or outs[0].dtype != torch.float16:
            _log(f"perturb {name}: first output is not fp16, skipped")
            return
        x = outs[0]
        bits = x.view(torch.int16)
        for spec in self.perturb[name]:
            kind = spec[0]
            if kind == "elem":
                row, col, ulps = int(spec[1]), int(spec[2]), int(spec[3])
                flat = bits.reshape(bits.shape[0], -1)
                v = int(flat[row, col].item()) & 0xFFFF
                sign, mag = v & 0x8000, v & 0x7FFF
                mag = max(1, min(0x7BFF, mag + ulps))
                new = sign | mag
                flat[row, col] = new - 0x10000 if new >= 0x8000 else new
                self.applied.append((name, "elem", row, col, ulps, v, new))
            elif kind == "noise":
                frac, seed = float(spec[1]), int(spec[2])
                g = torch.Generator(device=x.device)
                g.manual_seed(
                    seed * 100003 + self.fwd * 7919 + zlib.crc32(name.encode())
                )
                v = bits.to(torch.int32) & 0xFFFF
                sign, mag = v & 0x8000, v & 0x7FFF
                pick = torch.rand(v.shape, generator=g, device=x.device) < frac
                up = torch.rand(v.shape, generator=g, device=x.device) < 0.5
                ok = pick & (mag >= 2) & (mag <= 0x7BFE)
                new = sign | torch.where(up, mag + 1, mag - 1)
                new = torch.where(ok, new, v)
                new = torch.where(new >= 0x8000, new - 0x10000, new)
                bits.copy_(new.to(torch.int16))
                self.applied.append((name, "noise", frac, seed, int(ok.sum().item())))
            else:
                _log(f"perturb {name}: unknown kind {kind}")

    # --------------------------------------------------------------- helpers
    def _tensors(self, obj):
        torch = self.torch
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, dict):
            obj = list(obj.values())
        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                if isinstance(v, torch.Tensor):
                    out.append(v)
                elif isinstance(v, (list, tuple)):
                    out.extend(x for x in v if isinstance(x, torch.Tensor))
            return out
        return []

    def _w(self, n, device):
        key = (n, str(device))
        w = self.weights.get(key)
        if w is None:
            torch = self.torch
            g = torch.Generator(device="cpu")
            g.manual_seed(7919 + n)
            w = torch.randint(1, 32768, (n,), generator=g, dtype=torch.int32).to(device)
            self.weights[key] = w
        return w

    def _rowhash(self, t):
        torch = self.torch
        x = t.detach()
        if x.dim() >= 1 and x.shape[0] == self.cur_n:
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(1, -1)
        if x.numel() == 0:
            return torch.zeros((x.shape[0],), dtype=torch.int64)
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype == torch.bool or x.element_size() < 2:
            v = x.to(torch.int16)
        else:
            v = x.view(torch.int16)
        n = v.shape[1]
        w = self._w(n, v.device)
        out = torch.empty((v.shape[0],), dtype=torch.int64, device=v.device)
        step = max(1, (1 << 22) // n)
        for i in range(0, v.shape[0], step):
            vi = v[i : i + step].to(torch.int32)
            out[i : i + step] = (vi * w).sum(-1, dtype=torch.int64)
        return out

    def _record(self, name, kind, idx, t):
        torch = self.torch
        if not isinstance(t, torch.Tensor):
            return
        try:
            h = self._rowhash(t)
        except Exception as e:
            _log(f"hash failed for {name}: {e!r}")
            return
        self.rec.append((self.ncall, name, kind, idx, tuple(t.shape), str(t.dtype), h))

    def _dump_tensor(self, name, kind, idx, t, mode):
        if self.dump_ranks and self.rank not in self.dump_ranks:
            return
        x = t.detach()
        if mode == "rows" and self.dump_rows and x.dim() >= 1:
            if x.shape[0] != self.cur_n:
                return
            x = x[self.dump_rows]
        path = os.path.join(
            self.dir,
            "dump",
            f"r{self.rank}_f{self.fwd:03d}_c{self.ncall:05d}_{name}_{kind}{idx}.pt",
        )
        self.torch.save(x.to("cpu", copy=True), path)

    def _match(self, pattern):
        if not any(ch in pattern for ch in "*?["):
            return [pattern]
        return [n for n in self.names if fnmatch.fnmatchcase(n, pattern)]

    def _load_control_files(self):
        # Lines are module names or fnmatch patterns over the hooked names.
        self.dump = {}
        try:
            with open(self.dump_file) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        pattern, _, mode = ln.partition("@")
                        for name in self._match(pattern):
                            self.dump[name] = mode
        except FileNotFoundError:
            pass
        self.perturb = {}
        try:
            with open(self.perturb_file) as f:
                for ln in f:
                    parts = ln.split()
                    if len(parts) >= 2 and not parts[0].startswith("#"):
                        for name in self._match(parts[0]):
                            self.perturb.setdefault(name, []).append(parts[1:])
        except FileNotFoundError:
            pass
        if self.perturb or self.dump:
            _log(
                f"fwd {self.fwd}: dump {len(self.dump)} names, "
                f"perturb {len(self.perturb)} names"
            )

    def _flush(self):
        for i, (ncall, name, kind, idx, shape, dtype, h) in enumerate(self.rec):
            if hasattr(h, "cpu"):
                self.rec[i] = (ncall, name, kind, idx, shape, dtype, h.cpu().numpy())
        path = os.path.join(self.dir, f"r{self.rank}_f{self.fwd:03d}.pt")
        tmp = path + ".tmp"
        self.torch.save(
            {
                "rank": self.rank,
                "fwd": self.fwd,
                "time": time.time(),
                "elapsed": time.time() - self.t0,
                "n": self.cur_n,
                "perturb": list(self.applied),
                "rec": self.rec,
            },
            tmp,
        )
        os.replace(tmp, path)
