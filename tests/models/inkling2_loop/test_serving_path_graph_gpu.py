# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HIP-graph capture and replay of the whole Inkling-2 serving forward on the GPU.

The engine captures one FULL graph per ``(requests, query length)`` of a uniform decode
batch around ``model.forward`` (``cudagraph_utils.ModelCudaGraphManager``), with the
metadata of a dummy batch, and replays it for every real batch of that shape after
rewriting the persistent input buffers (ids, positions, ``query_start_loc``,
``seq_lens``, block tables, slot mappings) in place.  ``Rig`` emulates exactly that
around a ``PagedModel`` world, including the padding of the engine (padded requests
own no token: ``seq_lens = 0``, ``query_start_loc`` repeats the token count; padded
tokens carry ``slot_mapping = -1``), and the tests compare three runs of one schedule
(decode steps of a changing set of requests, one-token and multi-token chunks):

* ``A``  eager, unpadded: what the engine runs without graphs;
* ``B``  eager, padded to the graph shape with the host numbers the graph was captured
  with (``max_seq_len = max_model_len`` and so the split-K factors): the computation
  the graph records;
* ``C``  the graph, replayed.

``B == C`` is the replay gate: the captured launch parameters and pointers are right and
nothing the graph reads is frozen at capture.  ``A ~ B`` bounds the padding and
the capture-time split-K factors.  Both are compared after a run-to-run floor of the
eager path was measured: the expert GEMMs add a token's expert outputs with fp16 CAS
atomics, which commute for two experts per token and not beyond, so the floor is 0 for
the tiny top-2 models and noise for top-8 (``test_run_to_run_difference_by_experts_per_
token`` in ``test_serving_path_gpu.py``).

    HIP_VISIBLE_DEVICES=2 PYTHONPATH=$TREE:$TFM_MAIN pytest \
        tests/models/inkling2_loop/test_serving_path_graph_gpu.py -q -s
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip(
    "transformers.models.inkling2_loop.modeling_inkling2_loop",
    reason="needs a transformers build that ships inkling2_loop",
)
if not torch.cuda.is_available():
    pytest.skip("needs a ROCm GPU", allow_module_level=True)

from paged_harness import cdiv, patched  # noqa: E402
from test_paged_model import PagedModel, make, ref  # noqa: E402

from vllm.models.inkling2_loop.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2_loop.amd import state as state_mod  # noqa: E402

if not hasattr(torch.ops._rocm_C, "inkling2_attention"):
    pytest.skip("this build has no Inkling-2 HIP kernels", allow_module_level=True)

SHAPE = {"head_dim": 128, "d_rel": 16, "num_heads": 6, "num_kv_heads": 2}
MAX_REQS, MAX_TOKENS = 8, 64
KV_BS = 16
LENGTHS = {0: 30, 1: 45, 2: 22, 3: 60, 4: 18}  # tokens of each request's sequence
PREFILL = {0: 20, 1: 30, 2: 12, 3: 40, 4: 10}


def _pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


class Rig:
    """The engine's persistent buffers around one ``PagedModel`` world, with FULL graph
    capture and replay of the whole forward."""

    def __init__(self, pm: PagedModel) -> None:
        assert pm.world is not None
        self.pm, self.world = pm, pm.world
        dev = pm.device
        self.ids = torch.zeros(MAX_TOKENS, dtype=torch.long, device=dev)
        self.pos = torch.zeros(MAX_TOKENS, dtype=torch.long, device=dev)
        self.qsl = torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=dev)
        self.seq_lens = torch.zeros(MAX_REQS, dtype=torch.int32, device=dev)
        self.tables: dict[str, torch.Tensor] = {}
        self.slots: dict[str, torch.Tensor] = {}
        self.scratch: dict[str, state_mod.SeqIdxScratch] = {}
        for g in self.world.groups.values():
            width = cdiv(self.world.max_len, g.bs) + 1
            self.tables[g.name] = torch.zeros(
                MAX_REQS, width, dtype=torch.int32, device=dev
            )
            self.slots[g.name] = torch.full(
                (MAX_TOKENS,), -1, dtype=torch.int64, device=dev
            )
            self.scratch[g.name] = state_mod.SeqIdxScratch(MAX_TOKENS, dev)
        self.hidden = torch.zeros(
            MAX_TOKENS, pm.cfg.hidden_size, dtype=torch.float16, device=dev
        )
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mds: dict[str, state_mod.Inkling2Metadata] = {}

    # -- batches (what the runner computes on the host) -----------------------------
    def real_batch(self, work):
        world = self.world
        starts: list[int] = [0]
        ids: list[int] = []
        pos: list[int] = []
        seq_lens: list[int] = []
        for r, toks in work:
            p0 = world.computed.get(r, 0)
            ids += toks
            pos += range(p0, p0 + len(toks))
            seq_lens.append(p0 + len(toks))
            starts.append(starts[-1] + len(toks))
        tables, slots = {}, {}
        for g in world.groups.values():  # PagedWorld.metadata's block bookkeeping
            width = cdiv(world.max_len, g.bs) + 1
            rows, sl = [], []
            for r, toks in work:
                p0 = world.computed.get(r, 0)
                g.tables.setdefault(r, [])
                g.remove_skipped(r, p0)
                g.allocate(r, p0 + len(toks))
                rows.append(g.row(r, width, p0))
                t = g.tables[r]
                sl += [
                    t[p // g.bs] * g.bs + p % g.bs for p in range(p0, p0 + len(toks))
                ]
            tables[g.name], slots[g.name] = rows, sl
        return SimpleNamespace(
            ids=ids, pos=pos, qsl=starts, seq_lens=seq_lens, tables=tables, slots=slots
        )

    @staticmethod
    def dummy_batch(reqs: int, q: int):
        """The runner's dummy batch of a capture: ``q`` tokens per request, zeroed block
        tables (the null block), every slot ``-1``."""
        n = reqs * q
        return SimpleNamespace(
            ids=[0] * n,
            pos=[0] * n,
            qsl=[i * q for i in range(reqs + 1)],
            seq_lens=[q] * reqs,
            tables={},
            slots={},
        )

    # -- the persistent buffers and the metadata built over them -----------------------
    def load(self, b, reqs: int, tokens: int, max_query_len: int, max_seq_len: int):
        n = len(b.ids)
        self.ids.zero_()
        self.ids[:n] = torch.tensor(b.ids, dtype=torch.long)
        self.pos.zero_()
        self.pos[:n] = torch.tensor(b.pos, dtype=torch.long)
        qsl = b.qsl + [n] * (reqs - (len(b.qsl) - 1))  # padded requests: no tokens
        self.qsl.fill_(n)
        self.qsl[: len(qsl)] = torch.tensor(qsl, dtype=torch.int32)
        self.seq_lens.zero_()  # padded requests: seq_len 0
        self.seq_lens[: len(b.seq_lens)] = torch.tensor(b.seq_lens, dtype=torch.int32)
        self.mds = {}
        for g in self.world.groups.values():
            table, slot = self.tables[g.name], self.slots[g.name]
            table.zero_()
            rows = b.tables.get(g.name, [])
            if rows:
                table[: len(rows)] = torch.tensor(rows, dtype=torch.int32)
            slot.fill_(-1)  # padded tokens: slot -1
            if b.slots.get(g.name):
                slot[:n] = torch.tensor(b.slots[g.name], dtype=torch.int64)
            common = SimpleNamespace(
                num_reqs=reqs,
                num_actual_tokens=tokens,
                query_start_loc=self.qsl,
                query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                seq_lens=self.seq_lens,
                block_table_tensor=table[:reqs],
                slot_mapping=slot[:tokens],
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
            )
            md = state_mod.make_metadata(common, self.scratch[g.name])
            for owner in g.owners:
                self.mds[owner.prefix] = md

    @torch.no_grad()
    def forward(self, tokens: int) -> torch.Tensor:
        with patched(model_mod, "get_metadata", lambda prefix: self.mds[prefix]):
            return self.pm.lm(self.ids[:tokens], self.pos[:tokens])

    @torch.no_grad()
    def capture(self, reqs: int, q: int) -> None:
        tokens = reqs * q
        self.load(
            self.dummy_batch(reqs, q),
            reqs,
            tokens,
            max_query_len=q,
            max_seq_len=self.world.max_len,
        )
        self.hidden[:tokens] = self.forward(tokens)  # warm up: memoised views
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.hidden[:tokens] = self.forward(tokens)
        self.graphs[(reqs, q)] = graph

    # -- one padded step, replayed or eager ------------------------------------------
    @torch.no_grad()
    def padded_step(self, work, mode: str):
        """``mode``: ``graph`` (replay), ``frozen`` (eager with the host numbers the
        graph was captured with: ``max_seq_len = max_model_len``) or ``live`` (eager
        with the numbers of this batch, so only the padding differs from the unpadded
        eager step)."""
        q = len(work[0][1])
        reqs, tokens = _pow2(len(work)), _pow2(len(work)) * q
        replay = mode == "graph"
        if replay and (reqs, q) not in self.graphs:
            self.capture(reqs, q)
        spec = [(r, len(t)) for r, t in work]
        b = self.real_batch(work)
        max_seq_len = max(b.seq_lens) if mode == "live" else self.world.max_len
        self.load(b, reqs, tokens, max_query_len=q, max_seq_len=max_seq_len)
        if replay:
            self.graphs[(reqs, q)].replay()
        else:
            self.hidden[:tokens] = self.forward(tokens)
        n = len(b.ids)
        logits = self.pm.lm.compute_logits(self.hidden[:n]).float().cpu()
        self.world.commit(spec)
        out, at = {}, 0
        for r, t in spec:
            out[r] = logits[at : at + t]
            at += t
        return out


def _sequences(seed: int = 7) -> dict[int, list[int]]:
    gen = torch.Generator().manual_seed(seed)
    return {
        r: torch.randint(0, 500, (n,), generator=gen).tolist()
        for r, n in LENGTHS.items()
    }


def _schedule(seqs):
    """Steps ``[(request, tokens)]``: one prefill step of all requests, then uniform
    decode / verification batches over a changing set of requests."""
    done = dict(PREFILL)
    steps = [[(r, seqs[r][: PREFILL[r]]) for r in PREFILL]]
    plan = [
        ([0, 1, 2, 3, 4], 1),  # 5 requests: padded to 8
        ([0, 1, 2, 3, 4], 1),
        ([0, 1, 2, 3], 1),  # exactly a graph shape
        ([0, 1, 3], 1),  # 3 -> 4
        ([0, 2], 3),  # multi-token chunks (verification), exact
        ([1, 3, 4], 3),  # 3 -> 4
        ([1, 3], 1),
        ([3], 1),  # 1
        ([0, 1, 2, 3, 4], 1),
        ([0, 1, 2, 3, 4], 2),
    ]
    for reqs, q in plan:
        work = []
        for r in reqs:
            work.append((r, seqs[r][done[r] : done[r] + q]))
            done[r] += q
        steps.append(work)
    return steps


def _run(pm: PagedModel, steps, mode: str) -> list[dict[int, torch.Tensor]]:
    """``mode``: ``eager`` (unpadded), ``live`` / ``frozen`` (eager at the padded shape,
    with the host numbers of the batch / of the capture) or ``graph``; every
    non-uniform step (the prefill) is eager in all of them."""
    pm.reset(kv_block_size=KV_BS)
    rig = Rig(pm) if mode != "eager" else None
    outs = []
    for work in steps:
        uniform = len({len(t) for _, t in work}) == 1 and len(work) <= MAX_REQS
        tokens = len(work[0][1]) * _pow2(len(work))
        if mode == "eager" or not uniform or tokens > MAX_TOKENS:
            outs.append(pm.step(work))
        else:
            assert rig is not None
            outs.append(rig.padded_step(work, mode))
    return outs


def _worst(a, b, q: int | None = None) -> float:
    """Largest |difference| of any logit; ``q``: only the steps of ``q`` tokens per
    request."""
    return max(
        float((x[r] - y[r]).abs().max())
        for x, y in zip(a, b)
        for r in x
        if q is None or len(x[r]) == q
    )


@pytest.fixture(scope="module", params=["tiny1", "tiny8"])
def stable(request, tmp_path_factory):
    """Router-stable HIP-shaped model (top-2 experts)."""
    shape = {**SHAPE, "router_bias_scale": 100.0}
    if request.param == "tiny1":
        spec = make.TinySpec(**shape)
    else:
        spec = make.TinySpec(
            num_hidden_layers=8, loop_start=2, loop_end=6, loop_num_passes=3, **shape
        )
    path = tmp_path_factory.mktemp(request.param)
    make.write_checkpoint(path, spec, seed=0)
    return request.param, PagedModel(
        path,
        ref.read_checkpoint(path),
        torch.float16,
        device="cuda",
        experts="kernel",
        path="hip",
    )


def test_graph_replay_equals_the_eager_forward(stable):
    name, pm = stable
    steps = _schedule(_sequences())
    eager, eager_again = _run(pm, steps, "eager"), _run(pm, steps, "eager")
    live, frozen = _run(pm, steps, "live"), _run(pm, steps, "frozen")
    graph = _run(pm, steps, "graph")
    scale = max(float(o[r].abs().max()) for o in eager for r in o)
    floor = _worst(eager, eager_again)
    padding = _worst(eager, live)
    splits = _worst(live, frozen)
    replay = _worst(frozen, graph)
    print(
        f"[graph] {name}: max |logit| {scale:.2f}; max |difference| of logits: "
        f"eager vs eager {floor:.3e} (the run-to-run floor); padded eager vs eager "
        f"{padding:.3e}; "
        f"capture-time max_seq_len vs live {splits:.3e} (decode steps "
        f"{_worst(live, frozen, 1):.3e}, two-token {_worst(live, frozen, 2):.3e}, "
        f"three-token {_worst(live, frozen, 3):.3e}); graph replay vs the same eager "
        f"step {replay:.3e}; {len(steps)} steps"
    )
    assert all(torch.isfinite(o[r]).all() for o in graph for r in o)
    # replay is the captured computation: same kernels, data and host numbers
    assert replay <= floor, (replay, floor)
    # padding (requests and tokens that own nothing) is exact; the split-K factors
    # derived from ``max_model_len`` at capture only move the result by rounding
    assert padding <= floor, (padding, floor)
    assert splits <= max(2 * floor, 1e-2 * scale), (splits, floor)


def test_graph_replay_ignores_what_the_padding_holds(stable):
    """Padded rows read whatever the buffers hold: poison every padded input of a
    replay (ids, positions, the attention output rows nothing writes, the rows of
    padded block tables) and the real rows must not change at all."""
    name, pm = stable
    seqs = _sequences()
    pm.reset(kv_block_size=KV_BS)
    rig = Rig(pm)
    pm.step([(r, seqs[r][: PREFILL[r]]) for r in (0, 1, 2)])
    rig.capture(4, 1)
    real = [(0, seqs[0][20:21]), (1, seqs[1][30:31]), (2, seqs[2][12:13])]

    def run_once(poison: bool) -> torch.Tensor:
        # no ``commit``: both runs are the same step (the block bookkeeping is
        # idempotent and the step rewrites the same K / V / conv rows)
        rig.load(
            rig.real_batch(real), 4, 4, max_query_len=1, max_seq_len=pm.world.max_len
        )
        if poison:
            rig.ids[3:] = 499
            rig.pos[3:] = 100
            for buf in pm.lm.model.glue_ws._attn_out.values():
                buf[3:4] = float("nan")  # the row of the padded token
            for table in rig.tables.values():
                table[3:] = (
                    1  # the padded request's row: a live block, not the null one
                )
        rig.graphs[(4, 1)].replay()
        return rig.hidden[:3].clone()

    clean, dirty = run_once(False), run_once(True)
    assert torch.isfinite(clean).all() and torch.equal(clean, dirty), name


def _cache_bits(pm: PagedModel) -> list[torch.Tensor]:
    """Every cache page of the world as raw 16-bit patterns (the NaN poison included, so
    a row nothing may write has to keep the poison)."""
    return [
        owner.kv_cache.view(torch.int16).clone()
        for g in pm.world.groups.values()
        for owner in g.owners
    ]


def test_padding_writes_no_cache_row(stable):
    """The caches after the padded steps are the caches after the unpadded ones, bit for
    bit: the K / V pages (padded tokens carry ``slot_mapping = -1``) and the conv rings.
    The ring is addressed by position and a padded token that read as request 0 would
    write the position past the end of request 0's chunk into a ring row, the history
    tap of the token that follows."""
    name, pm = stable
    # the prefill, 5 -> 8 requests twice, exactly 4, then 3 -> 4
    steps = _schedule(_sequences())[:5]
    _run(pm, steps, "eager")
    want = _cache_bits(pm)
    _run(pm, steps, "live")
    got = _cache_bits(pm)
    assert len(got) == len(want) > 0
    for i, (a, b) in enumerate(zip(want, got)):
        assert torch.equal(a, b), f"{name}: cache {i} differs after the padded steps"


def test_graph_replay_under_the_cas_noise_of_eight_experts(tmp_path_factory):
    """With eight experts per token the expert GEMMs add into the output with fp16 CAS
    atomics in an order that varies from run to run (see
    ``test_run_to_run_difference_by_experts_per_token``): the noise floor is the
    difference of eager runs of the schedule, measured first; graph replay is one more
    run of the same computation, so it differs from an eager run by that noise (and the
    capture-time split factors), never by an O(1) amount, and stays finite."""
    spec = make.TinySpec(
        num_hidden_layers=8,
        loop_start=2,
        loop_end=6,
        loop_num_passes=3,
        n_routed_experts=16,
        num_experts_per_tok=8,
        router_bias_scale=1e4,  # the expert choice is the bias alone: nothing flips it
        **SHAPE,
    )
    path = tmp_path_factory.mktemp("graph-top8")
    make.write_checkpoint(path, spec, seed=0)
    pm = PagedModel(
        path,
        ref.read_checkpoint(path),
        torch.float16,
        device="cuda",
        experts="kernel",
        path="hip",
    )
    steps = _schedule(_sequences())
    eager = [_run(pm, steps, "eager") for _ in range(3)]
    graph = [_run(pm, steps, "graph") for _ in range(2)]
    scale = max(float(o[r].abs().max()) for o in eager[0] for r in o)
    floor = max(_worst(eager[0], e) for e in eager[1:])
    vs_eager = max(_worst(eager[0], g) for g in graph)
    vs_graph = _worst(graph[0], graph[1])
    print(
        f"[graph] top-8: max |logit| {scale:.2f}; eager vs eager {floor:.3e} (the "
        f"noise floor); graph vs eager {vs_eager:.3e}; graph vs graph {vs_graph:.3e}"
    )
    assert all(torch.isfinite(o[r]).all() for g in graph for o in g for r in o)
    assert max(floor, vs_eager, vs_graph) < 0.05 * scale  # a race would show O(1)
