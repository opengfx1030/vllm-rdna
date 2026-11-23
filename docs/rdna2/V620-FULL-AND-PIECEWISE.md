# FULL and piecewise CUDA graphs

`FULL_AND_PIECEWISE` captures both graphs and replays each as itself:

- Uniform decode replays the FULL CUDA graph (the same live capture as
  mode-0 `FULL_DECODE_ONLY`, over the persistent batch buffers).
- Mixed and prefill batches replay the piecewise graphs.

`rdna_extra/v0.29.0` (a historical port branch) and the piecewise follow-up
([#20](https://github.com/opengfx1030/vllm-rdna/pull/20)) replayed every ROCm
FULL dispatch as piecewise. Decode then ran with eager GDN and attention
breaks between pieces (about 63–70 tok/s down to 26–34 tok/s on the V620
16k harness). That redirect is off. Piecewise capture still needs the
stride fixes from #20: a contiguous hyperconnection injection and contiguous
M-RoPE / XD-RoPE positions, or Inductor aborts capture.

Without compilation and without `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, a
`FULL_AND_PIECEWISE` request keeps `FULL_DECODE_ONLY` instead of dropping
every graph. A bare `PIECEWISE` request still becomes `NONE`.

The MTP draft manager captures its own FULL graphs even when the target
config also has piecewise graphs.

Models that are not compiled can still keep `FULL_AND_PIECEWISE` on RDNA: an
architecture listed in `default_breakable_cudagraph_architectures`
(`vllm/config/vllm.py`) records its eager forward as breakable CUDA graphs, with
the layers' custom ops as the eager breaks (Inkling-2: one `attend_hip` break per
layer visit, 49 segments). Uniform decode and speculative-verify batches replay the
FULL graph, prefill and mixed batches the breakable graphs, padded up to the
captured token count. See "integration part B" in `inkling2-port-spec.md` for the
capture contract and the measured cost: the padding is the whole cost of the
breakable graphs (a captured size is as fast as eager on the same real tokens), and
padded decode rows are real MoE work, so such a model's default decode grid is dense:
`dense_decode_cudagraph_architectures` (`vllm/config/vllm.py`, Inkling-2 on RDNA) makes
`decode_cudagraph_capture_request_sizes` every request count up to 16, then the usual
request-count grid, for plain decode and for every speculative-verification query length;
an explicit list still wins. 5 rows replayed in the 8-row graph measured 61.0 ms against
45.3 ms eager and 43.8 ms in an exact-size graph.
