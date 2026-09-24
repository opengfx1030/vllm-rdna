# FULL and piecewise CUDA graphs

`FULL_AND_PIECEWISE` captures both graphs and replays each as itself:

- Uniform decode replays the FULL CUDA graph (the same live capture as
  mode-0 `FULL_DECODE_ONLY`, over the persistent batch buffers).
- Mixed and prefill batches replay the piecewise graphs.

`rdna_extra/v0.29.0` and the piecewise follow-up
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
