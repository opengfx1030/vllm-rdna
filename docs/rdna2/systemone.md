# System One decision route (GLiNER2.5)

Opt-in serve-side route for typed decisions. It is **off** unless
`--systemone-backend` (or `VLLM_SYSTEMONE_BACKEND`) is set. Nothing in a
request body turns it on, and the decision model is not listed on
`GET /v1/models`.

GLiNER2.5-multi-Decide is a 287M mDeBERTa encoder (`gliner2`), not a vLLM
model class. `vllm serve` cannot load it as the main model. A second
`vllm serve` process cannot load it either. Mode B below is a small
standalone process that speaks the same `/v1/systemone` contract.

The decision model runs in the API-server process (or in that standalone
process). It does not enter EngineCore, the scheduler, the decode loop, the
KV cache, CUDA/HIP graph capture, or the main model's memory profile.

## Contract

`POST /v1/systemone` follows the System One native contract
(`typesafe-2026-09-18`):

- `state`: string, object, array, or null (required).
- `questions`: 1–32 named questions.
- `type`: `choice`, `score`, or `noul` (yes/no probability). The wire type
  is not `boolean`.
- Choice `criteria` is an object of option name to description (1–255).
- Score `criteria` is an ordered array of 2–10 level descriptions. The
  answer `score` is the probability-weighted index, and `legend` maps
  `"0"`… to those descriptions.
- Noul `criteria` may be omitted, null, or `{true, false}` descriptions.
  The answer is `noul`: P(yes), with no separate confidence.
- Choice and score answers include `probabilities` and `confidence`.
  Confidence is the gap between the top two probabilities.

Optional question fields `threshold` and `cls_threshold` (0–1, exclusive)
are forwarded to GLiNER as the task threshold. `multi_label: true` is
rejected: System One answers are single-valued. Malformed questions return
400. The body limit is 64 KiB.

Clients use `SYSTEM_ONE_BASE_URL`, `SYSTEM_ONE_MODEL`, and
`SYSTEM_ONE_PATH=/v1/systemone`. Those client variables do not configure
this server. Server settings use `VLLM_SYSTEMONE_*` or `--systemone-*`.

## Mode A: same instance

Preferred when the API server can host the encoder. The model is loaded
once at startup, on a worker thread, behind an asyncio micro-batcher
(`--systemone-max-batch`, `--systemone-max-wait-ms`,
`--systemone-max-queue`, `--systemone-timeout-s`). Decision calls and
`/v1/chat/completions` run concurrently. A full queue returns 429. A
timed-out request returns 504.

```bash
uv pip install gliner2   # optional; vLLM imports without it

vllm serve <main-model> \
  --tensor-parallel-size 4 \
  --systemone-backend gliner2 \
  --systemone-model fastino/GLiNER2.5-multi-Decide \
  --systemone-device cpu
```

CPU is the default on purpose. The encoder is about 594 MB in fp16 and
does not need a GPU. A `cuda:N` device is opt-in and must be a **spare**
device: not one of the logical GPUs occupied by
`data_parallel_size * tensor_parallel_size * pipeline_parallel_size`,
after `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` remapping, and not
one of `--device-ids`. Engine ranks own uncached P2P, FA/QSA graph
capture, and the persistent heaps. Startup fails closed if `cuda:N` maps
onto one of them.

On a 4-GPU box with TP=4 the mask is exactly the engine, so there is no
spare device. Use CPU in Mode A, or Mode B on another host. Lowering
`--gpu-memory-utilization` does not create a spare device: that fraction
is applied on every TP rank.

A spare GPU (a 5th card listed after the engine ranks) also has to show
at least `--systemone-vram-reserve-gb` free (default 2) or startup fails.
The decision model uses float16 on CUDA and float32 on CPU. bfloat16 is
rejected. CUDA work runs on a dedicated torch stream created for this
model, never the default stream, and is refused if graph capture is
active.

```bash
# 5 visible GPUs, engine uses the first 4, decision model uses the last.
HIP_VISIBLE_DEVICES=0,1,2,3,4 \
vllm serve <main-model> \
  --tensor-parallel-size 4 \
  --systemone-backend gliner2 \
  --systemone-model fastino/GLiNER2.5-multi-Decide \
  --systemone-device cuda:4 \
  --systemone-vram-reserve-gb 2
```

English-only text can use `fastino/GLiNER2.5-Decide` the same way. The
loader is `AutoExtractor`; probabilities come from
`Classifier.batch_classify`, because `classify_text` does not return a
full distribution.

## Mode B: separate process

Use this when the encoder cannot sit beside the engine (the TP=4 case
above, or a host with no room for another CUDA context).

```bash
python -m vllm.entrypoints.systemone.server \
  --model fastino/GLiNER2.5-multi-Decide \
  --device cuda:0 \
  --port 8091
```

This process has no engine ranks, so `cuda:0` is its own GPU. The VRAM
reserve still applies. Point the main server at it:

```bash
vllm serve <main-model> \
  --systemone-backend http \
  --systemone-url http://127.0.0.1:8091/v1/systemone \
  --systemone-model fastino/GLiNER2.5-multi-Decide
```

Clients keep calling `/v1/systemone` on the main server. A bare origin in
`--systemone-url` is treated as that path.

## Example

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fastino/GLiNER2.5-multi-Decide",
    "state": "I was charged twice for one order.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Choose the reviewing team.",
        "criteria": {
          "billing": "Payments and refunds",
          "support": "Technical help"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "Evaluate urgency with the ordered rubric.",
        "criteria": [
          "Normal review",
          "Timely response",
          "Immediate human attention"
        ]
      },
      "duplicate": {
        "type": "noul",
        "instructions": "Does the message report a duplicate charge?"
      }
    }
  }'
```

```json
{
  "model": "fastino/GLiNER2.5-multi-Decide",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.8, "support": 0.2},
      "confidence": 0.6
    },
    "urgency": {
      "type": "score",
      "score": 1.5,
      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
      "confidence": 0.25,
      "legend": {
        "0": "Normal review",
        "1": "Timely response",
        "2": "Immediate human attention"
      }
    },
    "duplicate": {"type": "noul", "noul": 0.8}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

The numbers above illustrate the contract. The `stub` backend uses fixed
weights in label order: `(0.2, 0.8)` for two labels and `(0.1, 0.3, 0.6)`
for three, so this example's stub choice is `support` and its noul is
`0.2`. Choice confidence for two options is the top-two margin (`0.6`).
Score confidence for `(0.1, 0.3, 0.6)` is `0.25` under the TypeSafe spread
formula shared with llama.cpp and SGLang. A live GLiNER2 call returns its own
probabilities. `usage` is always present; this encoder reports zeros because
it does not count tokens.

## Upstream wire

The request and response follow the fields that
[llama.cpp#29818](https://github.com/ggml-org/llama.cpp/pull/29818),
[SGLang#42183](https://github.com/sgl-project/sglang/pull/42183), and
[vLLM#59299](https://github.com/vllm-project/vllm/pull/59299) share:
`state`, `questions` of `choice` / `score` / `noul`, and answers with
`probabilities` and `confidence` (noul is only `noul`, P(yes)). Invalid
requests are HTTP 400 with an `error` object (`message`, `type`, `param`,
`code`). Image input on this encoder is HTTP 501, the same code llama.cpp
uses when the model cannot take images. `temperature`,
`prompt_format_version`, and `return_prompt_token_ids` are rejected with
400, as SGLang does, because ignoring them would change the answer.

Where those implementations disagree, this server follows the column below.

| field | llama.cpp | SGLang | vLLM #59299 | this server |
| --- | --- | --- | --- | --- |
| question types | choice, score, noul | choice, score, noul | choice only | choice, score, noul |
| choice / score confidence | TypeSafe formulas | same formulas | `p_top * label_mass` | TypeSafe formulas |
| noul confidence | absent | `x_label_mass` extension | no noul type | absent |
| `usage` | real `input_tokens`, `output_tokens` 0 | real `input_tokens` | real input and output | present, zeros on the encoder |
| `id`, `object`, `created`, `diagnostics` | absent | absent | present | absent |
| score levels | 2 to 10 | 1 to 10 | n/a | 2 to 10 |
| questions per request | model limit | no fixed 32 | 64 | 32 |
| `images` | 501 if unsupported | supported | absent | 501 |
| `seed`, `chat_template_kwargs` | absent | kwargs accepted | both accepted | ignored (no label shuffle) |

#59299 registers `POST /v1/systemone` from
`vllm.entrypoints.generate.structured_decisions` behind
`--enable-structured-decisions`. This package does not occupy that path.
If that route is already on the app, this server does not register a second
one. The loaded encoder is published as `systemone_backend_provider` and
answers only when the request `model` is the configured decision model.
Other models stay on the upstream handler. A rebase that contains #59299
deletes `vllm/entrypoints/systemone` instead of merging two protocol trees.

## LiteLLM

LiteLLM only proxies OpenAI routes it knows. `/v1/systemone` is not one of
them, so a pass-through entry has to point at whichever server is live:
the vLLM API server in Mode A, or the standalone server in Mode B. Clients
then keep a single path.

```yaml
general_settings:
  pass_through_endpoints:
    - path: "/v1/systemone"
      # Mode A (vLLM API server). For Mode B, use the standalone origin.
      target: "http://127.0.0.1:8000"
      headers: {}
```

A request to the proxy at `POST /v1/systemone` is forwarded to
`target` + `/v1/systemone`. Switching Mode A to Mode B is a change of
`target`, not of the client path.

## Not implemented

Platform concerns that belong to a hosted System One gateway are not
implemented: idempotency cache, credit and rate-limit headers, `GET /v1/models`
catalog changes, and `/api/system-one/*`. `classify` and `route` are not
question types in this HTTP contract. Extension fields other than
`threshold` / `cls_threshold` are ignored rather than echoed byte-for-byte.
