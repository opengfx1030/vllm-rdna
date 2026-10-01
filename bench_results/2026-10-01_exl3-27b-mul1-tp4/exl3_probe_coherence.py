import json, sys, urllib.request, time

BASE = "http://127.0.0.1:18105/v1/chat/completions"
MODEL = "exl3-27b-mul1"

def chat(prompt, max_tokens=64, temperature=0.0, dump=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode())
    dt = time.time() - t0
    choice = data["choices"][0]
    text = choice.get("message", {}).get("content", "")
    if dump:
        print("  RAW keys:", list(data.keys()), "| choice keys:", list(choice.keys()))
        print("  RAW content:", repr(text))
        print("  RAW message:", json.dumps(choice.get("message", {}), indent=2)[:1200])
    return text, dt

probes = [
    ("The capital of France is", "Paris"),
    ("2 + 2 =", "4"),
    ("1+1=", "2"),
    ("Once upon a time", None),
    ("Write one sentence about the moon.", None),
]

ok = True
for prompt, expect in probes:
    try:
        text, dt = chat(prompt, dump=(prompt == "The capital of France is"))
        print(f"PROMPT: {prompt!r}")
        print(f"  OUTPUT: {text!r}  ({dt:.2f}s)")
        if expect and expect.lower() not in text.lower():
            print(f"  [FAIL] expected substring {expect!r} not found")
            ok = False
        else:
            print("  [OK]")
    except Exception as e:
        print(f"PROMPT: {prompt!r}")
        print(f"  [EXC] {type(e).__name__}: {e}")
        ok = False
    print("---")

print("COHERENCE_RESULT:", "PASS" if ok else "FAIL")
