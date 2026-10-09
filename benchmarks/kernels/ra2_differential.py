import random
import sys

sys.path.insert(0, "/Users/kletorch/Projects/infrastructure/gfx1030_optimized/vllm-rdna-0.28.0/benchmarks/kernels")
from ra2_protocol_model import model

random.seed(7)
fails = 0
cases = 0
for W in (2, 4, 8):
    for CH in range(1, 17):
        if CH % W or CH < W:
            continue
        for N in (W, 2 * W, 4 * W * CH):
            ce = N // CH
            if ce < 1:
                continue
            data = [[random.randrange(-50, 50) / 4.0 for _ in range(N)] for _ in range(W)]
            expect = [sum(data[r][i] for r in range(W)) for i in range(N)]
            out, f = model(W, CH, data)
            cases += 1
            if out is None:
                print(f"W={W} CH={CH} N={N}: FAIL {f}")
                fails += 1
                continue
            for r in range(W):
                if max(abs(out[r][i] - expect[i]) for i in range(N)) > 1e-6:
                    print(f"W={W} CH={CH} N={N} rank{r}: MISMATCH")
                    fails += 1
                    break

print(f"\nrandomized differential: {cases - fails}/{cases} cases pass")
print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
