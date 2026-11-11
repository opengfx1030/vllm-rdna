# Task suite, hi2 bench/compare.py

## B vs A

Comparison status: **complete**. Scores below use completed pairs only.
This is a quality comparison, not a native-kernel performance comparison.

| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % | recovery | n for 1 pt |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 120 | 96.7 | 95.8 | -0.8 [-2.5, +0.0] | 1 | 0.8 | 0.0 | 0.991 | 783 |
| humaneval | 120 | 90.8 | 92.5 | +1.7 [-3.3, +6.7] | 0.754 | 3.3 | 5.0 | 1.018 | 6539 |
| mbpp | 120 | 93.3 | 94.2 | +0.8 [+0.0, +2.5] | 1 | 0.0 | 0.8 | 1.009 | 783 |
| ifeval | 120 | 96.7 | 92.5 | -4.2 [-8.3, +0.0] | 0.125 | 5.0 | 0.8 | 0.957 | 4577 |
| toolcall | 120 | 65.8 | 65.0 | -0.8 [-4.2, +2.5] | 1 | 2.5 | 1.7 | 0.987 | 3269 |
| needle | 120 | 99.2 | 100.0 | +0.8 [+0.0, +2.5] | 1 | 0.0 | 0.8 | 1.008 | 783 |
| decision | 66 | 75.8 | 75.8 | +0.0 [-9.1, +9.1] | 1 | 6.1 | 6.1 | 1.000 | 9512 |
| **all (pooled)** | 786 | 88.3 | 88.0 | -0.4 [-1.8, +1.0] | 0.728 | | | 0.996 | |

ref/cand % in the last row are macro means over tasks; diff and p are pooled over all items. R->W: reference right, candidate wrong.
n for 1 pt: paired items for 80% power to detect a 1-point difference (alpha 0.05, McNemar) at this pair's discordance rate (- when no item flipped); about 78500 x the discordance.

| task | comparison | reference completed / expected | reference status | candidate completed / expected | candidate status | unpaired observed items |
|---|---|---|---|---|---|---|
| gsm8k | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| humaneval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| mbpp | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| ifeval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| toolcall | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| needle | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| decision | complete | 66 / 66 | complete | 66 / 66 | complete | 0 |

| task | measure | ref | cand |
|---|---|---|---|
| ifeval | mean frac | 0.9750 | 0.9542 |
| toolcall | mean arg_f1 | 0.8358 | 0.8108 |
| decision | agree on tool | 100 | 80.3 [69.2, 88.1] |
| decision | agree on tool+arg | 100 | 74.2 [62.6, 83.3] |

| task | group | n | ref % | cand % | diff |
|---|---|---|---|---|---|
| ifeval | avoid_word | 8 | 75.0 | 75.0 | +0.0 |
| ifeval | bullets | 7 | 85.7 | 85.7 | +0.0 |
| ifeval | end_phrase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | highlights | 7 | 100.0 | 85.7 | -14.3 |
| ifeval | json | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | keywords | 7 | 100.0 | 71.4 | -28.6 |
| ifeval | lowercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | no_comma | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | paragraphs | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | placeholders | 7 | 100.0 | 85.7 | -14.3 |
| ifeval | postscript | 7 | 85.7 | 71.4 | -14.3 |
| ifeval | quotes | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | sections | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | title | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | uppercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_max | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_min | 7 | 100.0 | 100.0 | +0.0 |
| toolcall | multi | 62 | 58.1 | 56.5 | -1.6 |
| toolcall | single | 58 | 74.1 | 74.1 | +0.0 |
| needle | 16k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 16k-single | 15 | 93.3 | 100.0 | +6.7 |
| needle | 2k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 2k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-single | 15 | 100.0 | 100.0 | +0.0 |
| decision | fresh | 12 | 50.0 | 16.7 | -33.3 |
| decision | listing | 12 | 91.7 | 100.0 | +8.3 |
| decision | read | 12 | 100.0 | 100.0 | +0.0 |
| decision | test_fail | 12 | 83.3 | 91.7 | +8.3 |
| decision | test_pass | 12 | 66.7 | 75.0 | +8.3 |
| decision | vague | 6 | 50.0 | 66.7 | +16.7 |

## A2 vs A

Comparison status: **complete**. Scores below use completed pairs only.
This is a quality comparison, not a native-kernel performance comparison.

| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % | recovery | n for 1 pt |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 120 | 96.7 | 96.7 | +0.0 [+0.0, +0.0] | 1 | 0.0 | 0.0 | 1.000 | - |
| humaneval | 120 | 90.8 | 89.2 | -1.7 [-6.7, +2.5] | 0.727 | 4.2 | 2.5 | 0.982 | 5231 |
| mbpp | 120 | 93.3 | 94.2 | +0.8 [+0.0, +2.5] | 1 | 0.0 | 0.8 | 1.009 | 783 |
| ifeval | 120 | 96.7 | 96.7 | +0.0 [-3.3, +3.3] | 1 | 1.7 | 1.7 | 1.000 | 2614 |
| toolcall | 120 | 65.8 | 69.2 | +3.3 [+0.0, +7.5] | 0.219 | 0.8 | 4.2 | 1.051 | 3923 |
| needle | 120 | 99.2 | 100.0 | +0.8 [+0.0, +2.5] | 1 | 0.0 | 0.8 | 1.008 | 783 |
| decision | 66 | 75.8 | 74.2 | -1.5 [-7.6, +4.5] | 1 | 4.5 | 3.0 | 0.980 | 5944 |
| **all (pooled)** | 786 | 88.3 | 88.6 | +0.4 [-0.9, +1.7] | 0.69 | | | 1.004 | |

ref/cand % in the last row are macro means over tasks; diff and p are pooled over all items. R->W: reference right, candidate wrong.
n for 1 pt: paired items for 80% power to detect a 1-point difference (alpha 0.05, McNemar) at this pair's discordance rate (- when no item flipped); about 78500 x the discordance.

| task | comparison | reference completed / expected | reference status | candidate completed / expected | candidate status | unpaired observed items |
|---|---|---|---|---|---|---|
| gsm8k | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| humaneval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| mbpp | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| ifeval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| toolcall | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| needle | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| decision | complete | 66 / 66 | complete | 66 / 66 | complete | 0 |

| task | measure | ref | cand |
|---|---|---|---|
| ifeval | mean frac | 0.9750 | 0.9792 |
| toolcall | mean arg_f1 | 0.8358 | 0.8532 |
| decision | agree on tool | 100 | 87.9 [77.9, 93.7] |
| decision | agree on tool+arg | 100 | 83.3 [72.6, 90.4] |

| task | group | n | ref % | cand % | diff |
|---|---|---|---|---|---|
| ifeval | avoid_word | 8 | 75.0 | 87.5 | +12.5 |
| ifeval | bullets | 7 | 85.7 | 85.7 | +0.0 |
| ifeval | end_phrase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | highlights | 7 | 100.0 | 85.7 | -14.3 |
| ifeval | json | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | keywords | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | lowercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | no_comma | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | paragraphs | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | placeholders | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | postscript | 7 | 85.7 | 85.7 | +0.0 |
| ifeval | quotes | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | sections | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | title | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | uppercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_max | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_min | 7 | 100.0 | 100.0 | +0.0 |
| toolcall | multi | 62 | 58.1 | 64.5 | +6.5 |
| toolcall | single | 58 | 74.1 | 74.1 | +0.0 |
| needle | 16k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 16k-single | 15 | 93.3 | 100.0 | +6.7 |
| needle | 2k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 2k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-single | 15 | 100.0 | 100.0 | +0.0 |
| decision | fresh | 12 | 50.0 | 50.0 | +0.0 |
| decision | listing | 12 | 91.7 | 91.7 | +0.0 |
| decision | read | 12 | 100.0 | 100.0 | +0.0 |
| decision | test_fail | 12 | 83.3 | 75.0 | -8.3 |
| decision | test_pass | 12 | 66.7 | 75.0 | +8.3 |
| decision | vague | 6 | 50.0 | 33.3 | -16.7 |

## B2 vs B

Comparison status: **complete**. Scores below use completed pairs only.
This is a quality comparison, not a native-kernel performance comparison.

| task | n | ref % | cand % | diff [95% CI] | McNemar p | R->W % | W->R % | recovery | n for 1 pt |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 120 | 95.8 | 96.7 | +0.8 [+0.0, +2.5] | 1 | 0.0 | 0.8 | 1.009 | 783 |
| humaneval | 120 | 92.5 | 91.7 | -0.8 [-5.0, +3.3] | 1 | 3.3 | 2.5 | 0.991 | 4577 |
| mbpp | 120 | 94.2 | 92.5 | -1.7 [-4.2, +0.0] | 0.5 | 1.7 | 0.0 | 0.982 | 1306 |
| ifeval | 120 | 92.5 | 95.8 | +3.3 [+0.0, +7.5] | 0.219 | 0.8 | 4.2 | 1.036 | 3923 |
| toolcall | 120 | 65.0 | 66.7 | +1.7 [-2.5, +5.8] | 0.688 | 1.7 | 3.3 | 1.026 | 3923 |
| needle | 120 | 100.0 | 100.0 | +0.0 [+0.0, +0.0] | 1 | 0.0 | 0.0 | 1.000 | - |
| decision | 66 | 75.8 | 77.3 | +1.5 [-6.1, +9.1] | 1 | 4.5 | 6.1 | 1.020 | 8323 |
| **all (pooled)** | 786 | 88.0 | 88.7 | +0.6 [-0.6, +2.0] | 0.458 | | | 1.009 | |

ref/cand % in the last row are macro means over tasks; diff and p are pooled over all items. R->W: reference right, candidate wrong.
n for 1 pt: paired items for 80% power to detect a 1-point difference (alpha 0.05, McNemar) at this pair's discordance rate (- when no item flipped); about 78500 x the discordance.

| task | comparison | reference completed / expected | reference status | candidate completed / expected | candidate status | unpaired observed items |
|---|---|---|---|---|---|---|
| gsm8k | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| humaneval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| mbpp | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| ifeval | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| toolcall | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| needle | complete | 120 / 120 | complete | 120 / 120 | complete | 0 |
| decision | complete | 66 / 66 | complete | 66 / 66 | complete | 0 |

| task | measure | ref | cand |
|---|---|---|---|
| ifeval | mean frac | 0.9542 | 0.9750 |
| toolcall | mean arg_f1 | 0.8108 | 0.8149 |
| decision | agree on tool | 100 | 80.3 [69.2, 88.1] |
| decision | agree on tool+arg | 100 | 66.7 [54.7, 76.8] |

| task | group | n | ref % | cand % | diff |
|---|---|---|---|---|---|
| ifeval | avoid_word | 8 | 75.0 | 87.5 | +12.5 |
| ifeval | bullets | 7 | 85.7 | 85.7 | +0.0 |
| ifeval | end_phrase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | highlights | 7 | 85.7 | 85.7 | +0.0 |
| ifeval | json | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | keywords | 7 | 71.4 | 85.7 | +14.3 |
| ifeval | lowercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | no_comma | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | paragraphs | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | placeholders | 7 | 85.7 | 100.0 | +14.3 |
| ifeval | postscript | 7 | 71.4 | 100.0 | +28.6 |
| ifeval | quotes | 7 | 100.0 | 85.7 | -14.3 |
| ifeval | sections | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | title | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | uppercase | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_max | 7 | 100.0 | 100.0 | +0.0 |
| ifeval | word_min | 7 | 100.0 | 100.0 | +0.0 |
| toolcall | multi | 62 | 56.5 | 59.7 | +3.2 |
| toolcall | single | 58 | 74.1 | 74.1 | +0.0 |
| needle | 16k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 16k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 2k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 2k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 32k-single | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-multikey | 15 | 100.0 | 100.0 | +0.0 |
| needle | 8k-single | 15 | 100.0 | 100.0 | +0.0 |
| decision | fresh | 12 | 16.7 | 50.0 | +33.3 |
| decision | listing | 12 | 100.0 | 100.0 | +0.0 |
| decision | read | 12 | 100.0 | 100.0 | +0.0 |
| decision | test_fail | 12 | 91.7 | 83.3 | -8.3 |
| decision | test_pass | 12 | 75.0 | 66.7 | -8.3 |
| decision | vague | 6 | 66.7 | 50.0 | -16.7 |
