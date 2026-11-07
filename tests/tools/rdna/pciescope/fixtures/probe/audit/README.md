# Probe audit fixtures

Real output of the `p2p_probe` batch mode, captured on par1-cs25 during the
PCIe audit of 2026-11-06 and 2026-11-07. They are the input of
`test_pciescope_probe_parse.py`. Nothing in these files was edited or
invented: the legacy binary has no `READY`, `END`, `VERSION`, `HBW`, `HAGG`,
`HLAT`, `HFLAG` or `CURVE` lines, so those tags are covered by hand-written
format examples inside the test file, which are not measurements.

All source paths are relative to `~/w4a8_runs/pcie_audit/p2p/` on the box
(`chenco_adm@par1-cs25`). The log header (`# hip=... buses={...}`) is written by
`run_matrix.py` between the `P2P` lines and the first `###` command.

| Fixture | Source | Original bytes | Fixture bytes | Fixture lines |
| --- | --- | --- | --- | --- |
| `s1_base.probe.log` | `res/s1_base.probe.log` | 14685 | 14685 | 320 |
| `w7_win1.head220.log` | `w7_win1.log` | 46732 | 10122 | 220 |
| `w7_win3.head220.log` | `w7_win3.log` | 114594 | 9187 | 220 |
| `w8_win4.head220.log` | `w8_win4.log` | 93743 | 9843 | 220 |
| `w8_win5.head220.log` | `w8_win5.log` | 43498 | 9634 | 220 |

## Fetch commands

`s1_base.probe.log` is the whole file (8 V620 GPUs, sub suite of `bw`,
`latc` and `a2a`; 66 commands, 8 `DEV` lines, 56 `P2P` lines, no `END`
lines):

```bash
scp -i ~/.ssh/id_ed25519_ansible \
  chenco_adm@par1-cs25:w4a8_runs/pcie_audit/p2p/res/s1_base.probe.log .
```

The four `w*` fixtures are the first 220 lines of the original (a cut that may
end inside a command on purpose, to exercise truncated blocks; `lat` all pairs,
`bw` push, pull and concurrent runs):

```bash
for n in w7_win1 w7_win3 w8_win4 w8_win5; do
  ssh -i ~/.ssh/id_ed25519_ansible chenco_adm@par1-cs25 \
    "head -n 220 ~/w4a8_runs/pcie_audit/p2p/$n.log" > "$n.head220.log"
done
```

## Checksums (SHA-256)

```text
0e1729bb4838ddfee705a27d15c918fdf95e553177bf3c02317dcf157282be33  s1_base.probe.log
f42849d031feba1e4d070bb07e570625207a9eefa8c88cc2df4b54b30aee7cdc  w7_win1.head220.log
b9778b4d55136863186c1ff399b2f0d40d0db80677fb7b9bae88098e8566ecc8  w7_win3.head220.log
7d7628dc9b3bcbfba69d5c6783e98a0843873675b3b97ed8f91501ff4b4f0e2a  w8_win4.head220.log
95b307e9638aa098d4ca704e2eb89bf7e3580313df7e5ca0e1b0ff957934aa29  w8_win5.head220.log
```

The first two were also computed on the box and match (`s1_base` whole file,
`w7_win1` the `head -n 220` output).

## What the cuts end with

* `w7_win1.head220.log`: 7 devices; the last command is a `bw` with 7 transfers
  and only 6 `X` rows are present.
* `w7_win3.head220.log`: 7 devices; the last command has one `X` row and no `AGG`.
* `w8_win4.head220.log`: 8 devices; the last command has 4 of 6 `X` rows.
* `w8_win5.head220.log`: 8 devices; the last command is whole (`X` and `AGG`)
  but has no `END`, so the parser cannot know it is finished.
