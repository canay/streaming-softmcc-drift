# Post-processing reanalysis, 2026-09-20

`reanalysis.py` is the executable path from the frozen raw experiment outputs to
three quantities the manuscript reports. It exists because the frozen runs under
`experiments/` were executed with earlier analysis choices and their own entry
points still summarise the data that way, so running those entry points
reproduces the earlier numbers rather than the reported ones.

The script **re-runs no experiment** and **modifies no frozen artifact**. It
reads only the files listed in `results/MANIFEST.json` and writes only into
`results/`.

## What it recomputes, and against what

| Quantity | Reported form | Archived form the run scripts implement |
|---|---|---|
| Ordered-stream direction count | paired over anchors where both kernels are defined: **9 window / 24 fading** | each kernel averaged over its own valid anchors: 15 / 18 |
| log\|bias\| ~ log *w* slope interval | one unit index set per draw, no sign filter: **[−1.76, −0.52]** window, **[−1.73, −0.53]** fading | a separate index set per memory setting, draws not sharing a sign discarded: [−1.71, −0.54] and [−1.68, −0.54] |
| Table 7 stationary mean deviations | regenerated from the raw unit values | transcribed |

Both forms are emitted for the first two, so the difference is auditable rather
than asserted. 8 of the 33 ordered-stream contrasts reverse direction between
the two forms; every one of the 33 remains eligible on the intersection under
the same two-anchor and half-anchor rule.

## Controls

- The ordered-stream analysis reproduces all 33 published per-kernel means to
  better than 1e-9 **before** computing anything new. If that control fails the
  script stops.
- The slope analysis reports how many draws each form keeps (2000 of 2000 for
  the reported form; 840 and 851 for the archived form), so the effect of the
  sign filter is visible rather than implicit.
- Every cell of Table 7 is regenerated, not only the one that was found wrong,
  because the same transcription path produced all of them.

## Running it

```
python reanalysis.py            # writes results/ and results/MANIFEST.json
```

`MANIFEST.json` records the script hash, the SHA-256 of every input and output,
the RNG seed and draw count, and the interpreter and NumPy versions.
