# gta6-trophy-rarity-forecast-01

Forecast, for every GTA VI trophy on PlayStation 5, its earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy, scored by mean pinball loss on
the logit scale.

## Program

`main.ts` (Node 22, standard library only) reads every `*.in` case in `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes `/output/<id>.out` for each. It is deterministic:
two runs on the same case write byte-identical output, and it runs in well under
the 8-second cap.

## Model — decomposed log-time earners/players

A trophy's rate is a ratio, `rate = earners / players_tracked`, and the two sides
move for different reasons, so they are forecast separately and divided again at
the target:

- **Players tracked P(t):** `log P` linear in `log(1 + t)`, `t` in hours since
  release midnight. Fast early growth bending over as the launch cohort saturates.
- **Earners E_i(t):** per trophy, `log E_i` linear in `log(1 + t)`, slope shrunk
  toward the trophy's tier mean (story bronze fast, gold/platinum slow), because a
  single trophy's counts are noisy when few players hold it.

Exophase refreshes in jumps, not continuously: the house's reads show the page
frozen for hours, then jumping at a refresh (launch night: 638 players tracked
23:50Z–04:00Z, then 3,868 at 04:30Z, every rate but the one at 100% lower). A
refresh is a **dilution** event — newly tracked players hold fewer trophies — so
the history is first collapsed into distinct refresh states (consecutive identical
reads are one observation) and the fits run on states, not reads.

When the history holds a single state (no refresh seen yet) no slope can be
fitted, so the model falls back to a launch-refresh prior calibrated on the
launch-night refresh the kit's rules publish: players 638 → 3,868 and trophies
earned 943 → 4,276 between the first probe and 04:30Z 19 Nov — a players slope of
0.963 and an earners slope of 0.156 in `log(1 + t)` space, i.e. rates fall to
about a fifth (×0.221) over the first 4.5 hours. A trophy every tracked player
holds (100%) stays at 100%.

Quantile spreads are honest logit spreads that widen with the extrapolation
distance in log-time: slope uncertainty from each fit, a per-tier base spread
(late/rare trophies are least predictable), and a model-risk term for the
log-time approximation bending over a multi-day horizon.

## Local check

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Practice cases p00/p01: medians come out at last-rate × 0.221 (p00) and × 0.206
(p01), the dilution the dev board's top models use; the practice loss itself is
only readable from the server-side dev grade (answers are not in the kit).
