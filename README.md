# Roblox Up-and-Coming top-10 forecaster

Forecasts Roblox's Up-and-Coming chart (device computer, country US), place
by place, for every hour of the 72 hours after entries close — from the
chart's history alone. Entry for the make.fun bounty
`roblox-up-and-coming-forecast-01` ($500).

## How it works

Monte Carlo rank diffusion (`main.py`, Python 3.13 stdlib only, no network,
deterministic — fixed seed, two runs write byte-identical output):

- Each game has a continuous latent rank position. Every simulated hour it
  moves by its drift (its own recent rank velocity, shrunk toward its rank
  band's mean drift and decaying toward that mean with a 24h half-life) plus
  Gaussian noise whose SD depends on its rank band (0.18 / 0.36 / 0.84 /
  1.48 ranks/hour for ranks 1-3 / 4-10 / 11-30 / 31+).
- Games leave by a per-hour hazard by band, or at `created` + 28 days when
  the creation date is known. New games join near the bottom (~0.55/hour)
  with drift from a mixture (rockets / climbers / faders). Pool games may
  "ignite" and climb fast for 12-36 hours — games do ignite, climb 1-5
  ranks/hour, hold the top 1-3 days, then decay (Do Not Stop. went
  37 -> 7 -> 2 -> 1 in ~26h).
- 1000 simulated paths x 72 hours: the frequency of each (game, place) is
  its probability. Games below 0.003 are left unnamed; the leftover — new
  games not yet in the history plus sub-threshold games — is the unnamed
  share, exactly as the scorer spreads it. The unnamed share at place 1 is
  large (~0.39) because new games take the top spot often: 3 of the 10
  place-1 slots in the practice window were a game never in the history.

Player counts are deliberately not used: they do not predict rank moves
(correlation -0.02 with the next rank change).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR/history.ndjson` and `task.json` (places, output
name), writes `$MAKEFUN_OUTPUT_DIR/forecast.json`. Runs in ~11s on the
practice history (limit 300s).

## Results (local backtest)

The kit's scorer is not published, so this was validated with a local
re-implementation of the scoring rule (verified: it reproduces the house
baseline's published practice score of -43.512039 exactly) and the practice
window rebuilt from the public archive
(github.com/Kenny-Dhub/roblox-snapshots — the extraction reproduces the
kit's last history read exactly).

| cutoff | model | house baseline | diff |
|---|---|---|---|
| Practice (train to 2026-11-07, 10 window reads) | **-32.51** | -43.51 | +11.00 |
| Walk-forward B (train to 2026-11-04) | -41.91 | -56.91 | +15.01 |
| Walk-forward C (train to 2026-11-02) | -36.55 | -42.86 | +6.31 |

Practice: 7.5 mean top-10 hits, 56.4 mean agreement (baseline 6.4 / 53.5).
For reference the house model scored -37.2 on practice and the best
practice entry so far -34.8. Output shape validated: 307 rows, 17.6 KB,
21-45 rows per place (limit 100), per-place sums 0.61-0.92 (limit 1),
top10 exactly 10 distinct games.

## License

MIT — see LICENSE.
