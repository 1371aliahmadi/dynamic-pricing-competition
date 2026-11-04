# v3 — model-based "fluid" pricing (uploaded 2026-10-20 for the 21 Oct run)

## Why change v2
v2 = hand-made rules with fixed numbers (cap $42 when on pace, $22 in endgame, $55/68/78 when rival sold out).
Real results: Oct 17 $2,955/season (rank 9), Oct 19 $2,320/season. Lost badly to the low-price GorgeousNewt ($1,509).

## How v3 works
1. **Demand model** — Poisson regression on 270k past periods (all our runs Oct 7–19):
   log(sales/day) = -0.52 + time_effect(t) - 0.0362·own_price + 0.0148·comp_price + 0.67·[we are cheaper] + 0.06·[same price]
   Rival sold out: log(sales/day) = -0.52 + time_effect(t) - 0.0275·own_price + 1.056
2. **Market factor** — (seats sold + 3) / (seats predicted + 3): scales the model to this season's market size.
3. **Daily decision** — for each price $5–$150: expected revenue = price × min(seats left, predicted sales for remaining days). Pick the best.

## Local simulation (250 seasons, real opponent price paths from all runs)
| scenario | v2 | v3 draft (old fit) | **v3 (uploaded)** |
|---|---|---|---|
| old demand world | $2,742 / 68.3 | $3,125 / 75.1 | $3,109 / 74.5 |
| refit demand world | $2,455 / 65.9 | $2,871 / 73.3 | $2,868 / 73.2 |
| more price-sensitive | $1,754 / 55.4 | $2,276 / 67.2 | $2,252 / 66.7 |
| less price-sensitive | $3,291 / 73.2 | $3,685 / 77.3 | $3,707 / 77.3 |
| weak monopoly demand | $2,310 / 63.6 | $2,698 / 70.1 | $2,696 / 69.9 |
(revenue per season / seats sold) — v3 is +$390 to +$520 per season vs v2 in every scenario. Max ~2 ms per call, no errors.

## Known limitations
- Simulator uses the same model family as v3 → optimistic for v3.
- Opponent price paths are replayed; real opponents react to us and change their code 3×/week.
- Assumes competitor keeps his last price; does not learn the opponent across seasons (idea for v4).
