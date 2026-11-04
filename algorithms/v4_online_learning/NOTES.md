# v4 — learn this competition's demand curve while playing (prepared 2026-10-23 for the Sat 24 Oct run)

## Why change v3
Oct 21 (v3): $2,504/season, rank 17. Two competitions vs RationalLion collapsed ($624, $555):
we priced $27.50 for 70 days and sold 0 seats. v3 assumed the SAME price sensitivity in every
competition. Per-competition fits on all 34 of our competitions show the price where
elasticity = 1 ranges from $3.90 to $42.70 (10x). The platform draws a new demand model per
competition (Week 1 slides), constant across its 100 seasons.

## What v4 adds to v3
1. Stores every observation of this competition in information_dump (survives across seasons).
2. Re-estimates market size `a` and price sensitivity `b` by MAP Poisson regression
   (Newton's method, prior = pooled fit) — every 5 days, on the last 6000 observations.
3. Same optimisation as v3: price = argmax price × min(seats left, predicted remaining sales).
4. Exploration: random price noise ±15% in season 1, fading to ±3% by season 16.
5. Safety: after 6 days with zero sales, price ≤ 88% of yesterday's. Price ≤ 3× rival's price.
6. Writes a per-season summary (revenue, seats, a, b) to duopoly_feedback.data every 10 seasons.

## Tests
| Simulator | v2 | v3 | v4 |
|---|---|---|---|
| sim2: own demand per competition, 24 comps × 40 seasons | $2,380 | $2,741 | **$3,375** (beats v2 in 24/24) |
| sim2 with logit customer-choice demand, 12 × 20 | $1,995 | $2,040 | **$2,176** (beats v2 in 10/12) |
| old sim: pooled model, params change EVERY season | $2,350 | **$2,821** | $2,719 |
Cheap markets (best price ≈ $4–6): v3 $38–$578 → v4 $341–$873 per season.
Worst call 0.05 s (limit 5 s, avg 0.2 s). No crashes in edge tests.

## Known limitations
- Exploration costs ~2–4% in normal markets in the first seasons.
- Only a and b are learned; competitor effect, cheaper bonus and time pattern still from the prior.
- Simulators are built by us; the real test is the platform.
