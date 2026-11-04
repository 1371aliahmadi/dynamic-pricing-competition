# v5 — learn the cheaper bonus + anticipate a rival who reacts (prepared 2026-10-25 for the Mon 26 Oct run)

## Why change v4
Oct 24 (v4): $1,380/season, rank 25, total $552,089. All 4 competitions vs KickassMoth.
- Seasons 21–50: v4 had learned that the market barely reacts to the price LEVEL (b: −0.054 → −0.015)
  and earned $2,000–3,900/season at $44–66.
- Season 51: KickassMoth dropped to ~$20 in all competitions; v4 followed, KickassMoth followed v4 →
  price war, ~$900–1,000/season for the last 50 seasons.
- Data: sales/day ≈ 0.5–0.6 when cheaper, ≈ 0.3 when dearer — at ANY price from $15 to $70.
  The cheaper bonus matters, the level hardly. v4 used a FIXED cheaper bonus (0.67).
- Both players follow each other: corr(rival today, our price yesterday) = 0.91, and the reverse 0.91.
  v4 assumed the rival's price is fixed → kept undercutting → spiral (slide "Pricing strategies":
  price matching risks a price-decline spiral).

## What v5 adds to v4
1. Learns `d` (cheaper bonus) per competition together with `a` and `b` (MAP Poisson regression,
   Newton's method, prior 0.67 ± 0.5, d ≥ 0).
2. Learns the rival's reaction: comp_t = α + β·my_{t−1} + γ·comp_{t−1} (least squares on the last
   1,500 observations; re-fitted every 10 days). Rival = **follower** if β > 0.5, β > γ, R² > 0.5.
3. If follower: for every candidate price p, predict the rival moves to (α + β·p)/(1 − γ) → undercutting
   no longer "buys" the cheaper bonus, the optimiser picks the best price LEVEL (a simple Stackelberg /
   game-theoretic response). Otherwise: v4 behaviour.
4. Unchanged from v4: capacity rule, exploration (±15 % → ±3 %), zero-sales safety, feedback summary
   (now also logs d, follower flag, β).

## Tests
| Simulator | v2 | v4 | **v5** |
|---|---|---|---|
| sim3: reacting opponents, 25 comps × 60 seasons | $1,927 | $2,133 | **$2,270** (beats v4 in 14/25, v2 in 20/25) |
| … vs undercutters | $1,304 | $1,807 | **$2,285** |
| … vs followers | $1,519 | $1,560 | **$1,749** |
| … vs KickassMoth-like (regime switch at s51) | $1,603 | $1,577 | **$1,655** |
| … vs fixed price | $2,057 | $2,702 | $2,704 |
| … vs recorded paths (don't react) | **$3,153** | $3,020 | $2,960 |
| sim2: own demand per competition, no reaction (24 × 40) | $2,380 | $3,289 | $3,154 (equal without one unrealistic market where the rival priced $300–450) |
| Oct-24 replica (follower → switch at s51), seasons 51–80 | – | $2,261 | **$2,849** (+26 %) |
Worst call 0.03 s. No crashes in edge tests.

## Known limitations
- Slightly worse (−2 %) vs opponents that ignore us (recorded paths).
- Learning d needs both cheaper and dearer observations; if we are always cheaper, d stays near the prior.
- Follower detection needs ~150 observations (≈ 2 seasons); regime switches are picked up within ~10 seasons.
- Simulators are ours; the platform is the real test.
