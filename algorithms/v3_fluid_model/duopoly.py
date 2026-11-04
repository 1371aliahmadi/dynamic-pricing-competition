"""
DPC duopoly — v3: model-based "fluid" pricing with online market-size correction.

Idea in one sentence:
  Every day, pick the price that maximises expected revenue for the rest of the
  season, given (1) how many seats we have left, (2) how many days are left,
  (3) the competitor's latest price, and (4) a demand model learned from our
  past results — corrected on the fly by how fast we are actually selling.

Step by step (each step is one small function below):
  1. demand_rate(p, pc, t, comp_cap): expected sales per day at our price p.
     Poisson regression fitted on 270k past periods (all our runs Oct 7-19):
        log(rate) = a + a_t + b*p + c*pc + d*[we are cheaper] + e*[same price]
  2. market_factor(): this season's market can be bigger/smaller than average.
     factor = (actual sales + 3) / (model-predicted sales + 3)   (3 = prior, so
     early days don't over-react). Multiplies every predicted rate.
  3. best_price(): try a grid of prices. For each price, assume we keep it for
     the remaining days and the competitor keeps his last price:
        expected sales  = min(seats_left, factor * sum of daily rates)
        expected revenue = price * expected sales
     Choose the price with the highest expected revenue.
     -> If seats are scarce, this automatically raises the price (protect stock).
     -> If seats are plentiful, it lowers the price to avoid empty seats.
     -> If the competitor is sold out, the monopoly demand curve is used.
"""
from typing import Any, Optional, Tuple, Union
import numpy as np

CAPACITY = 80
HORIZON = 100

# ---- 1. demand model (fitted coefficients) ----------------------------------
# refit 2026-10-20 on all our runs (Oct 7, 10, 12, 17, 19; 270k periods)
A = -0.52
B_OWN = -0.0362       # own price effect   (higher own price  -> fewer sales)
C_COMP = 0.0148       # competitor price   (higher comp price -> more sales)
D_CHEAPER = 0.67      # bonus when we are the cheaper seller
E_TIE = 0.06          # bonus when prices are equal
B_MONO = -0.0275      # own price effect when competitor is sold out
K_MONO = 1.056        # extra demand when competitor is sold out
TIME_EFFECT = np.zeros(HORIZON + 2)
for t in range(1, HORIZON + 2):
    TIME_EFFECT[t] = (0.0 if t <= 20 else 0.23 if t <= 40 else 0.39 if t <= 60
                      else 0.48 if t <= 80 else 0.53 if t <= 90 else 0.36)

PRICE_GRID = np.round(np.arange(5.0, 150.01, 0.5), 2)
PRIOR = 3.0


def demand_rate(p, pc, t_idx, comp_cap):
    """Expected sales per day. p can be an array of prices, t_idx an array of days."""
    p = np.asarray(p, dtype=float)[:, None]
    tt = TIME_EFFECT[np.asarray(t_idx)][None, :]
    if comp_cap:
        x = A + tt + B_OWN * p + C_COMP * pc
        x = x + np.where(p < pc - 0.01, D_CHEAPER, np.where(np.abs(p - pc) <= 0.01, E_TIE, 0.0))
    else:
        x = A + tt + B_MONO * p + K_MONO
    return np.exp(x)          # shape (n_prices, n_days)


def market_factor(prices_hist, demand_hist, comp_cap_hist):
    """Ratio actual / predicted sales so far this season (shrunk towards 1)."""
    if demand_hist is None or len(demand_hist) == 0:
        return 1.0
    pred = 0.0
    for i in range(len(demand_hist)):
        pred += demand_rate([prices_hist[0, i]], prices_hist[1, i], [i + 1],
                            comp_cap_hist[i] if i < len(comp_cap_hist) else True)[0, 0]
    return float((np.sum(demand_hist) + PRIOR) / (pred + PRIOR))


def best_price(t, seats_left, pc, comp_cap, factor):
    days = np.arange(t, HORIZON + 1)
    total = factor * demand_rate(PRICE_GRID, pc, days, comp_cap).sum(axis=1)
    exp_sales = np.minimum(seats_left, total)
    revenue = PRICE_GRID * exp_sales
    # also consider exactly matching / just undercutting the competitor
    if comp_cap:
        extra = np.array([pc, pc - 0.01])
        extra = extra[extra > 1.0]
        if len(extra):
            tot2 = factor * demand_rate(extra, pc, days, comp_cap).sum(axis=1)
            rev2 = extra * np.minimum(seats_left, tot2)
            if rev2.max() > revenue.max():
                return float(extra[np.argmax(rev2)])
    return float(PRICE_GRID[np.argmax(revenue)])


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    t = selling_period_in_current_season
    comp_cap = bool(competitor_has_capacity_current_period_in_current_season)
    try:
        if t == 1 or not isinstance(information_dump, dict):
            information_dump = {'comp_cap_hist': []}
        information_dump['comp_cap_hist'].append(comp_cap)

        if t > HORIZON:                      # period 101: price is not used
            return 50.0, information_dump

        if t == 1:
            pc = 40.0                        # no info yet: assume a typical price
            seats_left = CAPACITY
            factor = 1.0
        else:
            prices = np.asarray(prices_historical_in_current_season, dtype=float)
            demand = np.asarray(demand_historical_in_current_season, dtype=float)
            pc = float(prices[1, -1])
            seats_left = CAPACITY - int(demand.sum())
            factor = market_factor(prices, demand, information_dump['comp_cap_hist'])

        if seats_left <= 0:
            return 150.0, information_dump

        price = best_price(t, seats_left, pc, comp_cap, factor)
        information_dump['last_factor'] = factor
        return float(np.clip(round(price, 2), 1.0, 150.0)), information_dump
    except Exception:
        # never crash: fall back to a safe price
        return 40.0, information_dump if isinstance(information_dump, dict) else None
