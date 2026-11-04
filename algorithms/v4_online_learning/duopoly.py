"""
DPC duopoly — v4: v3 + LEARN THIS COMPETITION'S DEMAND CURVE WHILE PLAYING.

Why: v3 used one demand model (fitted on all past runs) for every competition.
But the platform draws a NEW demand model per competition: in some markets the price
where elasticity = 1 is ~$40, in others ~$4 (RationalLion, Oct 21: we sold 0 seats at $27.50).

How v4 works (one decision per day):
  1. Data:      store every (day, my price, comp price, comp has capacity, my sales) of this
                competition in information_dump (it survives across the 100 seasons).
  2. Learn:     fit   log(rate) = a + time(t) + b*p + c*pc + d*[cheaper] + e*[equal]
                with  a (market size) and b (price sensitivity) learned for THIS competition
                by Poisson regression (Newton's method), with a prior = pooled values from
                past runs (MAP / ridge towards prior). c, d, e, time effects come from the prior.
  3. Optimise:  same as v3: for each price on a grid, expected revenue for the rest of the
                season = price * min(seats_left, factor * sum of predicted daily sales).
  4. Explore:   early on, multiply the chosen price by a small random factor so prices vary
                and b can be estimated. Exploration fades out over the seasons.
  5. Safety:    if we sold nothing for 6+ days while the model expected sales, step the price
                down 12% per day (the model is obviously wrong here) — this also creates data.
  6. Log:       write a compact per-season summary to duopoly_feedback.data.
"""
from typing import Any, Optional, Tuple, Union
import pickle
import numpy as np

CAPACITY, HORIZON = 80, 100

# ---- prior (pooled fit on all runs Oct 7-19, 270k periods) --------------------
A0, B0 = -0.52, -0.0362
C_COMP, D_CHEAPER, E_TIE = 0.0148, 0.67, 0.06
MONO_RATIO, K_MONO = 0.76, 1.056          # rival sold out: b*0.76 and +1.056
PRIOR_SD_A, PRIOR_SD_LOGB = 1.0, 1.0       # wide: allows |b| from ~0.013 to ~0.1 within 1 sd
TIME_EFFECT = np.array([0.0] + [0.0 if t <= 20 else 0.23 if t <= 40 else 0.39 if t <= 60
                                else 0.48 if t <= 80 else 0.53 if t <= 90 else 0.36
                                for t in range(1, HORIZON + 2)])
PRICE_GRID = np.round(np.exp(np.linspace(np.log(1.0), np.log(200.0), 220)), 2)
PRIOR_N = 3.0


def offset(pc, t, comp_cap, p):
    """Everything in log(rate) except a and b*p (known parts)."""
    t = np.asarray(t)
    p = np.asarray(p, dtype=float)
    pc = np.asarray(pc, dtype=float)
    cc = np.asarray(comp_cap, dtype=bool)
    duo = C_COMP * pc + np.where(p < pc - 0.01, D_CHEAPER, np.where(np.abs(p - pc) <= 0.01, E_TIE, 0.0))
    return TIME_EFFECT[t] + np.where(cc, duo, K_MONO)


def fit_ab(P, PC, T, CC, Y, a_init, logb_init):
    """MAP Poisson fit of a and log(-b) with Gaussian prior. Newton's method, a few steps."""
    a, lb = a_init, logb_init
    off = offset(PC, T, CC, P)
    pm = np.where(CC, P, P * MONO_RATIO)
    for _ in range(8):
        b = -np.exp(lb)
        eta = np.clip(a + b * pm + off, -20, 5)
        mu = np.exp(eta)
        r = Y - mu
        # gradient of log-posterior wrt (a, lb); d eta/d lb = b*pm
        da = np.ones_like(P); dl = b * pm
        g = np.array([np.sum(r * da) - (a - A0) / PRIOR_SD_A ** 2,
                      np.sum(r * dl) - (lb - np.log(-B0)) / PRIOR_SD_LOGB ** 2])
        H = np.array([[np.sum(mu * da * da) + 1 / PRIOR_SD_A ** 2, np.sum(mu * da * dl)],
                      [np.sum(mu * da * dl), np.sum(mu * dl * dl) - np.sum(r * dl) + 1 / PRIOR_SD_LOGB ** 2]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -1.0, 1.0)
        a, lb = a + step[0], lb + step[1]
        lb = float(np.clip(lb, np.log(0.005), np.log(0.6)))
        if np.max(np.abs(step)) < 1e-4:
            break
    return float(a), float(lb)


def rate(prices, pc, days, comp_cap, a, b):
    prices = np.asarray(prices, dtype=float)[:, None]
    days = np.asarray(days)[None, :]
    if comp_cap:
        x = a + TIME_EFFECT[days] + b * prices + C_COMP * pc
        x = x + np.where(prices < pc - 0.01, D_CHEAPER, np.where(np.abs(prices - pc) <= 0.01, E_TIE, 0.0))
    else:
        x = a + TIME_EFFECT[days] + b * MONO_RATIO * prices + K_MONO
    return np.exp(np.clip(x, -20, 5))


def best_price(t, seats_left, pc, comp_cap, a, b, factor):
    days = np.arange(t, HORIZON + 1)
    cands = PRICE_GRID
    if comp_cap and pc > 1.0:
        cands = np.concatenate([PRICE_GRID, [pc, pc - 0.01]])
    tot = factor * rate(cands, pc, days, comp_cap, a, b).sum(axis=1)
    rev = cands * np.minimum(seats_left, tot)
    return float(cands[np.argmax(rev)])


def _new_dump():
    return {'P': [], 'PC': [], 'T': [], 'CC': [], 'Y': [], 'a': A0, 'lb': float(np.log(-B0)),
            'cc_season': [], 'zero_run': 0, 'n_fit': 0, 'summary': [], 'rng': np.random.default_rng(12345)}


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    t = selling_period_in_current_season
    season = current_selling_season
    comp_cap = bool(competitor_has_capacity_current_period_in_current_season)
    try:
        dmp = information_dump if isinstance(information_dump, dict) and 'P' in information_dump else _new_dump()

        # ---- 1. store yesterday's observation ----------------------------------
        if t == 1:
            dmp['cc_season'] = []
            dmp['zero_run'] = 0
        dmp['cc_season'].append(comp_cap)
        prices = demand = None
        if t > 1 and prices_historical_in_current_season is not None:
            prices = np.asarray(prices_historical_in_current_season, dtype=float)
            demand = np.asarray(demand_historical_in_current_season, dtype=float)
            sold_before = demand[:-1].sum()
            if sold_before < CAPACITY:                       # only periods where we could sell
                dmp['P'].append(prices[0, -1]); dmp['PC'].append(prices[1, -1]); dmp['T'].append(t - 1)
                dmp['CC'].append(dmp['cc_season'][t - 2] if t - 2 < len(dmp['cc_season']) else True)
                dmp['Y'].append(demand[-1])
            dmp['zero_run'] = dmp['zero_run'] + 1 if demand[-1] == 0 else 0
            if len(dmp['Y']) > 6000:                         # keep memory small
                for k in ('P', 'PC', 'T', 'CC', 'Y'):
                    del dmp[k][:1000]

        if t > HORIZON:                                      # period 101: log season, price unused
            if demand is not None:
                dmp['summary'].append({'season': season, 'revenue': float((prices[0] * demand).sum()),
                                       'sold': float(demand.sum()), 'a': dmp['a'], 'b': -np.exp(dmp['lb'])})
                if season % 10 == 0 or season >= 100:
                    try:
                        with open('duopoly_feedback.data', 'wb') as fh:
                            pickle.dump({'summary': dmp['summary']}, fh)
                    except Exception:
                        pass
            return 50.0, dmp

        # ---- 2. learn a, b for this competition --------------------------------
        n = len(dmp['Y'])
        if n >= 10 and (t % 5 == 1 or n < 300):
            keep = slice(max(0, n - 6000), n)
            dmp['a'], dmp['lb'] = fit_ab(np.array(dmp['P'][keep]), np.array(dmp['PC'][keep]),
                                         np.array(dmp['T'][keep]), np.array(dmp['CC'][keep]),
                                         np.array(dmp['Y'][keep]), dmp['a'], dmp['lb'])
        a, b = dmp['a'], -np.exp(dmp['lb'])

        # ---- 3. optimise ------------------------------------------------------------
        if t == 1:
            pc, seats_left, factor = (1.0 / -b), CAPACITY, 1.0
        else:
            pc = float(prices[1, -1])
            seats_left = CAPACITY - int(demand.sum())
            # within-season level correction (as in v3), shrunk towards 1
            ccs = dmp['cc_season']
            pred = rate_hist = 0.0
            for i in range(len(demand)):
                pred += rate([prices[0, i]], prices[1, i], [i + 1], ccs[i] if i < len(ccs) else True, a, b)[0, 0]
            factor = float(np.clip((demand.sum() + PRIOR_N) / (pred + PRIOR_N), 0.33, 3.0))
        if seats_left <= 0:
            return 200.0, dmp
        price = best_price(t, seats_left, pc, comp_cap, a, b, factor)
        if comp_cap and t > 1 and pc > 1.0:                 # sanity cap: never > 3x the rival
            price = min(price, 3.0 * pc)

        # ---- 4. explore (fades out) / 5. safety ----------------------------------------
        explore = 0.15 * max(0.0, 1.0 - (season - 1) / 15.0) + 0.03
        price *= float(np.exp(dmp['rng'].normal(0, explore)))
        if t > 1 and dmp['zero_run'] >= 6:
            last = float(prices[0, -1])
            price = min(price, last * 0.88)

        return float(np.clip(round(price, 2), 0.5, 200.0)), dmp
    except Exception:
        try:
            last = float(prices_historical_in_current_season[0, -1])
        except Exception:
            last = 30.0
        return last, information_dump
