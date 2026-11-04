"""
DPC duopoly — v5: v4 + learn the "cheaper bonus" + anticipate a competitor who reacts to us.

What v4 got wrong on Oct 24 (all 4 competitions vs KickassMoth, rank 25):
  * The market hardly reacted to the price LEVEL, but strongly to being cheaper than the rival.
    v4 learned the level effect (b) but used a FIXED cheaper bonus from old data.
  * KickassMoth follows our price (its price today ~ our price yesterday). v4 assumed the
    rival's price is fixed, kept undercutting him, he followed us down -> price war (~$20).

What v5 changes:
  1. Learn a (market size), b (price sensitivity) AND d (cheaper bonus) for THIS competition
     (MAP Poisson regression, Newton's method, prior = pooled values from past runs).
  2. Learn how the rival reacts:  comp_t = alpha + beta * my_{t-1} + gamma * comp_{t-1}
     (least squares on this competition's recent data). If beta is clearly positive the
     rival is a FOLLOWER: then for every candidate price p we predict that his price will
     move to (alpha + beta*p) / (1 - gamma) -- so undercutting him does not keep us cheaper,
     it only drags both prices down. The optimiser then chooses the best LEVEL instead.
  3. Everything else as v4: capacity rule (price x min(seats, demand)), exploration that
     fades out, zero-sales safety, feedback summary file.
"""
from typing import Any, Optional, Tuple, Union
import pickle
import numpy as np

CAPACITY, HORIZON = 80, 100

# ---- prior (pooled fit on all runs Oct 7-19) ------------------------------------
A0, B0, D0 = -0.52, -0.0362, 0.67
C_COMP, E_TIE = 0.0148, 0.06
MONO_RATIO, K_MONO = 0.76, 1.056
SD_A, SD_LOGB, SD_D = 1.0, 1.0, 0.5
TIME_EFFECT = np.array([0.0] + [0.0 if t <= 20 else 0.23 if t <= 40 else 0.39 if t <= 60
                                else 0.48 if t <= 80 else 0.53 if t <= 90 else 0.36
                                for t in range(1, HORIZON + 2)])
PRICE_GRID = np.round(np.exp(np.linspace(np.log(1.0), np.log(200.0), 220)), 2)
PRIOR_N = 3.0
KEEP = 6000          # observations kept for the demand fit
REACT_WIN = 1500     # observations used for the rival-reaction fit


# ================================ demand model ===================================
def log_rate(p, pc, t, cc, a, b, d):
    """log(expected sales per day). p, pc, t, cc broadcast against each other."""
    p = np.asarray(p, float); pc = np.asarray(pc, float)
    cheaper = (p < pc - 0.01).astype(float)
    tie = (np.abs(p - pc) <= 0.01).astype(float)
    duo = a + TIME_EFFECT[t] + b * p + C_COMP * pc + d * cheaper + E_TIE * tie
    mono = a + TIME_EFFECT[t] + b * MONO_RATIO * p + K_MONO
    return np.where(cc, duo, mono)


def fit_abd(P, PC, T, CC, Y, th):
    """MAP fit of (a, log|b|, d) by Newton's method."""
    a, lb, d = th
    cheaper = ((P < PC - 0.01) & CC).astype(float)
    tie = ((np.abs(P - PC) <= 0.01) & CC).astype(float)
    pm = np.where(CC, P, P * MONO_RATIO)
    known = TIME_EFFECT[T] + np.where(CC, C_COMP * PC + E_TIE * tie, K_MONO)
    prior_mean = np.array([A0, np.log(-B0), D0])
    prior_prec = np.array([1 / SD_A ** 2, 1 / SD_LOGB ** 2, 1 / SD_D ** 2])
    for _ in range(10):
        b = -np.exp(lb)
        eta = np.clip(a + b * pm + d * cheaper + known, -20, 5)
        mu = np.exp(eta)
        r = Y - mu
        X = np.vstack([np.ones_like(P), b * pm, cheaper])          # d eta / d theta
        th_v = np.array([a, lb, d])
        g = X @ r - prior_prec * (th_v - prior_mean)
        H = (X * mu) @ X.T + np.diag(prior_prec)
        H[1, 1] -= np.sum(r * b * pm)                               # 2nd derivative wrt log|b|
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -1.0, 1.0)
        a, lb, d = a + step[0], lb + step[1], d + step[2]
        lb = float(np.clip(lb, np.log(0.005), np.log(0.6)))
        d = float(np.clip(d, 0.0, 2.0))          # being cheaper never hurts
        if np.max(np.abs(step)) < 1e-4:
            break
    return float(a), float(lb), float(d)


# ============================ rival reaction model ===============================
def fit_reaction(MYP, CPP, CP):
    """comp_t = alpha + beta*my_{t-1} + gamma*comp_{t-1}  (least squares, small ridge)."""
    n = len(CP)
    if n < 150:
        return None
    X = np.column_stack([np.ones(n), MYP, CPP])
    y = np.asarray(CP)
    XtX = X.T @ X + np.diag([1e-6, 1e-3, 1e-3]) * n
    coef = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ coef
    r2 = 1 - resid.var() / max(y.var(), 1e-9)
    alpha, beta, gamma = coef
    is_follower = bool(beta > 0.5 and beta > gamma and r2 > 0.5)
    return dict(alpha=float(alpha), beta=float(beta), gamma=float(np.clip(gamma, 0.0, 0.8)),
                r2=float(r2), follower=is_follower)


# ================================ optimisation ===================================
def best_price(t, seats_left, pc_today, comp_cap, a, b, d, factor, react, my_last):
    days_future = np.arange(t + 1, HORIZON + 1)
    cands = PRICE_GRID
    if react is not None and react['follower'] and comp_cap:
        # rival follows us: today he plays his reaction to our yesterday's price,
        # from tomorrow on he moves towards his reaction to the price we choose now
        g = react['gamma']
        pc_t = react['alpha'] + react['beta'] * my_last + g * pc_today
        pc_future = np.clip((react['alpha'] + react['beta'] * cands) / (1 - g), 0.5, 300.0)
        r_today = np.exp(log_rate(cands, pc_t, t, True, a, b, d))
        r_future = np.exp(log_rate(cands[:, None], pc_future[:, None], days_future[None, :], True, a, b, d)).sum(axis=1) \
            if len(days_future) else 0.0
        tot = factor * (r_today + r_future)
    else:
        if comp_cap and pc_today > 1.0:
            cands = np.concatenate([PRICE_GRID, [pc_today, pc_today - 0.01]])
        days = np.arange(t, HORIZON + 1)
        tot = factor * np.exp(log_rate(cands[:, None], pc_today, days[None, :], comp_cap, a, b, d)).sum(axis=1)
    rev = cands * np.minimum(seats_left, tot)
    return float(cands[np.argmax(rev)])


def _new_dump():
    return {'P': [], 'PC': [], 'T': [], 'CC': [], 'Y': [],
            'MYP': [], 'CPP': [], 'CP': [],                    # rival-reaction data
            'th': (A0, float(np.log(-B0)), D0), 'react': None,
            'cc_season': [], 'zero_run': 0, 'summary': [], 'rng': np.random.default_rng(12345)}


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
        dmp = information_dump if isinstance(information_dump, dict) and 'MYP' in information_dump else _new_dump()

        # ---- 1. store yesterday's observation ----------------------------------------
        if t == 1:
            dmp['cc_season'] = []
            dmp['zero_run'] = 0
        dmp['cc_season'].append(comp_cap)
        prices = demand = None
        if t > 1 and prices_historical_in_current_season is not None:
            prices = np.asarray(prices_historical_in_current_season, dtype=float)
            demand = np.asarray(demand_historical_in_current_season, dtype=float)
            cc_y = dmp['cc_season'][t - 2] if t - 2 < len(dmp['cc_season']) else True
            if demand[:-1].sum() < CAPACITY:
                dmp['P'].append(prices[0, -1]); dmp['PC'].append(prices[1, -1]); dmp['T'].append(t - 1)
                dmp['CC'].append(cc_y); dmp['Y'].append(demand[-1])
            if t > 2 and cc_y:                                   # rival reaction: comp_{t-1} vs my_{t-2}
                dmp['MYP'].append(prices[0, -2]); dmp['CPP'].append(prices[1, -2]); dmp['CP'].append(prices[1, -1])
            dmp['zero_run'] = dmp['zero_run'] + 1 if demand[-1] == 0 else 0
            for k, lim in (('P', KEEP), ('MYP', REACT_WIN)):
                if len(dmp[k]) > lim + 500:
                    group = ('P', 'PC', 'T', 'CC', 'Y') if k == 'P' else ('MYP', 'CPP', 'CP')
                    for kk in group:
                        del dmp[kk][:500]

        if t > HORIZON:                                          # period 101: log the season
            if demand is not None:
                th, rc = dmp['th'], dmp['react'] or {}
                dmp['summary'].append({'season': season, 'revenue': float((prices[0] * demand).sum()),
                                       'sold': float(demand.sum()), 'a': th[0], 'b': -np.exp(th[1]), 'd': th[2],
                                       'follower': rc.get('follower'), 'beta': rc.get('beta')})
                if season % 10 == 0 or season >= 100:
                    try:
                        with open('duopoly_feedback.data', 'wb') as fh:
                            pickle.dump({'summary': dmp['summary']}, fh)
                    except Exception:
                        pass
            return 50.0, dmp

        # ---- 2. learn demand (a, b, d) and the rival's reaction -------------------------
        n = len(dmp['Y'])
        if n >= 10 and (t % 5 == 1 or n < 300):
            dmp['th'] = fit_abd(np.array(dmp['P']), np.array(dmp['PC']), np.array(dmp['T']),
                                np.array(dmp['CC'], dtype=bool), np.array(dmp['Y']), dmp['th'])
        if t == 1 or (t % 10 == 1 and len(dmp['CP']) >= 150):
            dmp['react'] = fit_reaction(np.array(dmp['MYP']), np.array(dmp['CPP']), np.array(dmp['CP']))
        a, lb, d = dmp['th']
        b = -np.exp(lb)

        # ---- 3. optimise ---------------------------------------------------------------
        if t == 1:
            pc, seats_left, factor, my_last = (1.0 / -b), CAPACITY, 1.0, 1.0 / -b
        else:
            pc = float(prices[1, -1]); my_last = float(prices[0, -1])
            seats_left = CAPACITY - int(demand.sum())
            ccs = dmp['cc_season']
            lr = log_rate(prices[0], prices[1], np.arange(1, len(demand) + 1),
                          np.array([ccs[i] if i < len(ccs) else True for i in range(len(demand))]), a, b, d)
            factor = float(np.clip((demand.sum() + PRIOR_N) / (np.exp(lr).sum() + PRIOR_N), 0.33, 3.0))
        if seats_left <= 0:
            return 200.0, dmp
        react = dmp['react'] if t > 1 else None
        price = best_price(t, seats_left, pc, comp_cap, a, b, d, factor, react, my_last)
        if comp_cap and t > 1 and pc > 1.0 and not (react and react['follower']):
            price = min(price, 3.0 * pc)

        # ---- 4. explore (fades out) / 5. safety ------------------------------------------
        explore = 0.15 * max(0.0, 1.0 - (season - 1) / 15.0) + 0.03
        price *= float(np.exp(dmp['rng'].normal(0, explore)))
        if t > 1 and dmp['zero_run'] >= 6:
            price = min(price, float(prices[0, -1]) * 0.88)
        return float(np.clip(round(price, 2), 0.5, 200.0)), dmp
    except Exception:
        try:
            last = float(prices_historical_in_current_season[0, -1])
        except Exception:
            last = 30.0
        return last, information_dump
