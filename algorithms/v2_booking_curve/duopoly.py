from typing import Any, Optional, Tuple, Union
import uuid
import numpy as np
import pandas as pd
import pickle
import os

# ==============================================================================
# ALGORITHM V3: CAPACITY UTILIZATION (BOOKING CURVE) PACING CONTROLLER
# ==============================================================================
# Strategy Overview:
# 1. Target Booking Curve: Derived from top 20% highest revenue seasons.
#    Target seats pacing roughly: S*(t) = 80 * (utilization at period t)
# 2. Pacing Feedback Controller:
#    - Behind target (sold < target): Lower price slightly below competitor to win demand.
#    - Ahead of target (sold > target): Raise price above competitor to conserve inventory.
#    - On pace: Shadow competitor's price at competitive margin.
# 3. Monopoly Surge: If competitor runs out of capacity, immediately surge price
#    to harvest monopoly profits (55.0 - 75.0).
# 4. Endgame Clearance: Clear remaining seats in final periods (t > 85) to avoid spoilage.
# 5. Robust State Logging: Maintains official DPC feedback pickle history.
# ==============================================================================

# Target cumulative seats sold at each milestone (1 to 100) based on top revenue seasons
TARGET_BOOKING_CURVE = {
    1: 0, 5: 2, 10: 5, 15: 8, 20: 11, 25: 14, 30: 17, 35: 20, 
    40: 23, 45: 27, 50: 32, 55: 36, 60: 41, 65: 45, 70: 50, 
    75: 55, 80: 60, 85: 65, 90: 71, 95: 75, 100: 79
}

def get_target_seats(period: int) -> float:
    """Interpolate target sold seats for any period t in [1, 100]."""
    if period <= 1:
        return 0.0
    if period >= 100:
        return 79.0
    keys = sorted(TARGET_BOOKING_CURVE.keys())
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if k1 <= period <= k2:
            s1, s2 = TARGET_BOOKING_CURVE[k1], TARGET_BOOKING_CURVE[k2]
            return s1 + (s2 - s1) * (period - k1) / (k2 - k1)
    return 0.8 * period


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[Optional[float], Any]:
    """Determine price for next period using Booking Curve Pacing Control."""
    t = selling_period_in_current_season
    season = current_selling_season

    # Period 101: End-of-season reporting period (platform sets no price)
    if t == 101:
        return None, information_dump

    first_day = (t == 1)
    first_season = (season == 1)

    # Initialize feedback dictionary on period 1
    if first_day:
        if information_dump is None or not isinstance(information_dump, dict):
            information_dump = _initialize_data_feedback()
        if first_season:
            information_dump['current_simulation'] = str(uuid.uuid4())
        information_dump['cumulative_revenue_current_selling_season'] = 0.0

    # -------------------------------------------------------------------------
    # PRICING DECISION LOGIC
    # -------------------------------------------------------------------------
    if first_day:
        # Day 1: Start with a competitive opening price
        price = 36.0
    else:
        # Extract history
        own_past_prices = prices_historical_in_current_season[0, :]
        comp_past_prices = prices_historical_in_current_season[1, :]
        p_comp_last = float(comp_past_prices[-1])
        p_own_last = float(own_past_prices[-1])
        
        sold_seats = int(np.sum(demand_historical_in_current_season))
        remaining_capacity = max(0, 80 - sold_seats)
        
        # If we have 0 capacity left, price does not matter, but keep valid float
        if remaining_capacity <= 0:
            price = 99.0
        # CASE A: Competitor has run out of capacity -> MONOPOLY SURGE
        elif not competitor_has_capacity_current_period_in_current_season:
            if remaining_capacity > 30:
                price = 55.0
            elif remaining_capacity > 10:
                price = 68.0
            else:
                price = 78.0
        # CASE B: Endgame clearance (t >= 85)
        elif t >= 85 and remaining_capacity > (100 - t) * 0.9:
            # We have too many seats for the remaining days -> discount to avoid spoilage
            price = max(14.0, min(p_comp_last - 2.0, 22.0))
        # CASE C: Normal Booking Curve Pacing Control
        else:
            target_sold = get_target_seats(t)
            delta = sold_seats - target_sold  # delta > 0: ahead of pace; delta < 0: behind pace
            
            if delta < -2.0:
                # Behind target (selling too slow) -> undercut competitor to capture demand
                undercut = 0.8 + 0.3 * min(abs(delta), 10.0)
                price = max(12.0, p_comp_last - undercut)
            elif delta > 2.0:
                # Ahead of target (selling too fast) -> raise price above competitor to protect inventory
                markup = 1.5 + 0.5 * min(delta, 10.0)
                price = min(85.0, max(p_comp_last + markup, p_own_last + 1.0))
            else:
                # On pace -> stay closely aligned with competitor
                price = max(18.0, min(p_comp_last, 42.0))

    # Round price to 2 decimal places and clamp to sensible boundaries
    if price is not None:
        price = float(np.clip(round(price, 2), 5.0, 95.0))

    # -------------------------------------------------------------------------
    # BOOKKEEPING & FEEDBACK LOGGING (Matches official DPC template)
    # -------------------------------------------------------------------------
    if not first_day and demand_historical_in_current_season is not None:
        try:
            last_demand = float(demand_historical_in_current_season[-1])
            last_own_price = float(prices_historical_in_current_season[0, -1])
            last_comp_price = float(prices_historical_in_current_season[1, -1])
            sold_so_far = float(np.sum(demand_historical_in_current_season))
            rem_cap = max(0.0, 80.0 - sold_so_far)
            period_revenue = last_demand * last_own_price
            
            information_dump['cumulative_revenue_current_selling_season'] += period_revenue
            sim_id = information_dump.get('current_simulation', '')
            cum_rev = information_dump['cumulative_revenue_current_selling_season']
            
            new_row = pd.DataFrame([{
                'simulation': sim_id,
                'day': t - 1,
                'season': season,
                'demand': last_demand,
                'own_price': last_own_price,
                'competitor_price': last_comp_price,
                'remaining_capacity': rem_cap,
                'revenue': period_revenue,
                'cumulative_revenue': cum_rev
            }])
            
            if 'history' in information_dump and isinstance(information_dump['history'], pd.DataFrame):
                information_dump['history'] = pd.concat([information_dump['history'], new_row], ignore_index=True)
                
            # At end of season (t >= 100), persist to disk
            if t >= 100:
                try:
                    with open('duopoly_feedback.data', 'wb') as handle:
                        pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
        except Exception:
            pass

    return price, information_dump


def _initialize_data_feedback() -> dict:
    """Initialize or load persistent feedback dictionary."""
    if os.path.isfile('duopoly_feedback.data'):
        try:
            with open('duopoly_feedback.data', 'rb') as handle:
                return pickle.load(handle)
        except Exception:
            pass
            
    df_empty = pd.DataFrame({
        'simulation': pd.Series(dtype='object'),
        'day': pd.Series(dtype='float'),
        'season': pd.Series(dtype='float'),
        'demand': pd.Series(dtype='float'),
        'own_price': pd.Series(dtype='float'),
        'competitor_price': pd.Series(dtype='float'),
        'remaining_capacity': pd.Series(dtype='float'),
        'revenue': pd.Series(dtype='float'),
        'cumulative_revenue': pd.Series(dtype='float'),
    })
    return {
        'history': df_empty,
        'current_simulation': '',
        'cumulative_revenue_current_selling_season': 0.0
    }