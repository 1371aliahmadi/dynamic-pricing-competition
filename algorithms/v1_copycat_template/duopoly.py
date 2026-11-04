from typing import Any, Optional, Tuple, Union
import uuid
import numpy as np
import pandas as pd
import pickle

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    """
    MAIN PRICING FUNCTION WHICH IS REQUIRED IN ANY SUBMISSION

    Return the price to set for the next period.

    ARGUMENTS
    ----------
    current_selling_season : int
            The current selling season (1, 2, 3, ..., 100).
    selling_period_in_current_season : int
            The period in the current season (1, 2, ..., 100).
    prices_historical_in_current_season : Union[np.ndarray, None]
            A two-dimensional array of historical prices: 
                rows index the competitors
                columns index the historical selling periods. 
            Equal to `None` if `selling_period_in_current_season == 1`.
    demand_historical_in_current_season : Union[np.ndarray, None]
            A one-dimensional array of historical (own) demand. 
            Equal to `None` if `selling_period_in_current_season == 1`.
    competitor_has_capacity_current_period_in_current_season : bool
            `False` if competitor is out of stock, else `False`
    information_dump : Any, optional, default `None`
            Custom object to pass information from one selling_period to the other.
            Equal to `None` if `selling_period_in_current_season == 1`.
            To pass information from one selling_season to the other, 
            use the 'duopoly_feedback.data' file.

    RETURNS
    -------
    Tuple[float, Any]
        The price and a the information_dump (with, e.g., a state of the model).
    
    
    Examples
    --------

    >>> prices_historical_in_current_season.shape == (2, selling_period_in_current_season - 1)
    True

    >>> demand_historical_in_current_season.shape == (selling_period_in_current_season - 1, )
    True

    Hints
    ----------
    To pass to yourself information for download in the DPC website, use `duopoly_feedback.data`.
    This file will be available in the website under `Download Feedback Data` in the results page.
    
    """
    # first set some shorter names for convenience
    day = selling_period_in_current_season
    season = current_selling_season
    demand = demand_historical_in_current_season
    prices = prices_historical_in_current_season

    # set some indicator variables
    first_day = True if day == 1 else False
    first_season = True if season == 1 else False
    season_previous_period = season if not first_day else season - 1

    if first_day:
        # Randomize in the first period of the season
        price = np.random.uniform(1, 100)
    else:
        # Copycat: Set the price that my competitor set in the previous period
        price = prices_historical_in_current_season[1, -1]

    # if first day of simulation, initialize information dump from feedback data object
    if first_day:
        information_dump = _initialize_data_feedback()
        if first_season:
            # we just started a new simulation, let's give it an ID
            my_simulation_id = "%s" % uuid.uuid4()
            information_dump['current_simulation'] = my_simulation_id
        # initialize the
        information_dump['cumulative_revenue_current_selling_season'] = 0
        return price, information_dump

        
    if not first_day:
        # do the book keeping (unless we are on the first day)
        remaining_capacity = 80 - np.sum(demand)
        revenue = demand[-1] * prices[0, -1]
        my_simulation_id = information_dump['current_simulation']
        rev_sofar_current_selling_season = information_dump['cumulative_revenue_current_selling_season']
        
        # Create new row as a DataFrame and concatenate
        new_row = pd.DataFrame([{
            'simulation': my_simulation_id,
            'day': day - 1,
            'season': season_previous_period,
            'demand': demand[-1],
            'own_price': prices[0, -1],
            'competitor_price': prices[1, -1],
            'remaining_capacity': remaining_capacity,
            'revenue': revenue,
            'cumulative_revenue': rev_sofar_current_selling_season + revenue,
        }])
        
        # Use pd.concat instead of append
        information_dump['history'] = pd.concat(
            [information_dump['history'], new_row],
            ignore_index=True
        )
        
        information_dump['cumulative_revenue_current_selling_season'] = rev_sofar_current_selling_season + revenue

    ## Save information dump to disk (duopoly_feedback.data) in selling period 100 & 101
    # (Note, 101 is only happening under the hood in the simulation to obtain the demand of selling period 100.)
    if selling_period_in_current_season >= 100:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return price, information_dump


def _initialize_data_feedback():
    """Initialize the feedback data object"""

    ## Try to load duopoly feedback data from disk
    try:
        with open('duopoly_feedback.data', 'rb') as handle:
            feedback = pickle.load(handle)
        return feedback
    except:
        return {
            # keep track of historical observables
            'history': (
                pd.DataFrame({
                    'simulation': pd.Series(dtype='str'),
                    'day': pd.Series(dtype='int'),
                    'season': pd.Series(dtype='int'),
                    'demand': pd.Series(dtype='float'),
                    'own_price': pd.Series(dtype='float'),
                    'competitor_price': pd.Series(dtype='float'),
                    'remaining_capacity': pd.Series(dtype='float'),
                    'revenue': pd.Series(dtype='float'),
                    'cumulative_revenue': pd.Series(dtype='float'),
                })
            ),
            'current_simulation': '',
            'cumulative_revenue_current_selling_season': 0
        }