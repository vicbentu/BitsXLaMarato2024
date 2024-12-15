import os

import pandas as pd
import netCDF4 as nc
from scipy.optimize import minimize

from models import *

def predict(date, lat, long):
    date = pd.to_datetime(date)
    
    alpha = None, mode = None

    result_caliope, variance_caliope = caliope_model_victor_predict(date, lat, long)

    result_road = road_model_predict(date,lat, mode)


    



    return weighted_average

def objective_function(parameters):
    for p in parameters:

        objective_value
        our_value




result = minimize(objective_function, [parameters], method='Nelder-Mead')