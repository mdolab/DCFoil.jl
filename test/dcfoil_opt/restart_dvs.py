# --- Python 3.10 ---
"""
@File          :   restart_dvs.py
@Date created  :   2024/12/31
@Last modified :   2024/12/31
@Author        :   Galen Ng
@Desc          :   Store restart variables
"""

from collections import OrderedDict
from numpy import array

dv_dict = {}

# Optimization of just the wing with a CL, tip bending constraint for single point
dv_dict["2025-01-06_mothrudder_opt_elevator_wtipcon1pt"] = {
    OrderedDict(
        [
            ("sweep", array([21.24133939])),
            ("alfa0", array([0.92362402])),
            ("theta_f", array([0.0])),
        ]
    )
}
