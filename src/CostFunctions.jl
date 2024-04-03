# --- Julia 1.9---
"""
@File    :   costFunctions.jl
@Time    :   2024/04/02
@Author  :   Galen Ng
@Desc    :   Store all possible cost funcs
"""

module CostFunctions

# --- Solver cost funcs ---
staticCostFuncs = [
    "psitip"
    "wtip"
    "lift"
    "moment"
    "cl"
    "cmy"
    # Centers
    "cofz"
    "comy"
    # Lift-induced drag
    "cdi"
    "fxi"
    # Junction drag (empirical relation interference)
    "cdj"
    "fxj"
    # Spray drag
    "cds"
    "fxs"
    # Profile drag
    "cdpr"
    "fxpr"
]
forcedCostFuncs = [
    "peakpsitip" # maximum deformation amplitude (abs val) across forced frequency sweep
    "peakwtip"
    "vibareapsi" # integrated deformations under the spectral curve (see Ng et al. 2022)
    "vibareaw"
]
flutterCostFuncs = [
    "ksflutter" # flutter value (damping)
    "lockin" # lock-in value
    "gap" # mode gap width
]

end