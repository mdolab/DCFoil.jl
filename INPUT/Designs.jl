"""
This file is just meant to store DV dictionaries of designs we analyze in the paper
"""

# --- Foil from Deniz Akcabay's 2020 paper ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(neval), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 15 * π / 180, # fiber angle global [rad]
)

# --- Blake & Maga's cantilever strut in water (1975) ---
# Table 1: 2.75 in x 20 in strut
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 0.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "ss", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 2.75 * 2.54 / 100 * ones(neval), # chord length [m]
    "s" => 20 * 2.54 / 100, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 10 * π / 180, # fiber angle global [rad]
)
# --- Yingqian's Sweep & Anisotropy Paper (2018) ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.81 * ones(neval), # chord length [m]
    "s" => 2.7, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 30 * π / 180, # fiber angle global [rad]
)

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.0925 * ones(neval), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.03459, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)


# --- Dummy test with 1's ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "test", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 1 * ones(neval), # chord length [m]
    "s" => 1, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 1, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)