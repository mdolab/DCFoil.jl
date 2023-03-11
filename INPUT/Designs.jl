"""
This file is just meant to store DV dictionaries of designs we analyze in the paper
"""

# --- Foil from Deniz Akcabay's 2020 paper ---
# THIS COMPARES WELL WITH RESULTS FROM THE PAPER
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => deg2rad(-15.0), # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(15), # fiber angle global [rad]
)

# --- Blake & Maga's cantilever strut in water (1975) ---
# Not great agreement with paper, need to test
# Table 1: 2.75 in x 20 in strut
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 0.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "ss", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 2.75 * 2.54 / 100 * ones(nNodes), # chord length [m]
    "s" => 20 * 2.54 / 100, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# --- Yingqian's Sweep & Anisotropy Paper (2018) ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.81 * ones(nNodes), # chord length [m] THERE SHOULD BE TAPER
    "s" => 2.7, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 30 * π / 180, # fiber angle global [rad]
)

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.0925 * ones(nNodes), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.03459, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)


# --- Dummy test with 1's ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "test-iso", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 1 * ones(nNodes), # chord length [m]
    "s" => 1, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 1, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# --- Eirikur's flat plate ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 10.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1.2250, # fluid density [kg/m³]
    "material" => "eirikurPl", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.3 * ones(nNodes), # chord length [m]
    "s" => 0.85, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.00666666, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# ---------------------------
#   IMOCA 60
# ---------------------------
# --- IMOCA 60 bulb keel ---
DVDict = Dict(
    "name" => "IMOCA60Keel",
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 50.0 / 1.9438, # free stream velocity [m/s]
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "ρ_f" => 1025.0, # fluid density [kg/m³]
    # "material" => "ss", # preselect from material library
    # "toc" => 0.1, # thickness-to-chord ratio
    "material" => "cfrp", # preselect from material library
    "toc" => 0.15, # thickness-to-chord ratio
    "g" => 0.04, # structural damping percentage
    "c" => 0.65 * ones(nNodes), # chord length [m]
    "s" => 4.0, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(15), # fiber angle global [rad]
)
