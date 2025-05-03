"""
This file is just meant to store DV dictionaries of designs we analyze in the paper
"""

# ==============================================================================
#                         Deniz Akcabay's 2020 paper
# ==============================================================================
# THIS COMPARES WELL WITH RESULTS FROM THE PAPER
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 5.0, # free stream velocity [m/s]
    "sweep" => deg2rad(-15.0), # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
)

# Static divergence case
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 5.0, # free stream velocity [m/s]
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(-15), # fiber angle global [rad]
)

# --- Blake & Maga's cantilever strut in water (1975) ---
# Not great agreement with paper, need to test
# Table 1: 2.75 in x 20 in strut
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 0.0, # free stream velocity [m/s]
    "sweep" => 0.0 * π / 180, # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "ss", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 2.75 * 2.54 / 100 * ones(nNodes), # chord length [m]
    "s" => 20 * 2.54 / 100, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => 0 * π / 180, # fiber angle global [rad]
)

# --- AMC NACA0009 experimental model ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "material" => "al6061", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
)

# --- Yingqian's Sweep & Anisotropy Paper (2018) ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 5.0, # free stream velocity [m/s]
    "sweep" => 30.0 * π / 180, # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.81 * ones(nNodes), # chord length [m] THERE SHOULD BE TAPER
    "s" => 2.7, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => 30 * π / 180, # fiber angle global [rad]
)

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 5.0, # free stream velocity [m/s]
    "sweep" => 0.0 * π / 180, # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.0925 * ones(nNodes), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.03459, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => 0 * π / 180, # fiber angle global [rad]
)


# --- Dummy test with 1's ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 5.0, # free stream velocity [m/s]
    "sweep" => 30.0 * π / 180, # sweep angle [rad]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "test-iso", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 1 * ones(nNodes), # chord length [m]
    "s" => 1, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 1, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => 0 * π / 180, # fiber angle global [rad]
)

# --- Eirikur's flat plate ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "Uinf" => 10.0, # free stream velocity [m/s]
    "sweep" => 0.0 * π / 180, # sweep angle [rad]
    "rhof" => 1.2250, # fluid density [kg/m³]
    "material" => "eirikurPl", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.3 * ones(nNodes), # chord length [m]
    "s" => 0.85, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.00666666, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => 0 * π / 180, # fiber angle global [rad]
)

# ==============================================================================
#                         IMOCA60
# ==============================================================================
# --- IMOCA 60 bulb keel ---
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    # "toc" => 0.1, # thickness-to-chord ratio
    "toc" => 0.15, # thickness-to-chord ratio
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.65 * ones(nNodes), # chord length [m]
    "s" => 4.0, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
)

solverOptions = Dict(
    # --- I/O ---
    "name" => "IMOCA60",
    "debug" => debug,
    # --- General solver options ---
    "config" => "wing",
    "nNodes" => nNodes,
    "Uinf" => 5.0, # free stream velocity [m/s]
    "rhof" => 1025.0, # fluid density [kg/m³]
    "rotation" => 0.0, # deg
    "material" => "cfrp", # preselect from material library
    # "material" => "ss", # preselect from material library
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_tipMass" => tipMass,
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fRange" => fRange,
    "tipForceMag" => tipForceMag,
    # --- Eigen solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 500,
    "rhoKS" => 100.0,
)

# ==============================================================================
#                         AC75 AC37
# ==============================================================================
DVDict = Dict()
# mesh_dict = {
#             "num_x": 3,  # NOTE: Perform mesh convergence.
#             "num_y": 5,  # NOTE: Perform mesh convergence.
#             "span": 3.0,
#             "chord": 0.25,
#             "span_cos_spacing": 1.0,
#             "chord_cos_spacing": 1.0,
#             "wing_type": "rect",
#             "offset": np.array([0, 0, 0])
#         }

#         surface = {
#             "type": "aero",
#             "name": "rudder_strut",
#             "symmetry": True,
#             "S_ref_type": "projected",
#             "chord_cp": 0.25 * np.ones(3),
#             "mesh": mesh,
#             "CL0": 0.000000,
#             "CD0": 0.008880,  # At Re=3.0e6
#             "with_viscous": True,
#             "with_wave": False,
#             "groundplane": False,
#             "k_lam": 0.05,
#             # "t_over_c_cp": np.array([0.126]),
#             "t_over_c": 0.126,  # Eppler E836.
#             "c_max_t": 0.428,  # Eppler E836.
#         }

# ==============================================================================
#                         Moth T-foils
# ==============================================================================
# ************************************************
#     Rudder T-foil
# ************************************************
# Dimensions are from Yingqian
DVDictRudder = Dict(
    "alfa0" => 2.0, # initial angle of attack [deg] (base rake)
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    # "c" => 0.14 * ones(nNodes), # chord length [m]
    "c" => collect(LinRange(0.14, 0.095, nNodes)), # chord length [m]
    "s" => 0.333, # semispan [m]
    "ab" => 0.0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.075 * ones(nNodes), # thickness-to-chord ratio (mean)
    "x_ab" => 0.0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # [m]
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)


# ************************************************
#     Main T-foil (aka daggerboard)
# ************************************************
# Dimensions are from Day 2019
DVDictMain = Dict(
    "alfa0" => 2.0, # initial angle of attack [deg] (base rake)
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => collect(LinRange(0.125, 0.045, nNodes)), # chord length [m]
    "s" => 0.494, # semispan [m]
    "ab" => 0.0 * ones(Float64, nNodes), # dist from midchord to EA [m]
    "toc" => 0.128 * ones(Float64, nNodes), # thickness-to-chord ratio (max from paper)
    "x_ab" => 0.0 * ones(Float64, nNodes), # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "rake" => 0.0, # rake angle about top of strut [deg]
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # from Yingqian
    "c_strut" => 0.11 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.145 * ones(nNodesStrut), # thickness-to-chord ratio (max from paper)
    "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)

# ==============================================================================
#                         AM RUDDER T-FOIL
# ==============================================================================
DVDict = Dict(
    "alfa0" => 0.0, # initial angle of attack [deg]
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => ".dat", # chord length [m]
    "s" => 1.0, # semispan [m]
    "ab" => ".dat", # dist from midchord to EA [m]
    "toc" => ".dat", # thickness-to-chord ratio (mean)
    "x_ab" => ".dat", # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 2.8, # strut span [m]
    "c_strut" => ".dat", # chord length [m]
    "toc_strut" => ".dat", # thickness-to-chord ratio (mean)
    "ab_strut" => ".dat", # dist from midchord to EA [m]
    "x_ab_strut" => ".dat", # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)
solverOptions = Dict(
    # --- I/O ---
    "name" => "R3E6",
    "debug" => false,
    "writeTecplotSolution" => true,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => appendageList,
    "gravityVector" => [0.0, 0.0, -9.81],
    # ---------------------------
    #   Flow
    # ---------------------------
    "Uinf" => 18.0, # free stream velocity [m/s]
    # "Uinf" => 11.0, # free stream velocity [m/s]
    "rhof" => 1025.0, # fluid density [kg/m³]
    # "use_nlll" => false, # use non-linear lifting line code
    "use_nlll" => true, # use non-linear lifting line code
    "use_freeSurface" => true,
    "use_cavitation" => false,
    "use_ventilation" => false,
    "use_dwCorrection" => true,
    # ---------------------------
    #   Solver modes
    # ---------------------------
    # --- Static solve ---
    "run_static" => run_static,
    "res_jacobian" => "CS",
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fSweep" => fSweep,
    "tipForceMag" => tipForceMag,
    # --- p-k (Eigen) solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 100.0,
)