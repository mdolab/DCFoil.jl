# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle

using Printf # for better file name
using JLD
include("src/DCFoil.jl")

using .DCFoil

# ==============================================================================
# Setup hydrofoil model and solver settings
# ==============================================================================
# ************************************************
#     Task type
# ************************************************
# Set task you want to true
# Defaults
run = true # run the solver for a single point
run_static = false
run_forced = false
run_modal = false
run_flutter = false
debug = false
tipMass = false

# Uncomment here
run_static = true
# run_forced = true
run_modal = true
run_flutter = true
debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 20 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
df = 1
fSweep = 0.1:df:1000.0 # forcing and search frequency sweep [Hz]
# uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

DVDict = Dict(
    "name" => "akcabay-swept",
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

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
    DVDict["name"],
    DVDict["material"],
    rad2deg(DVDict["θ"]),
    rad2deg(DVDict["Λ"]))
mkpath(outputDir)

# ************************************************
#     Set solver options
# ************************************************
solverOptions = Dict(
    # --- I/O ---
    "debug" => debug,
    "outputDir" => outputDir,
    # --- General solver options ---
    "config" => "wing",
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "tipMass" => tipMass,
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fSweep" => fSweep,
    "tipForceMag" => tipForceMag,
    # --- Eigen solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 4000,
    "rhoKS" => 80.0,
)
# ==============================================================================
#                         Call DCFoil
# ==============================================================================
steps = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12] # step sizes
dvKey = "θ" # dv to test deriv

# ************************************************
#     Forward difference checks
# ************************************************
derivs = zeros(length(steps))
for (ii, dh) in enumerate(steps)
    costFuncs = DCFoil.run_model(
        DVDict,
        evalFuncs;
        solverOptions=solverOptions
    )
    flutt_i = costFuncs["ksflutter"]
    DVDict[dvKey] += dh
    costFuncs = DCFoil.run_model(
        DVDict,
        evalFuncs;
        # --- Optional args ---
        solverOptions=solverOptions
    )
    flutt_f = costFuncs["ksflutter"]

    derivs[ii] = (flutt_f - flutt_i) / dh
    @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

    # --- Reset DV ---
    DVDict[dvKey] -= dh
end

save("FWDDiff.jld", "data", derivs)

# # ************************************************
# #     Complex step checks
# # ************************************************
# derivs = zeros(length(steps))
# for (ii, dh) in enumerate(steps)
#     # costFuncs = DCFoil.run_model(
#     #     DVDict,
#     #     evalFuncs;
#     #     solverOptions=solverOptions
#     # )
#     # flutt_i = costFuncs["ksflutter"]
#     DVDict[dvKey] += 1im * dh

#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         # --- Optional args ---
#         solverOptions=solverOptions
#     )
#     flutt_f = costFuncs["ksflutter"]

#     derivs[ii] = Imag(flutt_f) / dh
#     @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

#     # --- Reset DV ---
#     DVDict[dvKey] -= 1im * dh
# end

# save("CStep.jld", "data", derivs)