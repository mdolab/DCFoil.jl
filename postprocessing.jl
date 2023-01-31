# --- Julia 1.8---
"""
@File    :   postprocessing.jl
@Time    :   2022/09/04
@Author  :   Galen Ng
@Desc    :   Plotting script used to visualize data

NOTE: Sometimes the python it calls is the wrong version so check with:
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
"""

# --- Import libs ---
# using PyPlot
using Plots
using JSON
using Printf
using LaTeXStrings
# ************************************************
#     I/O 
# ************************************************
dataDir = "./OUTPUT/testAero/"
outputDir = dataDir

# ************************************************
#     Read in results
# ************************************************
# --- Read in DVDict ---
DVDict::Dict = Dict()
open(dataDir * "init_DVDict.json", "r") do f
    global DVDict
    DVDict = JSON.parse(f)
end
# --- Read in funcs ---
funcs::Dict = Dict()
open(dataDir * "funcs.json", "r") do f
    global funcs
    funcs = JSON.parse(f)
end

# ==============================================================================
#                         Static hydroelastic
# ==============================================================================
is_static = true
if is_static
    # ************************************************
    #     Read in data
    # ************************************************
    # --- Read bending ---
    file = readlines(dataDir * "bending.dat")
    bending = zeros(length(file))
    nodes = LinRange(0, DVDict["s"], length(bending))

    counter = 1
    for line ∈ file
        bending[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read twisting ---
    file = readlines(dataDir * "twisting.dat")
    twisting = zeros(length(file))

    counter = 1
    for line ∈ file
        twisting[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read lift ---
    file = readlines(dataDir * "lift.dat")
    lift = zeros(length(file))
    counter = 1
    for line ∈ file
        lift[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read moment ---
    file = readlines(dataDir * "moments.dat")
    moment = zeros(length(file))
    counter = 1
    for line ∈ file
        moment[counter] = parse(Float64, line)
        counter += 1
    end

end
# ==============================================================================
#                         Dynamic hydroelastic
# ==============================================================================
is_dynamic = true
if is_dynamic
    # --- Read frequencies ---
    file = readlines(dataDir * "FreqSweep.dat")
    freqs = zeros(length(file))
    counter = 1
    for line ∈ file
        freqs[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read tip bending ---
    file = readlines(dataDir * "TipBendDyn.dat")
    dynTipBending = zeros(length(file))
    counter = 1
    for line ∈ file
        dynTipBending[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read tip twisting ---
    file = readlines(dataDir * "TipTwistDyn.dat")
    dynTipTwisting = zeros(length(file))
    counter = 1
    for line ∈ file
        dynTipTwisting[counter] = parse(Float64, line) * 180 / π # CONVERT TO DEGREES
        counter += 1
    end

    # --- Read tip lift ---
    file = readlines(dataDir * "TipLiftDyn.dat")
    dynTipLift = zeros(length(file))
    counter = 1
    for line ∈ file
        dynTipLift[counter] = parse(Float64, line)
        counter += 1
    end

    # --- Read tip moment ---
    file = readlines(dataDir * "TipMomentDyn.dat")
    dynTipMoment = zeros(length(file))
    counter = 1
    for line ∈ file
        dynTipMoment[counter] = parse(Float64, line)
        counter += 1
    end
end


# ==============================================================================
#                         Flutter solution
# ==============================================================================
is_flutter = true
if is_flutter
    # --- Read in data ---
end


# ==============================================================================
#                         Plot results
# ==============================================================================
if is_static
    liftTitle = @sprintf("Lift (%.1fN, CL=%.2f)", (funcs["lift"]), funcs["cl"])
    momTitle = @sprintf("Mom. (%.1fN-m, CM=%.2f)", (funcs["moment"]), funcs["cmy"])
    visuals = plot(
        [nodes, nodes, nodes, nodes], [bending, twisting * 180 / π, lift, moment],
        label=["" "" "" ""],
        layout=(2, 2),
        title=["Spanwise Bending" "Spanwise twist" liftTitle momTitle],
        xlabel="y [m]",
        ylabel=[L"w" * " [m]" L"\psi" * " " * L"[\circ]" L"L" * " [N/m]" L"M" * " [N-m/m]"],
    )

    fiberAngle = round(DVDict["θ"] * 180 / π; digits=2)
    flowSpeed = round(DVDict["U∞"]; digits=2)
    AOA = round(DVDict["α₀"]; digits=2)
    sweepAngle = round(DVDict["Λ"] * 180 / π; digits=2)

    titleTxt = L"U_{\infty} = %$flowSpeed \textrm{\,m/s}, α_0 = %$AOA^{\circ}, \Lambda = %$sweepAngle^{\circ}, θ_f = %$fiberAngle^{\circ}"
    title = plot(title=titleTxt, grid=false, xticks=false, yticks=false, showaxis=false, bottom_margin=-50Plots.px)
    plot(title, visuals, layout=@layout([A{0.1h}; B]))

    savefig(outputDir * "spanwise_view.pdf")
end

if is_dynamic
    visuals = plot(
        [freqs, freqs],
        [dynTipBending, dynTipTwisting],
        label=["" ""],
        layout=(2, 2),
        xlabel="Frequency [Hz]",
        ylabel=[L"w_{\textrm{tip}}" * " [m]" L"\psi_{\textrm{tip}}" * " " * L"[^{\circ}]" "test" "test"],
    )
    titleTxt = L"U_{\infty} = %$flowSpeed \textrm{\,m/s}, α_0 = %$AOA^{\circ}, \Lambda = %$sweepAngle^{\circ}, θ_f = %$fiberAngle^{\circ}"
    title = plot(title=titleTxt, grid=false, xticks=false, yticks=false, showaxis=false, bottom_margin=-50Plots.px)
    plot(title, visuals, layout=@layout([A{0.1h}; B]))

    savefig(outputDir * "tip_dynamics.pdf")
end
