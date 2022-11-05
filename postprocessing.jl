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
DVDict = Dict()
open(dataDir * "init_DVDict.json", "r") do f
    global DVDict
    DVDict = JSON.parse(f)
end
# --- Read in funcs ---
funcs = Dict()
open(dataDir * "funcs.json", "r") do f
    global funcs
    funcs = JSON.parse(f)
end

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

# ************************************************
#     Plot results
# ************************************************
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

titleTxt = L"V =" * string(DVDict["U∞"]) * "m/s, α₀ = " * string(DVDict["α₀"]) * "deg, Λ = " * string(DVDict["Λ"] * 180 / π) * L" \circ, θ_f = " * string(DVDict["θ"] * 180 / π) * L"\circ"
title = plot(title=titleTxt, grid=false, xticks=false, yticks=false, showaxis=false, bottom_margin=-50Plots.px)
plot(title, visuals, layout=@layout([A{0.1h}; B]))

savefig(outputDir * "spanwise_view.pdf")