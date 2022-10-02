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

# ************************************************
#     I/O 
# ************************************************
dataDir = "./OUTPUT/testAero/"
outputDir = dataDir

# ************************************************
#     Read in results
# ************************************************
# --- Read bending ---
file = readlines(dataDir * "bending.dat")
bending = zeros(length(file))
nodes = 0:length(bending)-1

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
plot(
    [nodes, nodes, nodes, nodes], [bending, twisting, lift, moment],
    label=["" "" "" ""],
    layout=(2, 2),
    title=["Spanwise Bending" "Spanwise twist" "Lift" "Moment"],
    xlabel="node #",
    ylabel=["w [m]" "psi [rad]" "L [N/m]" "M [N-m/m]"],
)

savefig(outputDir * "spanwise_view.pdf")