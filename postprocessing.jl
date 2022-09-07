# --- Julia 1.8---
"""
@File    :   postprocessing.jl
@Time    :   2022/09/04
@Author  :   Galen Ng
@Desc    :   Plotting script used to visualize data

NOTE: Sometimes the python it calls is the wrong version so check with:
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
"""


# using PyPlot
using Plots

x = range(0, 1, 100)
y = cos.(x)

plot(x, y, title="Test Lines")


bending = zeros(length(readlines("bending.dat")))
nodes = 0:length(bending)-1

counter = 1
for line ∈ readlines("bending.dat")
    bending[counter] = parse(Float64, line)
    counter += 1
end

twisting = zeros(length(readlines("twisting.dat")))

counter = 1
for line ∈ readlines("twisting.dat")
    twisting[counter] = parse(Float64, line)
    counter += 1
end


plot([nodes, nodes], [bending, twisting], layout=2, title=["Spanwise Bending" "Spanwise twist"], xlabel="node #", ylabel=["w [m]" "psi [rad]"])
savefig("deformations.pdf")