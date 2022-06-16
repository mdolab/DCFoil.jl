# ==============================================================================
# Run tests on hydro module file
# ==============================================================================

using ForwardDiff, ReverseDiff, FiniteDifferences
using Plots, LaTeXStrings
# Using the Hydro module
using .Hydro

# ---------------------------
#   Test glauert lift distribution
# ---------------------------
cl_α = Hydro.compute_glauert_circ(semispan=2.7, chord=LinRange(0.81, 0.405, 250), α₀=6, U∞=1, neval=250)
pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
plot!(title="lift slope",ylims=(0,0.03))
# ---------------------------
#   Test 𝙲(k)
# ---------------------------
kSweep = 0.01:0.01:2

datar = []
datai = []
dADr = []
dADi = []
dFDr = []
dFDi = []
for k ∈ kSweep
    datum = unsteadyHydro.𝙲(k)
    push!(datar, datum[1])
    push!(datai, datum[2])
    derivAD = ForwardDiff.derivative(unsteadyHydro.𝙲, k)
    derivFD = FiniteDifferences.forward_fdm(2, 1)(unsteadyHydro.𝙲, k)
    push!(dADr, derivAD[1])
    push!(dADi, derivAD[2])
    push!(dFDr, derivFD[1])
    push!(dFDi, derivFD[2])
end

# --- Derivatives ---
dADr
println("Forward AD:", ForwardDiff.derivative(unsteadyHydro.𝙲, 0.1))
println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(unsteadyHydro.𝙲, 0.1))

# --- Plot ---
if makePlots
    p1 = plot(kSweep, datar, label="Real")
    plot!(kSweep, datai, label="Imag")
    plot!(title="Theodorsen function")
    xlabel(L"k")
    ylabel!(L"C(k)")
    p2 = plot(kSweep, dADr, label="Real FAD")
    plot!(kSweep, dFDr, label="Real FD", line=:dash)
    plot!(kSweep, dADi, label="Imag FAD")
    plot!(kSweep, dFDi, label="Imag FD", line=:dash)
    plot!(title="Derivatives wrt k")
    xlabel!(L"k")
    ylabel!(L"\partial C(k)/ \partial k")

    plot(p1, p2)
end


