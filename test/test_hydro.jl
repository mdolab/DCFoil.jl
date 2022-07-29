# ==============================================================================
# Run tests on hydro module file
# ==============================================================================

using ForwardDiff, ReverseDiff, FiniteDifferences
using Plots, LaTeXStrings

include("../src/Hydro.jl")
using .Hydro # Using the Hydro module

neval = 3 # Number of spatial nodes
chordVec = vcat(LinRange(0.81, 0.405, neval))
# ---------------------------
#   Test glauert lift distribution
# ---------------------------
cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6, U∞=1.0, neval=neval)
pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
plot!(title="lift slope")

# ---------------------------
#   Test added mass
# ---------------------------
Hydro.compute_added_mass(ρ_f=1025.0, chordVec=chordVec)

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
    datum = Hydro.compute_theodorsen(k)
    push!(datar, datum[1])
    push!(datai, datum[2])
    derivAD = ForwardDiff.derivative(Hydro.compute_theodorsen, k)
    derivFD = FiniteDifferences.forward_fdm(2, 1)(Hydro.compute_theodorsen, k)
    push!(dADr, derivAD[1])
    push!(dADi, derivAD[2])
    push!(dFDr, derivFD[1])
    push!(dFDi, derivFD[2])
end

# --- Derivatives ---
dADr
println("Forward AD:", ForwardDiff.derivative(Hydro.compute_theodorsen, 0.1))
println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(Hydro.compute_theodorsen, 0.1))

# --- Plot ---
p1 = plot(kSweep, datar, label="Real")
plot!(kSweep, datai, label="Imag")
plot!(title="Theodorsen function")
plot!(xlabel=L"k", ylabel=L"C(k)")
p2 = plot(kSweep, dADr, label="Real FAD")
plot!(kSweep, dFDr, label="Real FD", line=:dash)
plot!(kSweep, dADi, label="Imag FAD")
plot!(kSweep, dFDi, label="Imag FD", line=:dash)
plot!(title="Derivatives wrt k")
xlabel!(L"k")
ylabel!(L"\partial C(k)/ \partial k")

plot(p1, p2)


