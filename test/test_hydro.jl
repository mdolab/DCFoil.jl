"""
Run tests on hydro module file
"""

using ForwardDiff, ReverseDiff, FiniteDifferences
using Plots, LaTeXStrings

include("../src/hydro/Hydro.jl")
using .Hydro # Using the Hydro module

function test_stiffness()
    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0
    U = 5
    Λ = 0
    ω = 0.1
    ρ = 1000
    Matrix = Hydro.compute_node_stiff(clα, b, eb, ab, U, Λ, ω, ρ)
    show(stdout, "text/plain", Matrix)
    # show(stdout, "text/plain", imag(Matrix))
end

function test_damping()
    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0
    U = 5
    Λ = 0
    ω = 0.1
    ρ = 1000
    Matrix = Hydro.compute_node_damp(clα, b, eb, ab, U, Λ, ω, ρ)
    show(stdout, "text/plain", real(Matrix))
    show(stdout, "text/plain", imag(Matrix))
end

function test_mass()
    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0
    U = 5
    Λ = 0
    ω = 0.1
    ρ = 1000
    Matrix = Hydro.compute_node_mass(b, ab, ω, ρ)
    show(stdout, "text/plain", real(Matrix))
    show(stdout, "text/plain", imag(Matrix))
end


neval = 3 # Number of spatial nodes
chordVec = vcat(LinRange(0.81, 0.405, neval))
# ---------------------------
#   Test glauert lift distribution
# ---------------------------
cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6, U∞=1.0, neval=neval)
pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
plot!(title="lift slope")

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


