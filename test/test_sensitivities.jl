"""
Test derivative routines with super basic tests
"""

include("../src/solvers/SolverRoutines.jl")
using .SolverRoutines
include("../src/hydro/Hydro.jl")
using .Hydro
using FiniteDifferences, ForwardDiff, Zygote
using Plots, LaTeXStrings, Printf

function test_jacobian()
    """
    Test the Jacobian construction using values from

    'Engineering Design Optimization' by Martins and Ning 2021
    Example 6.11 Differentiating an implicit function

    Natural frequency of a beam is
        f = λm²
    and λ is related to m through
        λ/m + cos(λ) = 0.
    we want
        df/dm
    so
        ∂f/∂x = ∂f/∂m = 2λm
        ∂r/∂x = ∂r/∂m = -λ/m²
        ∂f/∂u = ∂f/∂λ = m²
        ∂r/∂u = ∂r/∂λ = 1/m - sin(λ)
    and the final answer is
        df/dm = 2λm - λ / (1/m - sin(λ))

    For this test, to make it a system, we try two cases so
        f₁, x₁, u₁
        f₂, x₂, u₂
    """
    # ************************************************
    #     Reference values
    # ************************************************


    # ************************************************
    #     Call our routine
    # ************************************************
    # ---------------------------
    #   Some inputs
    # ---------------------------
    # DVs (x)
    m = [1.0, 2.0]
    # States (u)
    λ = [1.0, 2.0]

    # ---------------------------
    #   Partials
    # ---------------------------
    evalFuncs = ["f1", "f2"]
    # Build up partials
    ∂f∂x = Dict(
        "f1" => 2 * λ[1] * m[1],
        "f2" => 2 * λ[2] * m[2],
    )
    ∂r∂x = Dict(
        "f1" => -λ[1] / m[1]^2,
        "f2" => -λ[2] / m[2]^2,
    )
    ∂f∂u = Dict(
        "f1" => m[1]^2,
        "f2" => m[2]^2,
    )
    ∂r∂u = Dict(
        "f1" => 1 / m[1] - sin(λ[1]),
        "f2" => 1 / m[2] - sin(λ[2]),
    )
    partials = Dict(
        "∂f∂x" => Dict(),
        "∂r∂x" => Dict(),
        "∂f∂u" => Dict(),
        "∂r∂u" => Dict(),
        "ψ" => Dict(),
        "ϕ" => Dict(),
    )
    for func in evalFuncs
        partials["∂f∂x"][func] = ∂f∂x[func]
        partials["∂r∂x"][func] = ∂r∂x[func]
        partials["∂f∂u"][func] = ∂f∂u[func]
        partials["∂r∂u"][func] = ∂r∂u[func]
        partials["ψ"][func] = transpose(∂f∂u[func]) / transpose(∂r∂u[func])
        partials["ϕ"][func] = ∂r∂x[func] / ∂r∂u[func]
    end
    methods = ["adjoint", "direct"]
    for method in methods
        funcsSens = SolverRoutines.compute_jacobian(partials, evalFuncs; method=method)

    end

    return
end

function test_eigenvalueAD()
    """
    Dot product test!
    """

    # --- A test matrix ---
    A_r = [2.0 7.0; 1.0 8.0]
    A_i = [1.0 1.0; 0.0 1.0]

    # ---------------------------
    #   forward AD
    # ---------------------------
    dim = 2
    A_rd = zeros(Float64, dim, dim)
    A_id = zeros(Float64, dim, dim)
    # A_rd[2, 1] = 1.0
    A_rd .= 1.0 # poke all entries in matrix in forward
    # TODO: try other seed
    A_id .= 1.0
    # A_id[1, 1] = 1.0
    w_r, w_rd, w_i, w_id, VR_r, VR_rd, VR_i, VR_id = SolverRoutines.cmplxStdEigValProb_d(A_r, A_rd, A_i, A_id, dim)
    # println("Primal forward values:")
    # println("w_r = ", w_r)
    # println("w_i = ", w_i)
    # println("VR_r", VR_r)
    # println("VR_i", VR_i)
    # println("Dual forward values:")
    # println("w_rd = ", w_rd)
    # println("w_id = ", w_id)
    # println("VR_rd", VR_rd)
    # println("VR_id", VR_id)
    # ---------------------------
    #   backward AD
    # ---------------------------
    w_rb = zeros(Float64, dim)
    w_ib = zeros(Float64, dim)
    w_rb = [1, 1] # poke both eigenvalues in reverse
    w_rb = w_rd
    w_ib = [1, 1]
    Vrb_r = zeros(Float64, dim, dim)
    Vrb_i = zeros(Float64, dim, dim)
    A_rb, A_ib, w_r, w_rbz, w_i, w_ibz, _, _, _, _ = SolverRoutines.cmplxStdEigValProb_b(A_r, A_i, dim, w_rb, w_ib, Vrb_r, Vrb_i)
    # println("Primal reverse values:")
    # println("w_r = ", w_r)
    # println("w_i = ", w_i)
    # # println("VR_r", VR_r)
    # # println("VR_i", VR_i)
    # println("Dual reverse values:")
    # # println("wb_r = ", w_rb)
    # # println("wb_i = ", w_ib)
    # println("A_rb", A_rb)
    # println("A_ib", A_ib)

    # ---------------------------
    #   Dot product test
    # ---------------------------
    # --- Outputs ---
    ḟ = w_rd
    f̄ = w_rb
    # --- Inputs ---
    # The inputs were matrices so we just unroll them
    ẋ = vec(A_rd)
    x̄ = vec(A_rb)
    # --- Dot product ---
    lhs = (transpose(ẋ) * x̄)
    rhs = (transpose(ḟ) * f̄)
    # These should be equal if you did it right
    return lhs - rhs
end

function test_theodorsenDeriv()
    """
    Test the theodorsen functions
    """
    nNodes = 3 # Number of spatial nodes
    chordVec = vcat(LinRange(0.81, 0.405, nNodes))
    # ---------------------------
    #   Test glauert lift distribution
    # ---------------------------
    cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=1.0, nNodes=nNodes)
    pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
    plot!(title="lift slope")

    # ---------------------------
    #   Test 𝙲(k)
    # ---------------------------
    k0 = 1e-12
    kSweep = k0:0.001:2

    datar = []
    datai = []
    datar2 = []
    datai2 = []
    datar3 = []
    datai3 = []
    dADr = []
    dADi = []
    dFDr = []
    dFDi = []
    dRD_r = []
    dRD_i = []
    for k ∈ kSweep
        datum = Hydro.compute_theodorsen(k)
        datum2 = Hydro.compute_pade(k)
        datum3 = Hydro.compute_fraccalc(k)
        push!(datar, datum[1])
        push!(datai, datum[2])
        push!(datar2, datum2[1])
        push!(datai2, datum2[2])
        push!(datar3, datum3[1])
        push!(datai3, datum3[2])
        # derivAD = ForwardDiff.derivative(Hydro.compute_theodorsen, k)
        # derivFD = FiniteDifferences.forward_fdm(2, 1)(Hydro.compute_theodorsen, k)
        derivAD, = Zygote.jacobian(Hydro.compute_pade, k)
        derivFD, = Zygote.jacobian(Hydro.compute_fraccalc, k)
        derivRD, = Zygote.jacobian(Hydro.compute_theodorsen, k)
        push!(dADr, derivAD[1])
        push!(dADi, derivAD[2])
        push!(dFDr, derivFD[1])
        push!(dFDi, derivFD[2])
        push!(dRD_r, derivRD[1])
        push!(dRD_i, derivRD[2])
    end

    # # --- Derivatives ---
    # dADr
    # println("Forward AD:", ForwardDiff.derivative(Hydro.compute_theodorsen, 0.1))
    # println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(Hydro.compute_theodorsen, 0.1))
    # println("Reverse AD:", Zygote.jacobian(Hydro.compute_theodorsen, 0.1))

    # ************************************************
    #     Plot data
    # ************************************************
    lw = 2.0
    la = 0.5
    # --- Plot primal ---
    # Solid lines
    p1 = plot(kSweep, datar, label="Re", tick_dir=:out, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, datai, label="Im", color=:blue, linewidth=lw, linealpha=la)
    # Dashed lines
    plot!(kSweep, datar2, label="Re Pade-3", line=:dash, color=:red, linewidth=lw, linealpha=la, tick_dir=:out)
    plot!(kSweep, datai2, label="Im Pade-3", line=:dash, color=:blue, linewidth=lw, linealpha=la)
    # Dotted lines
    plot!(kSweep, datar3, label="Re FracCalc", line=:dot, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, datai3, label="Im FracCalc", line=:dot, color=:blue, linewidth=lw, linealpha=la)

    plot!(title="Theodorsen function")
    plot!(xlabel=L"k", ylabel=L"C(k)")

    # --- Plot derivatives ---
    p2 = plot(kSweep, dRD_r, label="Real RD", color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, dRD_i, label="Imag RD", color=:blue, linewidth=lw, linealpha=la)

    # p2 = plot(kSweep, dADr, label="Re FAD", tick_dir=:out, color=:red, linewidth=lw, linealpha=la)
    # plot!(kSweep, dADi, label="Im FAD", color=:blue, linewidth=lw, linealpha=la)
    plot!(kSweep, dADr, label="Re Pade-3", tick_dir=:out, color=:red, linewidth=lw, linealpha=la,line=:dash)
    plot!(kSweep, dADi, label="Im Pade-3", color=:blue, linewidth=lw, linealpha=la,line=:dash)

    plot!(kSweep, dFDr, label="Re FracCalc", line=:dot, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, dFDi, label="Im FracCalc", line=:dot, color=:blue, linewidth=lw, linealpha=la)


    ylims!(-10, 2.0)
    xlims!(-0.1, 1.1)
    plot!(title="RAD wrt k")
    xlabel!(L"k")
    ylabel!(L"\partial C(k)/ \partial k")
    plot(p1, p2)

    savefig("theodorsen.png")
end
