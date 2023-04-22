"""
Test derivative routines with super basic tests
"""

include("../src/solvers/SolverRoutines.jl")
using .SolverRoutines

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
    A_i = [1.0 0.0; 0.0 1.0]

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
    w_rb = w_rd # TODO: try this
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