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