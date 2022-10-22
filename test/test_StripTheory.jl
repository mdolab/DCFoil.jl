"""
unit test to verify the strip theory equations
"""

include("../src/Hydro.jl")

using LinearAlgebra
using .Hydro

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