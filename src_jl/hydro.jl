# --- Julia ---

# @File    :   hydro.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines
# TODO: declare data types for performance improvements

module Hydro

# --- Public functions ---
export 𝙲, compute_glauert_circ

using SpecialFunctions
using LinearAlgebra

function 𝙲(k)
    """
    Theodorsen's transfer function for unsteady aero/hydrodynamics 
    w/ separate real and imaginary parts. This is potential flow theory.

    Inputs:
        k: float, reduced frequency of oscillation (a.k.a. Strouhal number)

    Unicode character: \ttC 

    NOTE:
    Undefined for k = ωb/Ucos(Λ) = 0 (steady aero)
    """
    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    denom = ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    𝙲ᵣ = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) / denom
    𝙲ᵢ = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) / denom

    ans = [𝙲ᵣ, 𝙲ᵢ]

    return ans
end

function compute_glauert_circ(semispan, chord, α₀, U∞, neval)
    """
    Glauert's solution for the lift slope on a 3D hydrofoil

    The coordinate system is

    clamped root                         free tip
    `+-----------------------------------------+  (x=0 @ LE)
    `|                                         |
    `|               +-->y                     |
    `|               |                         |
    `|             x v                         |
    `+-----------------------------------------+
    `
    (y=0 @ root)

    where z is out of the page (thickness dir.)

    returns:
        cl_α : array, shape (neval,)
            sectional lift slopes for a 3D wing [rad⁻¹] 
            sometimes denoted in literature as 'a₀'
    NOTE:
    This follows the formulation in 
    'Principles of Naval Architecture Series (PNA) - Propulsion 2010' 
    by Justin Kerwin & Jacques Hadler
    """

    ỹ = π / 2 * ((1:1:neval) / neval) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
    y = -semispan * cos.(ỹ) # the physical coordinate (y) is only calculated to the root (-semispan, 0)

    # ---------------------------
    #   PLANFORM SHAPES: rectangular is outdated
    # ---------------------------
    # # --- Rectangular ---
    # chordₚ = chord
    # --- Elliptical planform ---
    chordₚ = chord .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?

    n = (1:1:neval) * 2 - ones(neval) # node numbers x2 (node multipliers)

    b = π / 4 * (chordₚ / semispan) * α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, neval]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = π / 4 * chordₚ / semispan .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:neval*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) #matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the coefficients in Glauert's Fourier series ---
    ã = chord11 \ b

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise free vortex strength (Γ/semispan)

    cl = (2 * γ) / (U∞ * chord) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl / (α₀ + 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    # TODO:interpolate load now
    cl_α = interp1(y, clα, ỹ)

    return cl_α
end

end

# ==============================================================================
# Grunt numerical methods
# ==============================================================================
using Interpolations

function interp1(xpt, ypt, x; method="linear", extrapvalue=nothing)

    if extrapvalue == nothing
        y = zeros(x)
        idx = trues(x)
    else
        y = extrapvalue * ones(x)
        idx = (x .>= xpt[1]) .& (x .<= xpt[end])
    end

    if method == "linear"
        intf = interpolate((xpt,), ypt, Gridded(Linear()))
        y[idx] = intf[x[idx]]

    elseif method == "cubic"
        itp = interpolate(ypt, BSpline(Cubic(Natural())), OnGrid())
        intf = scale(itp, xpt)
        y[idx] = [intf[xi] for xi in x[idx]]
    end

    return y
end
# ==============================================================================
# Tests for this module
# ==============================================================================
# --- Unit tests ---
using ForwardDiff, ReverseDiff, FiniteDifferences
using Plots, LaTeXStrings

hydro.compute_glauert_circ(2.7, LinRange(0.81, 0.405, 250), 6, 1, 250)

function unit_test(makePlots=false)
    """
    Run unit tests on all the functions in this module file
    """

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

end