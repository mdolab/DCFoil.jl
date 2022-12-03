# --- Julia ---

# @File    :   hydro.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines
# TODO: declare data types for performance improvements

module Hydro
"""
Hydrodynamics module
"""
# --- Public functions ---
export compute_theodorsen, compute_glauert_circ
export compute_node_mass, compute_node_damp, compute_node_stiff
export compute_AICs!, apply_BCs

# --- Libraries ---
using FLOWMath: linear
using SpecialFunctions
using LinearAlgebra
using Plots
include("../solvers/SolverRoutines.jl")
using .SolverRoutines

function compute_theodorsen(k::Float64)
    """
    Theodorsen's transfer function for unsteady aero/hydrodynamics 
    w/ separate real and imaginary parts. This is potential flow theory.

    Inputs:
        k: float, reduced frequency of oscillation (a.k.a. Strouhal number)

    return:
        C(k)

    NOTE:
    Undefined for k = ωb/Ucos(Λ) = 0 (steady aero)
    """
    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ::Float64 = besselj0(k)
    H₀²ᵢ::Float64 = -bessely0(k)
    H₁²ᵣ::Float64 = besselj1(k)
    H₁²ᵢ::Float64 = -bessely1(k)

    denom::Float64 = ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    𝙲ᵣ::Float64 = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) / denom
    𝙲ᵢ::Float64 = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) / denom

    ans = [𝙲ᵣ, 𝙲ᵢ]

    return ans
end

function compute_glauert_circ(; semispan, chordVec, α₀, U∞, neval)
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
    inputs:
        α₀: float, angle of attack [rad]

    returns:
        cl_α : array, shape (neval,)
            sectional lift slopes for a 3D wing [rad⁻¹] starting from the root
            sometimes denoted in literature as 'a₀'

    NOTE:
    We use keyword arguments (denoted by the ';' to be more explicit)

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
    chordₚ = chordVec .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?

    n = (1:1:neval) * 2 - ones(neval) # node numbers x2 (node multipliers)

    b = π / 4 * (chordₚ / semispan) * α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, neval]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = π / 4 * chordₚ / semispan .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:neval*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) #matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the coefficients in Glauert's Fourier series ---
    ã = chord11 \ b

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise free vortex strength (Γ/semispan)

    cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl / (α₀ + 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    # pGlauert = plot(LinRange(0, 2.7, 250), clα)
    cl_α = linear(y, clα, LinRange(-semispan, 0, neval)) # Use BYUFLOW lab math function

    return reverse!(cl_α)
end

# function compute_added_mass(ρ_f, chordVec)
#     """
#     Compute the added mass for a rectangular cross section

#     return:
#         added mass, Array
#         added inertia, Array
#     """
#     mₐ = π * ρ_f * chordVec .* chordVec / 4 # Fluid-added mass vector [kg/m]
#     Iₐ = π * ρ_f * chordVec .^ 4 / 128 # Fluid-added inertia [kg-m]

#     return mₐ, Iₐ
# end

# ************************************************
#     Hydrodynamic strip forces
# ************************************************
function compute_node_stiff(clα, b, eb, ab, U∞, Λ, ω, rho_f, Ck)
    qf = 0.5 * rho_f * U∞^2 # Dynamic pressure

    # Aerodynamic quasi-steady stiffness 
    # (1st row is lift, 2nd row is pitching moment)

    k_hα = -2 * b * clα * Ck # lift due to angle of attack
    k_αα = -2 * eb * b * clα * Ck # moment due to angle of attack
    K_f = qf * cos(Λ)^2 *
          [
              0.0 k_hα
              0.0 k_αα
          ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = U∞ * cos(Λ) * 2 * clα * Ck
    e_hα = U∞ * cos(Λ) * (-clα) * b * (1 - ab / b) * Ck
    e_αh = U∞ * cos(Λ) * clα * b * (1 + ab / b) * Ck
    e_αα = U∞ * cos(Λ) * π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2) * Ck
    K̂_f = qf / U∞ * sin(Λ) * b *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return K_f, K̂_f
end


function compute_node_damp(clα, b, eb, ab, U∞, Λ, ω, rho_f, Ck)
    """
    Fluid-added damping matrix
    """
    qf = 0.5 * rho_f * U∞^2 # Dynamic pressure

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2 * π + clα * (1 - 2 * ab / b) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * ab / b) * (2π * b - 2 * clα * eb * Ck)
    C_f = qf / U∞ * cos(Λ) * b *
          [
              c_hh c_hα
              c_αh c_αα
          ]

    # Sweep correction to aerodynamic quasi-steady damping
    e_hh = 2π * ab * b
    e_hα = 2π * ab * b
    e_αh = 2π * ab * b
    e_αα = 2π * b^3 * (0.125 + (ab / b)^2)
    Ĉ_f = qf / U∞ * sin(Λ) * b *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return C_f, Ĉ_f
end

function compute_node_mass(b, ab, rho_f)
    """
    Fluid-added mass matrix
    """
    m_hh = 1
    m_hα = ab
    m_αh = ab
    m_αα = b^2 * (0.125 + (ab / b)^2)
    M_f = π * rho_f * b^2 *
          [
              m_hh m_hα
              m_αh m_αα
          ]

    return M_f
end

function compute_steady_AICs!(AIC::Matrix{Float64}, mesh, FOIL, elemType="BT2")

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    end

    # fluid dynamic pressure    
    qf = 0.5 * FOIL.ρ_f * FOIL.U∞^2

    # strip width
    dy = FOIL.s / (FOIL.neval)
    planformArea = 0.0

    jj = 1 # node index
    for yⁿ ∈ mesh
        K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
        E_f = copy(K_f)  # Sweep correction matrix

        # --- Linearly interpolate values based on y loc ---
        clα = linear(mesh, FOIL.clα, yⁿ)
        c = linear(mesh, FOIL.c, yⁿ)
        b = 0.5 * c # semichord for more readable code
        ab = linear(mesh, FOIL.ab, yⁿ)
        eb = linear(mesh, FOIL.eb, yⁿ)

        # --- Compute forces ---
        # Aerodynamic stiffness (1st row is lift, 2nd row is pitching moment)
        k_hα = -2 * b * clα # lift due to angle of attack
        k_αα = -2 * eb * b * clα # moment due to angle of attack
        K_f = qf * cos(FOIL.Λ)^2 *
              [
                  0.0 k_hα
                  0.0 k_αα
              ]
        # Sweep correction to aerodynamic stiffness
        e_hh = 2 * clα # lift due to w'
        e_hα = -clα * b * (1 - ab / b) # lift due to ψ'
        e_αh = clα * b * (1 + ab / b) # moment due to w'
        e_αα = π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2) # moment due to ψ'
        E_f = qf * sin(FOIL.Λ) * cos(FOIL.Λ) * b *
              [
                  e_hh e_hα
                  e_αh e_αα
              ]

        # --- Compute Compute local AIC matrix for this element ---
        if elemType == "bend-twist"
            println("These aerodynamics are all wrong BTW...")
            AICLocal = -1 * [
                0.00000000 0.0 K_f[1, 2] # Lift
                0.00000000 0.0 0.00000000
                0.00000000 0.0 K_f[2, 2] # Pitching moment
            ]
        elseif elemType == "BT2"
            AICLocal = [
                0.0 E_f[1, 1] K_f[1, 2] E_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                0.0 E_f[2, 1] K_f[2, 2] E_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
        else
            println("nothing else works")
        end

        GDOFIdx = nDOF * (jj - 1) + 1

        AIC[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = AICLocal

        # Add rectangle to planform area
        planformArea += c * dy

        jj += 1 # increment strip counter
    end

    return AIC, planformArea
end

function compute_AICs!(globalMf::Matrix{Float64}, globalCf_r::Matrix{Float64}, globalCf_i::Matrix{Float64}, globalKf_r::Matrix{Float64}, globalKf_i::Matrix{Float64}, mesh, FOIL, U∞, ω, elemType="BT2")
    """
    Compute the AIC matrix for a given mesh
    """

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    end

    # strip width
    dy = FOIL.s / (FOIL.neval)
    planformArea = 0.0

    jj = 1 # node index
    for yⁿ in mesh
        # --- Linearly interpolate values based on y loc ---
        clα::Float64 = linear(mesh, FOIL.clα, yⁿ)
        c::Float64 = linear(mesh, FOIL.c, yⁿ)
        b::Float64 = 0.5 * c # semichord for more readable code
        ab::Float64 = linear(mesh, FOIL.ab, yⁿ)
        eb::Float64 = linear(mesh, FOIL.eb, yⁿ)

        k::Float64 = ω * b / (U∞ * cos(FOIL.Λ)) # reduced frequency

        # Do computation once for efficiency
        CKVec = compute_theodorsen(k)
        Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2] # TODO: for now, put it back together so solve is easy to debug

        K_f, K̂_f = compute_node_stiff(clα, b, eb, ab, U∞, FOIL.Λ, ω, FOIL.ρ_f, Ck)
        C_f, Ĉ_f = compute_node_damp(clα, b, eb, ab, U∞, FOIL.Λ, ω, FOIL.ρ_f, Ck)
        M_f = compute_node_mass(b, ab, FOIL.ρ_f)

        # --- Compute Compute local AIC matrix for this element ---
        if elemType == "bend-twist"
            println("These aerodynamics are all wrong BTW...")
            KLocal = -1 * [
                0.00000000 0.0 K_f[1, 2] # Lift
                0.00000000 0.0 0.00000000
                0.00000000 0.0 K_f[2, 2] # Pitching moment
            ]
        elseif elemType == "BT2"
            KLocal::Matrix{ComplexF64} = [
                0.0 K̂_f[1, 1] K_f[1, 2] K̂_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                0.0 K̂_f[2, 1] K_f[2, 2] K̂_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
            CLocal::Matrix{ComplexF64} = [
                C_f[1, 1] Ĉ_f[1, 1] C_f[1, 2] Ĉ_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                C_f[2, 1] Ĉ_f[2, 1] C_f[2, 2] Ĉ_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
            MLocal::Matrix{Float64} = [
                M_f[1, 1] 0.0 M_f[1, 2] 0.0  # Lift
                0.0 0.0 0.0 0.0
                M_f[2, 1] 0.0 M_f[2, 2] 0.0 # Pitching moment
                0.0 0.0 0.0 0.0
            ]
        else
            println("nothing else works")
        end

        GDOFIdx::Int64 = nDOF * (jj - 1) + 1

        globalKf_r[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(KLocal)
        globalKf_i[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(KLocal)
        globalCf_r[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(CLocal)
        globalCf_i[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(CLocal)
        globalMf[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = MLocal

        # Add rectangle to planform area
        planformArea += c * dy

        jj += 1 # increment strip counter
    end

    return globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, planformArea
end



function compute_steady_hydroLoads(foilStructuralStates, mesh, FOIL, elemType="bend-twist")
    """
    Computes the steady hydrodynamic vector loads 
    given the solved hydrofoil shape (strip theory)
    """
    # ---------------------------
    #   Initializations
    # ---------------------------
    foilTotalStates, nDOF = SolverRoutines.return_totalStates(foilStructuralStates, FOIL, elemType)
    nGDOF = FOIL.neval * nDOF

    # ---------------------------
    #   Strip theory
    # ---------------------------
    AIC = zeros(nGDOF, nGDOF)
    _, planformArea = compute_steady_AICs!(AIC, mesh, FOIL, elemType)

    # --- Compute fluid tractions ---
    hydroTractions = -1 * AIC * foilTotalStates # aerodynamic forces are on the RHS so we negate

    # # --- Debug printout ---
    # println("AIC")
    # show(stdout, "text/plain", AIC)
    # println("")
    # println("Aero loads")
    # println(fTractions)

    return hydroTractions, AIC, planformArea
end

function integrate_hydro_forces(nDOF, hydroTractions, FOIL, elemType="BT2")
    """
    Inputs
    ------
        nDOF: number of degrees of freedom per node
        hydroTractions: 1D array of hydrodynamic forces
    Returns
    -------
        total lift and moment
    """

    if elemType == "bend-twist"
        Moments = hydroTractions[nDOF:nDOF:end]
    elseif elemType == "BT2"
        Moments = hydroTractions[3:nDOF:end]
    else
        error("Invalid element type")
    end
    Lift = hydroTractions[1:nDOF:end]

    # --- Total dynamic hydro force calcs ---
    TotalLift = sum(Lift) * FOIL.s / FOIL.neval
    TotalMoment = sum(Moments) * FOIL.s / FOIL.neval

    return TotalLift, TotalMoment
end

function apply_BCs(K::Matrix{Float64}, C::Matrix{Float64}, M::Matrix{Float64}, globalDOFBlankingList::Vector{Int64})
    """
    Applies BCs for nodal displacements

    """
    newK = K[
        setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    ]
    newM = M[
        setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    ]
    newC = C[
        setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    ]
    return newK, newC, newM
end

end # end module

