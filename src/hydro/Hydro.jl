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
export compute_AICs, apply_BCs

# --- Libraries ---
using SpecialFunctions
using LinearAlgebra
using Statistics
using Plots
using Zygote, ChainRulesCore, FiniteDifferences
include("../solvers/SolverRoutines.jl")
using .SolverRoutines

# ==============================================================================
#                         Unsteady hydro functions
# ==============================================================================
function compute_theodorsen(k)
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
    if k < 1.11e-16
        println("You can't use the Theodorsen function for k = 0!")
        #     # println(k)
        #     k += 1.11e-16 # force it to be non-zero
        #     #     CᵣLim = 1.0
        #     #     Cᵢ = 0.0
        #     #     ans = [Cᵣ, Cᵢ]
    end

    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # --- These are the analytic solutions to Theodorsen's function ---
    C_r_analytic = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) * divDenom
    C_i_analytic = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) * divDenom

    # # --- Zero frequency limit ---
    # Cᵣ_lim = 1.0
    # Cᵢ_lim = 0.0
    # kSigmoid = 1000.0 # sigmoid steepness
    # logistic = 1 / (1 + exp(-kSigmoid * -1 * (k - 0.0))) # this is a L-R flipped sigmoid so below 0 the function is 1.0

    # C_r = Cᵣ_lim * logistic + C_r_analytic
    # C_i = Cᵢ_lim * logistic + C_i_analytic
    ans = [C_r_analytic, C_i_analytic]

    return ans
end

function compute_pade(k)
    """
    3-term Pade approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    """
    s̄ = 1im * k
    scube = s̄^3
    ssquare = s̄^2
    C = (scube + 3.5 * ssquare + 2.7125 * s̄ + 0.46875) / (2 * scube + 6.5 * ssquare + 4.25 * s̄ + 0.46875)
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

function compute_fraccalc(k)
    """
    Fractional calculus approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    """
    s̄ = 1im * k
    F = 2.19
    β = 5 / 6
    prod = F * s̄^β
    C = (1 + prod) / (1 + 2 * prod)
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

function compute_fraccalc_d(k)
    """
    Fractional calculus approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    Undefined at s = 0 b/c beta = 5/6 :(
    """
    s̄ = 1im * k
    F = 2.19
    β = 5 / 6
    prod = F * s̄^β
    prod2 = F * s̄^(β - 1)
    C = ((1 + 2 * prod) * (β * 1im * prod2) - (1 + prod) * (2 * β * 1im * prod2)) / (1 + 2 * prod)^2
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

# ==============================================================================
#                         Free surface effects
# ==============================================================================
# The following functions compute the generic force coefficients 'C' for the equation
#     C = Ci  α̈ + Cd α̇ + Cs α
# However, none of the ROM appears to account for heave effects
# The ROM should not be used for k > 0.2 and should DEFINITELY not be used for k > 1.0

function compute_clsROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 5.268
    p10 = 0.217
    p01 = -6.085
    p20 = -0.0141
    p11 = -0.0425
    p02 = 4.586
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_cldROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 0.0837
    p10 = -0.0192
    p01 = -5.597
    p20 = 0.0
    p11 = 0.0251
    p02 = 26.662
    p12 = 0.00304
    p03 = -16.218
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

#  NOTE: the moments are about the elastic axis
function compute_cmsROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 0.0633
    p10 = -0.00883
    p01 = -0.000890
    p20 = 0.000634
    p11 = 0.00106
    p02 = -0.127
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_cmdROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = -0.000675
    p10 = 0.000320
    p01 = -1.023#*0.5
    p20 = 0.0
    p11 = -0.00355#*0.25
    p02 = -0.177#*0.25
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_glauert_circ(semispan, chordVec, α₀, U∞, nNodes, h=nothing, useFS=false)
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
        cl_α : array, shape (nNodes,)
            sectional lift slopes for a 3D wing [rad⁻¹] starting from the root
            sometimes denoted in literature as 'a₀'

    NOTE:
    We use keyword arguments (denoted by the ';' to be more explicit)

    This follows the formulation in
    'Principles of Naval Architecture Series (PNA) - Propulsion 2010'
    by Justin Kerwin & Jacques Hadler
    """

    ỹ = π / 2 * ((1:1:nNodes) / nNodes) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
    y = -semispan * cos.(ỹ) # the physical coordinate (y) is only calculated to the root (-semispan, 0)

    # ---------------------------
    #   PLANFORM SHAPES: rectangular is outdated
    # ---------------------------
    # # --- Rectangular ---
    # chordₚ = chord
    # --- Elliptical planform ---
    chordₚ = chordVec .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?

    n = (1:1:nNodes) * 2 - ones(nNodes) # node numbers x2 (node multipliers)

    b = π / 4 * (chordₚ / semispan) * α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, nNodes]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = π / 4 * chordₚ / semispan .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:nNodes*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) #matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the coefficients in Glauert's Fourier series ---
    ã = chord11 \ b

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise distribution of free vortex strength (Γ(y) in textbook)

    if useFS
        γ_FS = use_free_surface(γ, α₀, U∞, chordVec, h)
    end

    cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl / (α₀ + 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    dl = semispan / (nNodes - 1)
    xq = -semispan:dl:0

    # TODO: THERE IS A BUG ON THIS LINE THAT MAKES THE DERIVATIVES WRT SPAN FOR THE WHOLE CODE WRONG
    cl_α = SolverRoutines.do_linear_interp(y, clα, xq)

    return reverse(cl_α)
end

function use_free_surface(γ, α₀, U∞, chordVec, h)
    """
    Modify hydro loads based on the free-surface condition that is Fn independent

    Inputs
    ------
        γ spanwise vortex strength m^2/s
        NOTE: with the current form, this is the negative of what some textbooks do so for example
        Typically L = - ρ U int( Γ(y))dy
        but Kerwin and Hadler do
        C_L = 2Γ/(Uc)
    Returns:
    --------
        γ_FS modified vortex strength using the high-speed, free-surface BC
    """

    Fnh = U∞ / (sqrt(9.81 * h))
    # Find limiting case
    if Fnh < 10 / sqrt(h / mean(chordVec))
        println("Violating high-speed free-surface BC with Fnh*sqrt(h/c) of")
        println(Fnh * sqrt(h / mean(chordVec)))
        println("Fnh is")
        println(Fnh)
    end

    # Circulation with no FS effect
    γ_2DnoFS = -U∞ * chordVec * π * α₀

    # Circulation with high-speed FS effect
    correctionVector = (1.0 .+ 16.0 .* (h ./ chordVec) .^ 2) ./ (2.0 .+ 16.0 .* (h ./ chordVec) .^ 2)
    γ_2DFS = -U∞ * chordVec * π * α₀ .* correctionVector

    # Corrected circulation
    γ_FS = γ + γ_2DnoFS - γ_2DFS

    return γ_FS
end

# ==============================================================================
#                         Hydrodynamic strip forces
# ==============================================================================
function compute_node_stiff(clα, b, eb, ab, U∞, Λ, rho_f, Ck)
    """
    Hydrodynamic stiffness force
    """
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b # precompute division by b to get a

    # Aerodynamic quasi-steady stiffness
    # (1st row is lift, 2nd row is pitching moment)

    k_hα = -2 * b * clα * Ck # lift due to angle of attack
    k_αα = -2 * eb * b * clα * Ck # moment due to angle of attack (disturbing)
    K_f = qf * cos(Λ) * cos(Λ) *
          [
              0.0 k_hα
              0.0 k_αα
          ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = U∞ * cos(Λ) * 2 * clα * Ck
    e_hα = U∞ * cos(Λ) * (-clα) * b * (1 - a) * Ck
    e_αh = U∞ * cos(Λ) * clα * b * (1 + a) * Ck
    e_αα = U∞ * cos(Λ) *
           (π * b * b - clα * eb * b * (1 - 2 * (a)) * Ck)
    K̂_f = qf / U∞ * sin(Λ) * b *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return K_f, K̂_f
end

function compute_node_damp(clα, b, eb, ab, U∞, Λ, rho_f, Ck)
    """
    Fluid-added damping matrix
    """
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b # precompute division by b to get a

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2π + clα * (1 - 2 * a) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * a) * (2π * b - 2 * clα * eb * Ck)
    C_f = qf / U∞ * cos(Λ) * b *
          [
              c_hh c_hα
              c_αh c_αα
          ]

    # Sweep correction to aerodynamic quasi-steady damping
    e_hh = 2π * b
    e_hα = 2π * ab * b
    e_αh = 2π * ab * b
    e_αα = 2π * b^3 * (0.125 + a * a)
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
    bSquared = b * b # precompute square of b
    a = ab / b # precompute division by b to get a
    m_hh = 1
    m_hα = ab
    m_αh = ab
    m_αα = bSquared * (0.125 + a * a)
    M_f = π * rho_f * bSquared *
          [
              m_hh m_hα
              m_αh m_αα
          ]

    return M_f
end

function compute_steady_AICs!(AIC::Matrix{Float64}, mesh, chordVec, abVec, ebVec, Λ, FOIL, elemType="BT2")
    """
    Compute the steady aerodynamic influence coefficients (AICs) for a given mesh
    This is different from the general AIC method because there is no frequency dependence
    Inputs
    ------
    AIC: Matrix
        Aerodynamic influence coefficient matrix
    mesh: Array
        Mesh of the foil
    FOIL: struct
        Struct containing the foil implicit constants
    elemType: String
        Element type

    Returns
    -------
    AIC: Matrix
        Aerodynamic influence coefficient matrix (now filled out)
    """

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    end

    # fluid dynamic pressure
    qf = 0.5 * FOIL.ρ_f * FOIL.U∞ * FOIL.U∞

    # --- Initialize planform area counter ---
    planformArea = 0.0

    jj = 1 # node index

    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    for yⁿ ∈ mesh
        # --- compute strip width ---
        Δy = 0.0
        if jj < FOIL.nNodes
            Δy = mesh[jj+1] - mesh[jj]
            if jj == 1
                Δy = Δy / 2
            end
        else
            Δy = mesh[jj] - mesh[jj-1]
            if jj == FOIL.nNodes
                Δy = Δy / 2
            end
        end

        # --- Initialize aero-force matrices ---
        K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
        E_f = copy(K_f)  # Sweep correction matrix


        # --- Interpolate values based on y loc ---
        clα = SolverRoutines.do_linear_interp(mesh, FOIL.clα, yⁿ)
        c = SolverRoutines.do_linear_interp(mesh, chordVec, yⁿ)
        ab = SolverRoutines.do_linear_interp(mesh, abVec, yⁿ)
        eb = SolverRoutines.do_linear_interp(mesh, ebVec, yⁿ)
        b = 0.5 * c # semichord for more readable code

        # --- Compute forces ---
        # Aerodynamic stiffness (1st row is lift, 2nd row is pitching moment)
        k_hα = -2 * b * clα # lift due to angle of attack
        k_αα = -2 * eb * b * clα # moment due to angle of attack
        K_f = qf * cos(Λ)^2 *
              [
                  0.0 k_hα
                  0.0 k_αα
              ]
        # Sweep correction to aerodynamic stiffness
        e_hh = 2 * clα # lift due to w'
        e_hα = -clα * b * (1 - ab / b) # lift due to ψ'
        e_αh = clα * b * (1 + ab / b) # moment due to w'
        e_αα = π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2) # moment due to ψ'
        E_f = qf * sin(Λ) * cos(Λ) * b *
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

        # Add local AIC to global AIC and remember to multiply by strip width to get the right result
        AIC[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = AICLocal * Δy

        # Add rectangle to planform area
        planformArea += c * Δy

        jj += 1 # increment strip counter
    end

    return AIC, planformArea
end

function compute_AICs(dim, mesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType="BT2")
    """
    Compute the AIC matrix for a given mesh using LHS convention
        (i.e., -ve force is disturbing, not restoring)
    Inputs
    ------
    AIC: Matrix
        Aerodynamic influence coefficient matrix broken up into added mass, damping, and stiffness
        in such a way that 
            {F} = -([Mf]{udd} + [Cf]{ud} + [Kf]{u})
        These are matrices
    mesh: Array
        Mesh of the foil
    FOIL: struct
        Struct containing the foil implicit constants
    elemType: String
        Element type

    Returns
    -------
    AIC: Matrix
        Aerodynamic influence coefficient matrix (now filled out)
    """

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    end

    # --- Initialize global matrices ---
    globalMf_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_i_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_i_z = Zygote.Buffer(zeros(dim, dim))
    # Zygote initialization
    for jj in 1:dim
        for ii in 1:dim
            globalMf_z[ii, jj] = 0.0
            globalCf_r_z[ii, jj] = 0.0
            globalCf_i_z[ii, jj] = 0.0
            globalKf_r_z[ii, jj] = 0.0
            globalKf_i_z[ii, jj] = 0.0
        end
    end
    # --- Initialize planform area counter ---
    planformArea = 0.0

    jj = 1 # node index
    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    nNodes = length(mesh)
    for yⁿ in mesh
        # --- compute strip width ---
        Δy = 0.0
        if jj < nNodes
            Δy = mesh[jj+1] - mesh[jj]
            if jj == 1
                Δy = Δy / 2
            end
        else
            Δy = mesh[jj] - mesh[jj-1]
            if jj == nNodes
                Δy = Δy / 2
            end
        end

        # --- Linearly interpolate values based on y loc ---
        clα::Float64 = SolverRoutines.do_linear_interp(mesh, FOIL.clα, yⁿ)
        c::Float64 = SolverRoutines.do_linear_interp(mesh, chordVec, yⁿ)
        ab::Float64 = SolverRoutines.do_linear_interp(mesh, abVec, yⁿ)
        eb::Float64 = SolverRoutines.do_linear_interp(mesh, ebVec, yⁿ)
        b::Float64 = 0.5 * c # semichord for more readable code

        k = ω * b / (U∞ * cos(Λ)) # local reduced frequency

        # Do computation once for efficiency
        CKVec = compute_theodorsen(k)
        Ck = CKVec[1] + 1im * CKVec[2]

        K_f, K̂_f = compute_node_stiff(clα, b, eb, ab, U∞, Λ, FOIL.ρ_f, Ck)
        C_f, Ĉ_f = compute_node_damp(clα, b, eb, ab, U∞, Λ, FOIL.ρ_f, Ck)
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
            KLocal = [
                0.0 K̂_f[1, 1] K_f[1, 2] K̂_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                0.0 K̂_f[2, 1] K_f[2, 2] K̂_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
            CLocal = [
                C_f[1, 1] Ĉ_f[1, 1] C_f[1, 2] Ĉ_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                C_f[2, 1] Ĉ_f[2, 1] C_f[2, 2] Ĉ_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
            MLocal = [
                M_f[1, 1] 0.0 M_f[1, 2] 0.0  # Lift
                0.0 0.0 0.0 0.0
                M_f[2, 1] 0.0 M_f[2, 2] 0.0 # Pitching moment
                0.0 0.0 0.0 0.0
            ]
        else
            println("nothing else works")
        end

        GDOFIdx::Int64 = nDOF * (jj - 1) + 1

        # Add local AIC to global AIC and remember to multiply by strip width to get the right result
        globalKf_r_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(KLocal) * Δy
        globalKf_i_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(KLocal) * Δy
        globalCf_r_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(CLocal) * Δy
        globalCf_i_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(CLocal) * Δy
        globalMf_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = MLocal * Δy

        # Add rectangle to planform area
        planformArea += c * Δy

        jj += 1 # increment strip counter
    end

    return copy(globalMf_z), copy(globalCf_r_z), copy(globalCf_i_z), copy(globalKf_r_z), copy(globalKf_i_z), planformArea
end



function compute_steady_hydroLoads(foilStructuralStates, mesh, α₀, chordVec, abVec, ebVec, Λ, FOIL, elemType="bend-twist")
    """
    Computes the steady hydrodynamic vector loads
    given the solved hydrofoil shape (strip theory)
    """
    # ---------------------------
    #   Initializations
    # ---------------------------
    foilTotalStates, nDOF = SolverRoutines.return_totalStates(foilStructuralStates, α₀, elemType)
    nGDOF = FOIL.nNodes * nDOF

    # ---------------------------
    #   Strip theory
    # ---------------------------
    AIC = zeros(nGDOF, nGDOF)
    _, planformArea = compute_steady_AICs!(AIC, mesh, chordVec, abVec, ebVec, Λ, FOIL, elemType)

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

function integrate_hydroLoads(foilStructuralStates, fullAIC, DFOIL, elemType="BT2")
    # function integrate_hydroLoads(foilStructuralStates, fullAIC, α₀, elemType="BT2")
    """
    Inputs
    ------
        fullAIC: AIC matrix
        FOIL: FOIL struct
        elemType: element type
    Returns
    -------
        force vector and total lift and moment

    TODO: have steady solver call this too
    """

    # --- Initializations ---
    # This is dynamic deflection + rigid shape of foil
    foilTotalStates, nDOF = SolverRoutines.return_totalStates(foilStructuralStates, α₀, elemType)

    # --- Strip theory ---
    # This is the hydro force traction vector
    ForceVector = fullAIC * foilTotalStates


    if elemType == "bend-twist"
        nDOF = 3
        Moments = ForceVector[nDOF:nDOF:end]
    elseif elemType == "BT2"
        nDOF = 4
        Moments = ForceVector[3:nDOF:end]
    else
        error("Invalid element type")
    end
    Lift = ForceVector[1:nDOF:end]

    # --- Total dynamic hydro force calcs ---
    TotalLift = sum(Lift)
    TotalMoment = sum(Moments)

    return ForceVector, TotalLift, TotalMoment
end

function apply_BCs(K, C, M, globalDOFBlankingList::Vector{Int64})
    """
    Applies BCs for nodal displacements
    """

    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newC = C[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]

    return newK, newC, newM
end

end # end module

