# --- Julia ---

# @File    :   HydroStrip.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines

module HydroStrip
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
using Zygote, ChainRulesCore
using Printf, DelimitedFiles
include("../solvers/SolverRoutines.jl")
using .SolverRoutines
using SparseArrays

# --- Globals ---
global XDIM = 1
global YDIM = 2
global ZDIM = 3

# ==============================================================================
#                         Unsteady hydro functions
# ==============================================================================
function compute_theodorsen(k)
    """
    Theodorsen's transfer function for unsteady aero/hydrodynamics of a sinusoidally oscillating foil.
    w/ separate real and imaginary parts. 
    This is potential flow theory.
    This form is also wrong for k < 0 and should use the modified bessel funcs

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

function compute_sears(k)
    """
    Sears transfer function for an airfoil subject to sinusoidal gusts.
    This is potential flow theory.
    """

    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    # TODO: do in real data type only
    # divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # S_r = divDenom

    H02 = H₀²ᵣ + 1im * H₀²ᵢ
    H12 = H₁²ᵣ + 1im * H₁²ᵢ
    S = 2 * 1im / (π * k) / (H12 + 1im * H02)

    return S
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

    mu = π / 4 * (chordₚ / semispan)
    b = mu * α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, nNodes]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = mu .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:nNodes*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

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

    cl_α = SolverRoutines.do_linear_interp(y, clα, xq)
    # If this is fully ventilated, can divide the slope by 4

    return reverse(cl_α)
end

function compute_LL_ventilated(semispan, submergedDepth, α₀, cl_α_FW)
    """
    Slope of the 3D lift coefficient with respect to the angle of attack considering surface-piercing vertical strut
    From Harwood 2019 Part 1

    a0 = π/2 * (1 - √(1 - (2 * submergedDepth / semispan)^2))
    """
    # TODO: get Lc from Casey's paper
    Lc_c = Lc / c

    a0 = ((π / 2) * (Lc_c^3) - 2 * (Lc_c^2) + 4.5 * Lc_c + 1) / ((Lc_c^3) - (Lc_c^2) + 0.75 * Lc_c + 1 / (2π))
    return a0
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
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b 
    clambda = cos(Λ)
    slambda = sin(Λ)
    # Aerodynamic quasi-steady stiffness
    # (1st row is lift, 2nd row is pitching moment)


    k_hα = -2 * b * clα * Ck # lift due to angle of attack
    k_αα = -2 * eb * b * clα * Ck # moment due to angle of attack (disturbing)
    K_f = qf * clambda * clambda *
          [
              0.0 k_hα
              0.0 k_αα
          ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = U∞ * clambda * 2 * clα * Ck
    e_hα = U∞ * clambda * (-clα) * b * (1 - a) * Ck
    e_αh = U∞ * clambda * clα * b * (1 + a) * Ck
    e_αα = U∞ * clambda *
           (π * b * b - clα * eb * b * (1 - 2 * (a)) * Ck)
    K̂_f = qf / U∞ * slambda * b *
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
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    clambda = cos(Λ)
    slambda = sin(Λ)
    coeff = qf / U∞ * b

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2π + clα * (1 - 2 * a) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * a) * (2π * b - 2 * clα * eb * Ck)
    C_f = coeff*clambda *
          [
              c_hh c_hα
              c_αh c_αα
          ]

    # Sweep correction to aerodynamic quasi-steady damping
    e_hh = 2π * b
    e_hα = 2π * ab * b
    e_αh = e_hα
    e_αα = 2π * b^3 * (0.125 + a * a)
    Ĉ_f = coeff * slambda *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return C_f, Ĉ_f
end

function compute_node_stiff_faster(clα, b, eb, ab, U∞, clambda,slambda, rho_f, Ck)
    """
    Hydrodynamic stiffness force
    """
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b 
    Uclambda = U∞ * clambda
    clalphabCk = clα * b * Ck
    # Aerodynamic quasi-steady stiffness
    # (1st row is lift, 2nd row is pitching moment)


    k_hα = -2 * b * clα * Ck # lift due to angle of attack
    k_αα = k_hα * eb # moment due to angle of attack (disturbing)
    K_f = qf * clambda * clambda *
          [
              0.0 k_hα;
              0.0 k_αα
          ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = Uclambda * 2 * clα * Ck
    e_hα = Uclambda * (1 - a) * (-clalphabCk)
    e_αh = Uclambda * (1 + a) * clalphabCk
    e_αα = Uclambda *
           (π * b * b - clalphabCk * eb * (1 - 2 * (a)))
    K̂_f = qf / U∞ * slambda * b *
           [
               e_hh e_hα;
               e_αh e_αα
           ]

    return K_f, K̂_f
end

function compute_node_damp_faster(clα, b, eb, ab, U∞, clambda, slambda, rho_f, Ck)
    """
    Fluid-added damping matrix
    """
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    coeff = qf / U∞ * b

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2π + clα * (1 - 2 * a) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * a) * (2π * b - 2 * clα * eb * Ck)
    C_f = coeff*clambda *
          [
              c_hh c_hα;
              c_αh c_αα
          ]

    # Sweep correction to aerodynamic quasi-steady damping
    e_hh = 2π * b
    e_hα = 2π * ab * b
    e_αh = e_hα
    e_αα = 2π * b^3 * (0.125 + a * a)
    Ĉ_f = coeff * slambda *
           [
               e_hh e_hα;
               e_αh e_αα
           ]

    return C_f, Ĉ_f
end

function compute_node_mass(b, ab, rho_f)
    """
    Fluid-added mass matrix
    """
    # --- Precomputes ---
    bSquared = b * b # precompute square of b
    a = ab / b # precompute division by b to get a

    m_hh = 1.0
    m_hα = ab
    m_αh = ab
    m_αα = bSquared * (0.125 + a * a)
    M_f = π * rho_f * bSquared *
          [
              m_hh m_hα;
              m_αh m_αα
          ]

    return M_f
end

function compute_steady_AICs!(AIC::Matrix{Float64}, aeroMesh, chordVec, abVec, ebVec, Λ, FOIL, elemType="BT2")
    """
    Compute the steady aerodynamic influence coefficients (AICs) for a given mesh
    This is different from the general AIC method because there is no frequency dependence
    Inputs
    ------
    AIC: Matrix
        Aerodynamic influence coefficient matrix
    aeroMesh: Array
        Mesh of the foil
    FOIL: struct
        Struct containing the foil implicit constants
    elemType: String
        Element type

    Returns
    -------
    AIC: Matrix
        Aerodynamic influence coefficient matrix (now filled out)
        In the global reference frame
    """

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nLocDOF = 4
        nDOF = nLocDOF # number of global DOFs at 1 node
    elseif elemType == "COMP2"
        nLocDOF = 9
        nDOF = nLocDOF
    end

    # fluid dynamic pressure
    qf = 0.5 * FOIL.ρ_f * FOIL.U∞ * FOIL.U∞

    # --- Initialize planform area counter ---
    planformArea = 0.0

    jj = 1 # node index

    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    # Basic straight line code
    if ndims(aeroMesh) == 1
        println("=============================")
        println("Using straight line code")
        println("=============================")
        for yⁿ ∈ aeroMesh
            # --- compute strip width ---
            Δy = 0.0
            if jj < FOIL.nNodes
                Δy = aeroMesh[jj+1] - aeroMesh[jj]
                if jj == 1
                    Δy = Δy / 2
                end
            else
                Δy = aeroMesh[jj] - aeroMesh[jj-1]
                if jj == FOIL.nNodes
                    Δy = Δy / 2
                end
            end

            # --- Initialize aero-force matrices ---
            K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
            E_f = copy(K_f)  # Sweep correction matrix


            # --- Interpolate values based on y loc ---
            clα = SolverRoutines.do_linear_interp(aeroMesh, FOIL.clα, yⁿ)
            c = SolverRoutines.do_linear_interp(aeroMesh, chordVec, yⁿ)
            ab = SolverRoutines.do_linear_interp(aeroMesh, abVec, yⁿ)
            eb = SolverRoutines.do_linear_interp(aeroMesh, ebVec, yⁿ)
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

            # --- Compute Compute local AIC matrix for this strip ---
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
    elseif ndims(aeroMesh) == 2
        # println("=============================")
        # println("Using 3D mesh code")
        # println("=============================")
        for yⁿ in aeroMesh[:, YDIM]
            # --- compute strip width ---
            Δy = 0.0
            if jj < FOIL.nNodes
                nVec = (aeroMesh[jj+1, :] - aeroMesh[jj, :])
            else
                nVec = (aeroMesh[jj, :] - aeroMesh[jj-1, :])
            end
            # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change
            lᵉ::Float64 = norm(nVec, 2) # length of elem
            Δy = lᵉ
            if jj == 1 || jj == FOIL.nNodes
                Δy = 0.5 * lᵉ
            end

            nVec = nVec / lᵉ # normalize

            # --- Initialize aero-force matrices ---
            K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
            E_f = copy(K_f)  # Sweep correction matrix


            # --- Interpolate values based on y loc ---
            # TODO: you will need to fix this later
            clα = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], FOIL.clα, yⁿ)
            c = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], chordVec, yⁿ)
            ab = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], abVec, yⁿ)
            eb = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], ebVec, yⁿ)
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
                    0.0 0.0 K_f[1, 2] # Lift
                    0.0 0.0 0.00000000
                    0.0 0.0 K_f[2, 2] # Pitching moment
                ]
            elseif elemType == "BT2"
                AICLocal = [
                    0.0 E_f[1, 1] K_f[1, 2] E_f[1, 2]  # Lift
                    0.0 0.0000000 0.0000000 0.0000000
                    0.0 E_f[2, 1] K_f[2, 2] E_f[2, 2] # Pitching moment
                    0.0 0.0000000 0.0000000 0.0000000
                ]
            elseif elemType == "COMP2"
                # NOTE: Done in local aero coordinates
                AICLocal = [
                    # u v   w   phi       theta     psi phi'     theta'
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
                    0.0 0.0 0.0 K_f[1, 2] E_f[1, 1] 0.0 E_f[1, 2] 0.0 0.0 # w
                    0.0 0.0 0.0 K_f[2, 2] E_f[2, 1] 0.0 E_f[2, 2] 0.0 0.0 # phi
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
                ]
            else
                println("nothing else works")
            end

            GDOFIdx = nDOF * (jj - 1) + 1

            # ---------------------------
            #   Transformation of AIC
            # ---------------------------
            # Aerodynamics need to happen in global reference frame
            Γ = SolverRoutines.get_transMat(nVec, 1.0, elemType)
            AICLocal = Γ'[1:nLocDOF, 1:nLocDOF] * AICLocal * Γ[1:nLocDOF, 1:nLocDOF]

            # Add local AIC to global AIC and remember to multiply by strip width to get the right result
            # AIC[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = AICStrip * Δy
            AIC[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = AICLocal * Δy

            # Add rectangle to planform area
            planformArea += c * Δy

            jj += 1 # increment strip counter
        end
    end

    return AIC, planformArea
end

function compute_AICs(dim, aeroMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType="BT2")
    """
    Compute the AIC matrix for a given aeroMesh using LHS convention
        (i.e., -ve force is disturbing, not restoring)
    Inputs
    ------
    AIC: Matrix
        Aerodynamic influence coefficient matrix broken up into added mass, damping, and stiffness
        in such a way that 
            {F} = -([Mf]{udd} + [Cf]{ud} + [Kf]{u})
        These are matrices
    aeroMesh: Array
        Mesh of the foil
    FOIL: struct
        Struct containing the foil implicit constants
    elemType: String
        Element type

    Returns
    -------
    AIC: Matrix
        Aerodynamic influence coefficient matrix (now filled out)
        in the global reference frame
    """

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    elseif elemType == "COMP2"
        nDOF = 9
    end

    # --- Initialize global matrices ---
    globalMf_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_i_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_i_z = Zygote.Buffer(zeros(dim, dim))
    # Zygote initialization
    # copyto!(globalMf_z, spzeros(dim,dim)) # copyto is unstable
    # TODO: is julia pass by reference or value?
    # It's pass by reference
    globalMf_z[:, :] = zeros(dim, dim)
    globalCf_r_z[:, :] = zeros(dim, dim)
    globalCf_i_z[:, :] = zeros(dim, dim)
    globalKf_r_z[:, :] = zeros(dim, dim)
    globalKf_i_z[:, :] = zeros(dim, dim)
    # --- Initialize planform area counter ---
    planformArea = 0.0

    jj = 1 # node index
    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    # Basic straight line code
    if ndims(aeroMesh) == 1
        for yⁿ in aeroMesh
            # --- compute strip width ---
            Δy = 0.0
            if jj < FOIL.nNodes
                Δy = aeroMesh[jj+1] - aeroMesh[jj]
                if jj == 1
                    Δy = Δy / 2
                end
            else
                Δy = aeroMesh[jj] - aeroMesh[jj-1]
                if jj == FOIL.nNodes
                    Δy = Δy / 2
                end
            end

            # --- Linearly interpolate values based on y loc ---
            clα::Float64 = SolverRoutines.do_linear_interp(aeroMesh, FOIL.clα, yⁿ)
            c::Float64 = SolverRoutines.do_linear_interp(aeroMesh, chordVec, yⁿ)
            ab::Float64 = SolverRoutines.do_linear_interp(aeroMesh, abVec, yⁿ)
            eb::Float64 = SolverRoutines.do_linear_interp(aeroMesh, ebVec, yⁿ)
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
            elseif elemType == "COMP2"
                # NOTE: Done in aero coordinates
                KLocal = [
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # u
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # v
                    0.0 0.0 0.0 K_f[1, 2] K̂_f[1, 1] 0.0 0.0 K̂_f[1, 2] 0.0  # w
                    0.0 0.0 0.0 K_f[2, 2] K̂_f[2, 1] 0.0 0.0 K̂_f[2, 2] 0.0 # phi
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # phi'
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta'
                    0.0 0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi'
                ]
                CLocal = [
                    # u v   w         phi 
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
                    0.0 0.0 C_f[1, 1] C_f[1, 2] Ĉ_f[1, 1] 0.0 Ĉ_f[1, 2] 0.0 0.0  # w
                    0.0 0.0 C_f[2, 1] C_f[2, 2] Ĉ_f[2, 1] 0.0 Ĉ_f[2, 2] 0.0 0.0 # phi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
                ]
                MLocal = [
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # u
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # v
                    0.0 0.0 M_f[1, 1] M_f[1, 2] 0.0 0.0 0.0 0.0 0.0 # w
                    0.0 0.0 M_f[2, 1] M_f[2, 2] 0.0 0.0 0.0 0.0 0.0 # phi
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # theta
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # psi
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # phi'
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # theta'
                    0.0 0.0 0.0000000 0.0000000 0.0 0.0 0.0 0.0 0.0 # psi'
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
    elseif ndims(aeroMesh) == 2
        for yⁿ in aeroMesh[:, YDIM]
            # --- compute strip width ---
            Δy = 0.0
            if jj < FOIL.nNodes
                nVec = (aeroMesh[jj+1, :] - aeroMesh[jj, :])
            else
                nVec = (aeroMesh[jj, :] - aeroMesh[jj-1, :])
            end
            # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change
            lᵉ::Float64 = norm(nVec, 2) # length of elem
            Δy = lᵉ
            if jj == 1 || jj == FOIL.nNodes
                Δy = 0.5 * lᵉ
            end

            nVec = nVec / lᵉ # normalize

            # --- Linearly interpolate values based on y loc ---
            clα::Float64 = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], FOIL.clα, yⁿ)
            c::Float64 = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], chordVec, yⁿ)
            ab::Float64 = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], abVec, yⁿ)
            eb::Float64 = SolverRoutines.do_linear_interp(aeroMesh[:, YDIM], ebVec, yⁿ)
            b::Float64 = 0.5 * c # semichord for more readable code

            
            # --- Precomputes ---
            clambda = cos(Λ)
            slambda = sin(Λ)
            k = ω * b / (U∞ * clambda) # local reduced frequency
            # Do computation once for efficiency
            CKVec = compute_theodorsen(k)
            Ck = CKVec[1] + 1im * CKVec[2]

            # K_f, K̂_f = compute_node_stiff(clα, b, eb, ab, U∞, Λ, FOIL.ρ_f, Ck)
            # C_f, Ĉ_f = compute_node_damp(clα, b, eb, ab, U∞, Λ, FOIL.ρ_f, Ck)
            K_f, K̂_f = compute_node_stiff_faster(clα, b, eb, ab, U∞, clambda,slambda, FOIL.ρ_f, Ck)
            C_f, Ĉ_f = compute_node_damp_faster(clα, b, eb, ab, U∞, clambda,slambda, FOIL.ρ_f, Ck)
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
            elseif elemType == "COMP2"
                # TODO: SLOW, is this form of matrix initialization slow?
                # NOTE: Done in local aero coordinates
                KLocal = [
                    # u v   w         phi       theta     psi phi'     theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
                    0.0 0.0 0.0000000 K_f[1, 2] K̂_f[1, 1] 0.0 K̂_f[1, 2] 0.0 0.0  # w
                    0.0 0.0 0.0000000 K_f[2, 2] K̂_f[2, 1] 0.0 K̂_f[2, 2] 0.0 0.0 # phi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
                ]
                CLocal = [
                    # u v   w         phi       theta     psi phi'     theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
                    0.0 0.0 C_f[1, 1] C_f[1, 2] Ĉ_f[1, 1] 0.0 Ĉ_f[1, 2] 0.0 0.0  # w
                    0.0 0.0 C_f[2, 1] C_f[2, 2] Ĉ_f[2, 1] 0.0 Ĉ_f[2, 2] 0.0 0.0 # phi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
                ]
                MLocal = [
                    # u v   w         phi       theta     psi phi'     theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # u
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # v
                    0.0 0.0 M_f[1, 1] M_f[1, 2] 0.0000000 0.0 0.0 0.0000000 0.0 # w
                    0.0 0.0 M_f[2, 1] M_f[2, 2] 0.0000000 0.0 0.0 0.0000000 0.0 # phi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # phi'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta'
                    0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi'
                ]
            else
                println("nothing else works")
            end

            # ---------------------------
            #   Transformation of AIC
            # ---------------------------
            # Aerodynamics need to happen in global reference frame
            Γ = SolverRoutines.get_transMat(nVec, 1.0, elemType)
            KLocal = Γ'[1:nDOF, 1:nDOF] * KLocal * Γ[1:nDOF, 1:nDOF]
            CLocal = Γ'[1:nDOF, 1:nDOF] * CLocal * Γ[1:nDOF, 1:nDOF]
            MLocal = Γ'[1:nDOF, 1:nDOF] * MLocal * Γ[1:nDOF, 1:nDOF]

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
    end

    # TODO: SLOW, this could potentially be because it is a large matrix...
    return copy(globalMf_z), copy(globalCf_r_z), copy(globalCf_i_z), copy(globalKf_r_z), copy(globalKf_i_z), planformArea
end

function compute_steady_hydroLoads(foilStructuralStates, mesh, α₀, chordVec, abVec, ebVec, Λ, FOIL, elemType="bend-twist",)
    """
    Computes the steady hydrodynamic vector loads
    given the solved hydrofoil shape (strip theory)

    foilStructuralStates: array
        Structural states of the foil in GLOBAL FRAME
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
    # writedlm("DebugAIC.csv", AIC, ',') # THESE ARE THE SAME
    # open("totalStates.dat", "w") do io # THESE ARE THE SAME
    #     stringData = elemType * "\n"
    #     write(io, stringData)
    #     if elemType == "COMP2"
    #         nDOF = 9
    #         nStart = 4
    #     elseif elemType == "BT2"
    #         nDOF = 4
    #         nStart = 3
    #     end
    #     for qⁿ ∈ foilTotalStates#[nStart:nDOF:end]
    #         stringData = @sprintf("%.8f\n", qⁿ)
    #         write(io, stringData)
    #     end
    # end
    # open("structuralStates.dat", "w") do io # THESE ARE THE SAME
    #     stringData = elemType * "\n"
    #     write(io, stringData)
    #     if elemType == "COMP2"
    #         nDOF = 9
    #         nStart = 4
    #     elseif elemType == "BT2"
    #         nDOF = 4
    #         nStart = 3
    #     end
    #     for qⁿ ∈ foilStructuralStates#[nStart:nDOF:end]
    #         stringData = @sprintf("%.8f\n", qⁿ)
    #         write(io, stringData)
    #     end
    # end
    # open("hydroTractions.dat", "w") do io
    #     stringData = elemType * "\n"
    #     write(io, stringData)
    #     if elemType == "COMP2"
    #         nDOF = 9
    #         nStart = 4
    #     elseif elemType == "BT2"
    #         nDOF = 4
    #         nStart = 3
    #     end
    #     for qⁿ ∈ hydroTractions#[nStart:nDOF:end]
    #         stringData = @sprintf("%.8f\n", qⁿ)
    #         write(io, stringData)
    #     end
    # end

    return hydroTractions, AIC, planformArea
end

# function integrate_hydroLoads(foilStructuralStates, fullAIC, DFOIL, elemType="BT2")
function integrate_hydroLoads(foilStructuralStates, fullAIC, α₀, elemType="BT2")
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
    ForceVector = (fullAIC * (foilTotalStates))


    if elemType == "bend-twist"
        nDOF = 3
        Moments = ForceVector[nDOF:nDOF:end]
    elseif elemType == "BT2"
        nDOF = 4
        Moments = ForceVector[3:nDOF:end]
        Lift = ForceVector[1:nDOF:end]
    elseif elemType == "COMP2"
        nDOF = 9
        Moments = ForceVector[4:nDOF:end]
        Lift = ForceVector[3:nDOF:end]
    else
        error("Invalid element type")
    end

    # --- Total dynamic hydro force calcs ---
    TotalLift = 0
    for secLift in Lift
        TotalLift += abs(secLift)
    end
    TotalMoment = 0
    for secMom in Moments
        TotalMoment += abs(secMom)
    end

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

