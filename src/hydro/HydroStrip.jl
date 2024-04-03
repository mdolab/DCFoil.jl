# --- Julia ---

# @File    :   HydroStrip.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines

module HydroStrip

# --- Public functions ---
export compute_theodorsen, compute_glauert_circ
export compute_node_mass, compute_node_damp, compute_node_stiff
export compute_AICs, apply_BCs

# --- PACKAGES ---
using SpecialFunctions
using LinearAlgebra
using Statistics
using Zygote, ChainRulesCore
using Printf, DelimitedFiles
using Plots
# using SparseArrays

# --- DCFoil modules ---
using ..SolverRoutines
using ..Unsteady: compute_theodorsen, compute_sears, compute_node_stiff_faster, compute_node_damp_faster, compute_node_mass
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE

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

# ==============================================================================
#                         Lift forces
# ==============================================================================
function compute_glauert_circ(semispan, chordVec, α₀, U∞, nNodes::Int64; h=nothing, useFS=false, rho=1000, twist=nothing, debug=false, config="wing")
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
        Fx_ind : float
            induced drag force
        CDi : float
            induced drag coefficient

    NOTE:
    We use keyword arguments (denoted by the ';' to be more explicit)
    THIS CODE DOES NOT WORK WITH ZERO ALPHA

    This follows the formulation in
    'Principles of Naval Architecture Series (PNA) - Propulsion 2010'
    by Justin Kerwin & Jacques Hadler
    """

    ỹ = π / 2 * ((1:1:nNodes) / nNodes) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
    y = -semispan * cos.(ỹ) # the physical coordinate (y) is only calculated to the root (-semispan, 0)
    # ---------------------------
    #   PLANFORM SHAPES: rectangular is outdated
    # ---------------------------
    # --- Actual chord ---
    chordₚ = chordVec
    # This does exhibit the correct tapering behavior where the vorticity increases outboard for more tapered geometry.
    # # --- Nearly "Elliptical" planform if the chordVec is all ones ---
    # chordₚ = chordVec .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?
    
    # --- Geometric twist effect ---
    if twist != nothing
        # First parametrize the twist as a function of ytilde
        xnew = cos.(ỹ) # cosine spacing
        xlin = LinRange(0.0, 1.0, nNodes) # linear spacing
        ftheta = SolverRoutines.do_linear_interp(xlin, twist, xnew)
        α₀ = α₀ .+ ftheta
    else
        α₀ = ones(nNodes) * α₀
    end
    n = (1:1:nNodes) * 2 - ones(nNodes) # node numbers x2 (node multipliers in the Fourier series)

    mu = π / 4 * (chordₚ / semispan)
    b = mu .* α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, nNodes]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = mu .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:nNodes*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) # matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the Fourier coefficients in Glauert's Fourier series ---
    ã = chord11 \ b

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise distribution of free vortex strength (Γ(y) in textbook)

    if useFS
        println("Using free surface")
        γ_FS = use_free_surface(γ, α₀, U∞, chordVec, h)
        γ = γ_FS
    end

    if config == "t-foil"
        γ[end] = 0.0
        # println("Zeroing out the root vortex strength")
    end

    cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl ./ (α₀ .+ 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    dl = semispan / (nNodes - 1)
    # xq::Vector{Float64} = collect(-semispan:dl:0)
    xq = LinRange(-semispan, 0.0, nNodes)

    cl_α = SolverRoutines.do_linear_interp(y, clα, xq)
    # If this is fully ventilated, can divide the slope by 4

    # if solverOptions["config"] == "t-foil"
    #     cl_α[end] = 0.0
    #     println("Zeroing out the root vortex strength")
    # end

    # ************************************************
    #     Lift-induced drag computation
    # ************************************************
    downwashDistribution = -U∞ * n .* (sin.(ỹn) * ã) ./ sin.(ỹ)
    wy = SolverRoutines.do_linear_interp(y, downwashDistribution, xq)

    Fx_ind::Float64 = 0.0
    dy::Float64 = semispan / (nNodes)
    spanwiseVorticity = SolverRoutines.do_linear_interp(y, γ, xq)
    
    for ii in 1:nNodes
        Fx_ind += -rho * spanwiseVorticity[ii] * wy[ii] * dy
    end

    # Assumes half wing drag
    CDi = Fx_ind / (0.5 * rho * U∞^2 * semispan * mean(chordVec))

    if debug
        println("Plotting debug hydro")
        layout = @layout [a b c]
        p1 = plot(y, γ./(4*U∞*semispan), label="γ(y)/2Us", xlabel="y", ylabel="γ(y)", title="Spanwise distribution")
        p2 = plot(y, wy./U∞, label="w(y)/Uinf", ylabel="w(y)", linecolor=:red)
        p3 = plot(y, cl, label="cl(y)", ylabel="cl(y)", linecolor=:green, ylim=(-1.0,1.0))
        plot(p1, p2, p3, layout=layout)
        savefig("spanwise_distribution.png")
    end

    return reverse(cl_α), Fx_ind, CDi
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

function use_free_surface(γ, α₀::Vector{Float64}, U∞, chordVec, h)
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
    γ_2DnoFS = -(U∞ * chordVec) .* (π * α₀)

    # Circulation with high-speed FS effect
    correctionVector = (1.0 .+ 16.0 .* (h ./ chordVec) .^ 2) ./ (2.0 .+ 16.0 .* (h ./ chordVec) .^ 2)
    γ_2DFS = -(U∞ * chordVec) .* (π * α₀) .* correctionVector

    # Corrected circulation
    γ_FS = γ .+ γ_2DnoFS .- γ_2DFS

    return γ_FS
end

function compute_desingularized_sources()
end

# function compute_spanwise_vortex(semispan, chordVec, α₀, U∞, nNodes, planform="elliptical")
#     """
#     Glauert's solution for the lift slope on a 3D hydrofoil

#     The coordinate system is

#     clamped root                         free tip
#     `+-----------------------------------------+  (x=0 @ LE)
#     `|                                         |
#     `|               +-->y                     |
#     `|               |                         |
#     `|             x v                         |
#     `+-----------------------------------------+
#     `
#     (y=0 @ root)

#     where z is out of the page (thickness dir.)
#     inputs:
#         α₀: float, angle of attack [rad]
#     """

#     ỹ = π / 2 * ((1:1:nNodes) / nNodes) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
#     y = -semispan * cos.(ỹ) # the physical coordinate (y) is only calculated to the root (-semispan, 0)

#     # ---------------------------
#     #   PLANFORM SHAPES: rectangular is outdated
#     # ---------------------------
#     # # --- Elliptical planform ---
#     if planform == "elliptical"
#         chordₚ = chordVec .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?
#     else
#         # --- Rectangular ---
#         chordₚ = chord

#     n = (1:1:nNodes) * 2 - ones(nNodes) # node numbers x2 (node multipliers)

#     mu = π / 4 * (chordₚ / semispan)
#     b = mu * α₀ .* sin.(ỹ) # RHS vector

#     ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

#     sinỹ_mat = repeat(sin.(ỹ), outer=[1, nNodes]) # parametrized square matrix where the columns go from 0 to 1
#     chord_ratio_mat = mu .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:nNodes*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

#     chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) # matrix-matrix multiplication to get the [A] matrix

#     # --- Solve for the Fourier coefficients in Glauert's Fourier series ---
#     ã = chord11 \ b

#     γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise distribution of free vortex strength (Γ(y) in textbook)

#     if useFS
#         γ_FS = use_free_surface(γ, α₀, U∞, chordVec, h)
#     end

#     cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
#     clα = cl / (α₀ + 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

#     # --- Interpolate lift slopes onto domain ---
#     dl = semispan / (nNodes - 1)
#     xq = -semispan:dl:0

#     cl_α = SolverRoutines.do_linear_interp(y, clα, xq)
#     cl = SolverRoutines.do_linear_interp(y, cl, xq)
#     gamma = SolverRoutines.do_linear_interp(y, γ, xq)
#     # If this is fully ventilated, can divide the slope by 4

#     downwashDistribution = -U∞ * n .* (sin.(ỹn) * ã) ./ sin.(ỹ)

#     return reverse(cl_α), reverse(cl), reverse(gamma)
# end

# ==============================================================================
#                         Static drag
# ==============================================================================


function compute_steady_AICs!(AIC, aeroMesh, chordVec, abVec, ebVec, Λ, FOIL, elemType="BT2")
    """
    Compute the steady aerodynamic influence coefficients (AICs) for a given mesh
    This is different from the general AIC method because there is no frequency dependence
    Inputs
    ------
    AIC: Matrix
        Aerodynamic influence coefficient matrix
    aeroMesh: Array
        Mesh of the foil (it's the same as the struct one)
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
        for jj in eachindex(aeroMesh[:,1])
            # --- compute strip width ---
            XN = aeroMesh[jj, :]
            yⁿ = XN[YDIM]
            zⁿ = XN[ZDIM]

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
            aprecomp = ab / b
            e_hh = 2 * clα # lift due to w'
            e_hα = -clα * b * (1 - aprecomp) # lift due to ψ'
            e_αh = clα * b * (1 + aprecomp) # moment due to w'
            e_αα = π * b^2 - 0.5 * clα * b^2 * (1 - (aprecomp*aprecomp)) # moment due to ψ'
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

            # jj += 1 # increment strip counter
        end
    end

    return AIC, planformArea
end

function compute_AICs(dim, aeroMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType="BT2"; appendageOptions=Dict{String,Any}("config"=>"wing"), STRUT=nothing, strutchordVec=nothing, strutabVec=nothing, strutebVec=nothing)
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
        Mesh of the foil (same as struct)
    stripVecs: 2d Array
        Spanwise tangent vectors for each strip
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
    # Is julia pass by reference or value?
    # It's pass by reference
    globalMf_z[:, :] = zeros(dim, dim)
    globalCf_r_z[:, :] = zeros(dim, dim)
    globalCf_i_z[:, :] = zeros(dim, dim)
    globalKf_r_z[:, :] = zeros(dim, dim)
    globalKf_i_z[:, :] = zeros(dim, dim)
    # --- Initialize planform area counter ---
    planformArea = 0.0

    jj = 1 # node index
    # nElemWing = solverOptions["nNodes"] - 1
    # nElemStrut = solverOptions["nNodeStrut"] - 1
    nElemWing = length(chordVec) - 1
    # Bit circular logic here
    appendageOptions["nNodes"] = nElemWing+1
    if STRUT != nothing
        nElemStrut = length(strutchordVec) - 1
        appendageOptions["nNodeStrut"] = nElemStrut+1
    end
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
        # for yⁿ in aeroMesh[:, YDIM]
        stripVecs = get_strip_vecs(aeroMesh, elemConn, appendageOptions)
        junctionNodeX = aeroMesh[1, :]

        for inode in eachindex(aeroMesh[:,1]) # loop aero strips (located at FEM nodes)
            # @inbounds begin
            # --- compute strip quantities ---
            XN = aeroMesh[inode, :]
            # TODO: redo y^n as a parametric coordinate along lifting surface
            yⁿ = XN[YDIM]
            zⁿ = XN[ZDIM]

            Δy = 0.0
            # if inode < FOIL.nNodes
            #     nVec = (aeroMesh[inode+1, :] - aeroMesh[inode, :])
            # else
            #     nVec = (aeroMesh[inode, :] - aeroMesh[inode-1, :])
            # end
            nVec = stripVecs[inode,:]

            # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change
            lᵉ::Float64 = sqrt(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # length of elem
            Δy = lᵉ
            # If we have end point nodes, we need to divide the length by 2
            if appendageOptions["config"] == "wing"
                if inode == 1 || inode == FOIL.nNodes
                    Δy = 0.5 * lᵉ
                end
            elseif appendageOptions["config"] == "full-wing"
                if inode == 1 || inode == FOIL.nNodes || (inode == nElemWing * 2 + 1)
                    Δy = 0.5 * lᵉ
                end
            elseif appendageOptions["config"] == "t-foil"
                if inode == 1 || inode == FOIL.nNodes || (inode == nElemWing * 2 + 1) || (inode == nElemWing * 2 + nElemStrut + 1)
                    Δy = 0.5 * lᵉ
                end
            else
                error("Invalid configuration")
            end

            nVec = nVec / lᵉ # normalize
            dR1 = nVec[XDIM]
            dR2 = nVec[YDIM]
            dR3 = nVec[ZDIM]

            # --- Linearly interpolate values based on y loc ---
            # THis chunk of code is super hacky based on assuming wing and t-foil strut order
            if inode <= FOIL.nNodes
                xDom = aeroMesh[1:FOIL.nNodes, YDIM]
                clα = SolverRoutines.do_linear_interp(xDom, FOIL.clα, yⁿ)
                c = SolverRoutines.do_linear_interp(xDom, chordVec, yⁿ)
                ab = SolverRoutines.do_linear_interp(xDom, abVec, yⁿ)
                eb = SolverRoutines.do_linear_interp(xDom, ebVec, yⁿ)
            else
                if appendageOptions["config"] == "t-foil"
                    if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                        # Put negative sign on the linear interp routine bc there is a bug!
                        xDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM] )
                        yⁿ = -1 * yⁿ
                    
                        clα = SolverRoutines.do_linear_interp(xDom, FOIL.clα, yⁿ)
                        c = SolverRoutines.do_linear_interp(xDom, chordVec, yⁿ)
                        ab = SolverRoutines.do_linear_interp(xDom, abVec, yⁿ)
                        eb = SolverRoutines.do_linear_interp(xDom, ebVec, yⁿ)
                        # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                        dR1 = -dR1
                        dR2 = -dR2
                        dR3 = -dR3
                        # println("I'm a port wing strip")
                    else
                        xDom = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
                        clα = SolverRoutines.do_linear_interp(xDom, STRUT.clα, zⁿ)
                        c = SolverRoutines.do_linear_interp(xDom, strutchordVec, zⁿ)
                        ab = SolverRoutines.do_linear_interp(xDom, strutabVec, zⁿ)
                        eb = SolverRoutines.do_linear_interp(xDom, strutebVec, zⁿ)
                        # println("I'm a strut strip")
                    end
                elseif appendageOptions["config"] == "full-wing"
                    if inode <= nElemWing * 2 + 1
                        # Put negative sign on the linear interp routine bc there is a bug!
                        xDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM] )
                        yⁿ = -1 * yⁿ

                        clα = SolverRoutines.do_linear_interp(xDom, FOIL.clα, yⁿ)
                        c = SolverRoutines.do_linear_interp(xDom, chordVec, yⁿ)
                        ab = SolverRoutines.do_linear_interp(xDom, abVec, yⁿ)
                        eb = SolverRoutines.do_linear_interp(xDom, ebVec, yⁿ)
                        # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                        dR1 = -dR1
                        dR2 = -dR2
                        dR3 = -dR3
                    end
                end
            end
            b = 0.5 * c # semichord for more readable code

            # --- Precomputes ---
            clambda = cos(Λ)
            slambda = sin(Λ)
            k = ω * b / (U∞ * clambda) # local reduced frequency
            # Do Theodorsen computation once for efficiency
            if abs(ω) <= MEPSLARGE
                Ck = 1.0
            else
                CKVec = compute_theodorsen(k)
                Ck = CKVec[1] + 1im * CKVec[2]
            end

            K_f, K̂_f = compute_node_stiff_faster(clα, b, eb, ab, U∞, clambda, slambda, FOIL.ρ_f, Ck)
            C_f, Ĉ_f = compute_node_damp_faster(clα, b, eb, ab, U∞, clambda, slambda, FOIL.ρ_f, Ck)
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
                # # NOTE: Done in local aero coordinates which matches the local beam coordinates
                # KLocal = zeros(9, 9)
                # CLocal = zeros(9, 9)
                # MLocal = zeros(Float64, 9, 9)
                # KLocal[3:4,4:7] = [
                #     K_f[1, 2] K̂_f[1, 1] 0.0 K̂_f[1, 2]
                #     K_f[2, 2] K̂_f[2, 1] 0.0 K̂_f[2, 2]
                # ]
                # CLocal[3:4,3:7] = [
                #     C_f[1, 1] C_f[1, 2] Ĉ_f[1, 1] 0.0 Ĉ_f[1, 2]
                #     C_f[2, 1] C_f[2, 2] Ĉ_f[2, 1] 0.0 Ĉ_f[2, 2]
                # ]
                # MLocal[3:4,3:4] = [
                #     M_f[1, 1] M_f[1, 2]
                #     M_f[2, 1] M_f[2, 2]
                # ]
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
            Γ = SolverRoutines.get_transMat(dR1, dR2, dR3, 1.0, elemType)
            # DOUBLE CHECK IF THIS TRANSFORMATION IS CORRECT FOR PORT WING
            # I think yes
            KLocal = Γ'[1:nDOF, 1:nDOF] * KLocal * Γ[1:nDOF, 1:nDOF]
            CLocal = Γ'[1:nDOF, 1:nDOF] * CLocal * Γ[1:nDOF, 1:nDOF]
            MLocal = Γ'[1:nDOF, 1:nDOF] * MLocal * Γ[1:nDOF, 1:nDOF]

            GDOFIdx::Int64 = nDOF * (inode - 1) + 1

            # Add local AIC to global AIC and remember to multiply by strip width to get the right result
            globalKf_r_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(KLocal) * Δy
            globalKf_i_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(KLocal) * Δy
            globalCf_r_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = real(CLocal) * Δy
            globalCf_i_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = imag(CLocal) * Δy
            globalMf_z[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = MLocal * Δy

            # Add rectangle to planform area
            if inode <= FOIL.nNodes
                planformArea += c * Δy
            # else
                # println("Not adding planform area for strut or mirrored wing")
            end
            # inode += 1 # increment strip counter
            # end # inbounds
        end
    end

    return copy(globalMf_z), copy(globalCf_r_z), copy(globalCf_i_z), copy(globalKf_r_z), copy(globalKf_i_z), planformArea
end

function get_strip_vecs(aeroMesh, elemConn, solverOptions)
    """
    Parameters
    ----------
    aeroMesh: array
        The aerodynamic mesh
    elemConn: array
        The element connectivity from the structural mesh
    """

    
    nStrips = size(aeroMesh)[1]

    stripVecs = zeros(nStrips, 3)
    stripVecs_z = Zygote.Buffer(stripVecs)
    stripVecs_z[:,:] = stripVecs
    nElemWing = solverOptions["nNodes"] - 1
    if haskey(solverOptions, "nNodeStrut")
        nElemStrut = solverOptions["nNodeStrut"] - 1
    else
        nElemStrut = 0
    end
    for istrip in 1:nStrips-1 # loop elements but these are filling aero strips (nodes)
        n1 = elemConn[istrip, 1]
        n2 = elemConn[istrip, 2]
        if istrip > 1 # apply tangency continuation for wing and strut
            if istrip == nElemWing + 1 || istrip == nElemWing*2 + 1 || istrip == nElemWing*2 + nElemStrut + 1
                n1 = elemConn[istrip-1, 1]
                n2 = elemConn[istrip-1, 2]
            end
        end

        nvec = aeroMesh[n2, :] - aeroMesh[n1, :]
        stripVecs_z[istrip, :] = nvec

    end

    # Treat last strip separately
    if size(elemConn)[1] == 3 || size(elemConn)[1] == 2
        stripVecs_z[end,:] = aeroMesh[end, :] - aeroMesh[1, :]
    else
        # if solverOptions["debug"]
        #     println("treating last strip (node) tangent vector as a continuation of previous node")
        # end
        stripVecs_z[end,:] = aeroMesh[end, :] - aeroMesh[end-1, :]
    end

    return copy(stripVecs_z)
end

# function ChainRulesCore.rrule(::typeof(compute_AICs), dim, aeroMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType="BT2"; config="wing", STRUT=nothing)

#     y = compute_AICs(dim, aeroMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType, config, STRUT)

#     function compute_AICs_pullback(y)
        # return
#     end

#     return y, compute_AICs_pullback
# end

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
    # Mf, Cf_r,Cf_i, Kf_r, Kf_i, planformArea = compute_AICs(dim, aeroMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, 0.0,)
    _, planformArea = compute_steady_AICs!(AIC, mesh, chordVec, abVec, ebVec, Λ, FOIL, elemType)

    # --- Compute fluid tractions ---
    hydroTractions = -1 * AIC * foilTotalStates # aerodynamic forces are on the RHS so we negate
    # hydroTractions = Kf_r * foilTotalStates # aerodynamic forces are on the RHS so we negate

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
    #         stringData = @sprintf("%.16f\n", qⁿ)
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
    #         stringData = @sprintf("%.16f\n", qⁿ)
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
    #         stringData = @sprintf("%.16f\n", qⁿ)
    #         write(io, stringData)
    #     end
    # end

    return hydroTractions, AIC, planformArea
end

function compute_genHydroLoadsMatrices(kMax, nk::Int64, U∞, b_ref, dim::Int64, structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, elemType)
    """
    Computes the hydrodynamic coefficients for a sweep of reduced frequencies

    Inputs
    ------
        kMax: maximum reduced frequency
        nk: number of reduced frequencies
        U∞: freestream velocity
    """

    kSweep_z = Zygote.Buffer(zeros(nk))

    for ii in 1:nk
        # Current value in a linspace from zero to one
        dist = (ii - 1.0) / (nk - 1)
        # Cubic distribution to bunch more points at lower k values
        if ii == 1
            kSweep_z[ii] = 1e-13
        else
            kSweep_z[ii] = kMax * dist * dist * dist
        end
    end
    kSweep = copy(kSweep_z)

    # --- Loop over reduced frequencies ---
    Cf_r_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Cf_i_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Kf_r_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Kf_i_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Mf_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    ii = 1
    for k in kSweep
        ω = k * U∞ * (cos(Λ)) / b_ref

        # Compute AIC
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(dim, structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType)

        # Accumulate in frequency sweep matrix
        # @inbounds begin
        Cf_r_sweep_z[:, :, ii] = globalCf_r
        Cf_i_sweep_z[:, :, ii] = globalCf_i
        Kf_r_sweep_z[:, :, ii] = globalKf_r
        Kf_i_sweep_z[:, :, ii] = globalKf_i
        Mf_sweep_z[:, :, ii] = globalMf
        # end
        ii += 1
    end

    return copy(Mf_sweep_z[:,:,1]), copy(Cf_r_sweep_z), copy(Cf_i_sweep_z), copy(Kf_r_sweep_z), copy(Kf_i_sweep_z), kSweep
end

function integrate_hydroLoads(foilStructuralStates, fullAIC, α₀, rake::Float64, elemType="BT2", config="wing"; appendageOptions=Dict(), solverOptions=Dict())
    """
    Inputs
    ------
        fullAIC: AIC matrix which in the DCFoil code base is Kf even though it's normally -Kf
        α₀: base angle of attack
        rake: rake angle
        FOIL: FOIL struct
        elemType: element type
    Returns
    -------
        force vector
        abs val of total lift and moment (needed for dynamic mode since they are complex)
    """

    # --- Initializations ---
    # This is dynamic deflection + rigid shape of foil
    foilTotalStates, nDOF = SolverRoutines.return_totalStates(foilStructuralStates, α₀, rake, elemType; appendageOptions=appendageOptions)

    # --- Strip theory ---
    # This is the hydro force traction vector
    # The problem is the full AIC matrix build (RHS). These states look good
    # fhydro RHS = -Kf * states
    ForceVector = -(fullAIC * (foilTotalStates))


    if elemType == "bend-twist"
        nDOF = 3
        My = ForceVector[nDOF:nDOF:end]
    elseif elemType == "BT2"
        nDOF = 4
        My = ForceVector[3:nDOF:end]
        Lift = ForceVector[1:nDOF:end]
    elseif elemType == "COMP2"
        nDOF = 9
        My = ForceVector[4:nDOF:end]
        Fz = ForceVector[3:nDOF:end]
        Fy = ForceVector[2:nDOF:end]
    else
        error("Invalid element type")
    end

    ChainRulesCore.ignore_derivatives() do
        if solverOptions["debug"]
            println("Plotting hydrodynamic loads")
            plot(1:length(Fz), Fz, label="Fz")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Lift (N/m)")
            fname = @sprintf("./DebugOutput/hydroloads_lift.png")
            savefig(fname)

            plot(1:length(My),My, label="Moment")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Moment (Nm/m)")
            fname = @sprintf("./DebugOutput/hydroloads_moments.png")
            savefig(fname)
        end
    end

    # --- Total dynamic hydro force calcs ---
    AbsTotalLift::Float64 = 0
    for secLift in Fz
        AbsTotalLift += abs(secLift)
    end
    AbsTotalMoment::Float64 = 0
    for secMom in My
        AbsTotalMoment += abs(secMom)
    end

    return ForceVector, AbsTotalLift, AbsTotalMoment
end

function apply_BCs(K, C, M, globalDOFBlankingList::UnitRange{Int64})
    """
    Applies BCs for nodal displacements
    """

    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newC = C[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]

    return newK, newC, newM
end


end # end module
