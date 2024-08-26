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
using FLOWMath: norm_cs_safe
# using SparseArrays

# --- DCFoil modules ---
using ..SolverRoutines
using ..Unsteady: compute_theodorsen, compute_sears, compute_node_stiff_faster, compute_node_damp_faster, compute_node_mass
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE, GRAV
using ..EBBeam: EBBeam as BeamElement, NDOF
using ..DCFoil: RealOrComplex, DTYPE

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
function compute_glauert_circ(
    semispan, chordVec, α₀, U∞, nNodes;
    h=nothing, useFS=false, rho=1000, twist=nothing, debug=false, config="wing"
)
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

    nNodes = length(chordVec)

    ỹ = π / 2 * (LinRange(1, nNodes, nNodes) / nNodes) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
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

    # Matrix
    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, nNodes]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = mu .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:nNodes*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) # matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the Fourier coefficients in Glauert's Fourier series ---
    ã = chord11 \ b # a_n

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise distribution of free vortex strength (Γ(y) in textbook)
    if config == "t-foil"
        γ = vcat(copy(γ[1:end-1]), 0.0)
        # println("Zeroing out the root vortex strength")
    end

    if useFS
        # ChainRulesCore.ignore_derivatives() do
        #     println("Using free surface")
        # end
        γ_FS = use_free_surface(γ, α₀, U∞, chordVec, h)
        γ = γ_FS
    end


    cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl ./ (α₀ .+ 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    # dl = semispan / (nNodes - 1)
    xq = LinRange(-semispan, 0.0, nNodes)

    cℓ = SolverRoutines.do_linear_interp(y, cl, xq)
    cl_α = SolverRoutines.do_linear_interp(y, clα, xq)
    # If this is fully ventilated, can divide the slope by 4

    # if solverOptions["config"] == "t-foil"
    #     cl_α[end] = 0.0
    #     println("Zeroing out the root vortex strength")
    # end

    # ************************************************
    #     Lift-induced drag computation
    # ************************************************
    downwashDistribution = -U∞ * (sin.(ỹn) * (n .* ã)) ./ sin.(ỹ)
    wy = SolverRoutines.do_linear_interp(y, downwashDistribution, xq)

    Fx_ind = 0.0
    dy = semispan / (nNodes)
    spanwiseVorticity = SolverRoutines.do_linear_interp(y, γ, xq)

    for ii in 1:nNodes
        Fx_ind += -rho * spanwiseVorticity[ii] * wy[ii] * dy
    end

    # Assumes half wing drag
    CDi = Fx_ind / (0.5 * rho * U∞^2 * semispan * mean(chordVec))

    if debug
        println("Plotting debug hydro")
        layout = @layout [a b c]
        p1 = plot(y, γ ./ (4 * U∞ * semispan), label="γ(y)/2Us", xlabel="y", ylabel="γ(y)", title="Spanwise distribution")
        p2 = plot(y, wy ./ U∞, label="w(y)/Uinf", ylabel="w(y)/Uinf", linecolor=:red)
        p3 = plot(y, cl, label="cl(y)", ylabel="cl(y)", linecolor=:green, ylim=(-1.0, 1.0))
        plot(p1, p2, p3, layout=layout)
        savefig("debug_spanwise_distribution.png")
    end

    return reverse(cl_α), Fx_ind, CDi
end

function correct_downwash(
    iComp::Int64, CLMain::DTYPE, DVDictList, solverOptions
)
    """
    """
    DVDict = DVDictList[iComp]
    Uinf = solverOptions["Uinf"]
    depth = DVDict["depth0"]
    xM = solverOptions["appendageList"][1]["xMount"]
    xR = solverOptions["appendageList"][iComp]["xMount"]
    ℓᵣ = xR - xM # distance from main wing AC to downstream wing AC, +ve downstream
    upstreamDict = DVDictList[1]
    sWing = upstreamDict["s"]
    cRefWing = sum(upstreamDict["c"]) / length(upstreamDict["c"])
    chordMMean = cRefWing
    # ChainRulesCore.ignore_derivatives() do
    #     if solverOptions["debug"]
    #         println(@sprintf("=========================================================================="))
    #         println(@sprintf("Computing downstream flow effects with ℓᵣ = %.2f m, C_L_M = %.1f ", ℓᵣ, CLMain))
    #     end
    # end

    # --- Compute the wake effect ---
    αiWake = compute_wakeDWAng(sWing, cRefWing, CLMain, ℓᵣ)

    # --- Compute the wave pattern effect ---
    Fnc = Uinf / (sqrt(GRAV * chordMMean))
    Fnh = Uinf / (sqrt(GRAV * depth))
    αiWave = compute_wavePatternDWAng(CLMain, chordMMean, Fnc, Fnh, ℓᵣ)

    # --- Correct the downwash ---
    alphaCorrection = αiWake .+ αiWave

    return alphaCorrection
end

function compute_wakeDWAng(sWing, cRefWing, CLWing, ℓᵣ)
    """
    Assume the downstream lifting surface is behind an elliptically loaded wing
    ℓᵣ = xM + xR downstream

    Inputs
    ------
    sWing: float
        span of the upstream wing
    cRefWing: float
        ref chord of the upstream wing
    CLWing: float
        Lift coefficient of the upstream wing
    ℓᵣ: float
        Distance from main wing AC to downstream wing AC, +ve downstream
    """

    ARwing = sWing / cRefWing

    l_div_s = ℓᵣ / sWing
    # THIS IS WRONG
    kappa = 1 + 1 / (sqrt(1 + (l_div_s)^2)) * (1 / (π * l_div_s) + 1)
    kappa = 2.0
    # println("k is", k)
    k = 1 / sqrt(1 + (l_div_s)^2)

    Ek = SpecialFunctions.ellipe(k^2)

    kappa = 1 + 2 / π * Ek / sqrt(1 - k^2)

    ε = kappa * CLWing / (π * ARwing)

    return -ε
end

function compute_wavePatternDWAng(clM, chordM, Fnc, Fnh, ξ)
    """
    Compute 2D wave pattern effect (transverse grav waves only)

    Inputs
    ------
    clM: vector
        Spanwise lift coefficient of the main wing
    chordM: vector
        Ref chord of the main wing
    Fnc: float
        Chord-based Froude number of the main wing
    Fnh: float
        Depth-based Froude number of the main wing
    ξ - Distance from the main wing to the downstream wing
    """

    divFncsq = 1 / (Fnc * Fnc)
    premult = divFncsq * exp(-2 / (Fnh * Fnh))

    # Vectorized computation
    αiwave = -clM .* cos.(divFncsq .* ξ ./ chordM) .* premult

    return αiwave
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
    if real(Fnh) < real(10 / sqrt(h / mean(chordVec)))
        println("Violating high-speed free-surface BC with Fnh*sqrt(h/c) of")
        println(Fnh * sqrt(h / mean(chordVec)))
        println("Fnh: $(Fnh)")
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

# ==============================================================================
#                         Static drag
# ==============================================================================
function compute_AICs(
    AEROMESH, FOIL, dim, Λ, U∞, ω, elemType="BT2";
    appendageOptions=Dict{String,Any}("config" => "wing"), STRUT=nothing
)
    """
    Compute the AIC matrix for a given aeroMesh using LHS convention
        (i.e., -ve force is disturbing, not restoring)
    Inputs
    ------
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
        Aerodynamic influence coefficient matrix broken up into added mass, damping, and stiffness
        in such a way that
            {F} = -([Mf]{udd} + [Cf]{ud} + [Kf]{u})
        These are matrices
        in the global reference frame
    """

    aeroMesh = AEROMESH.mesh
    # elemConn = AEROMESH.elemConn

    # --- Initialize global matrices ---
    globalMf_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalCf_r_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalCf_i_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalKf_r_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalKf_i_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    # Zygote initialization
    globalMf_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalCf_r_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalCf_i_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalKf_r_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalKf_i_z[:, :] = zeros(RealOrComplex, dim, dim)

    # --- Initialize planform area counter ---
    planformArea = 0.0
    chordVec = FOIL.chord
    abVec = FOIL.ab
    ebVec = FOIL.eb
    clαVec = FOIL.clα

    if STRUT != nothing
        strutclαVec = STRUT.clα
        strutChordVec = STRUT.chord
        strutabVec = STRUT.ab
        strutebVec = STRUT.eb
    end

    jj = 1 # node index
    # nElemWing = solverOptions["nNodes"] - 1
    # nElemStrut = solverOptions["nNodeStrut"] - 1
    nElemWing = length(chordVec) - 1
    # Bit circular logic here
    appendageOptions["nNodes"] = nElemWing + 1
    if STRUT != nothing
        nElemStrut = length(strutChordVec) - 1
        appendageOptions["nNodeStrut"] = nElemStrut + 1
    end
    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    if ndims(aeroMesh) == 2
        # for yⁿ in aeroMesh[:, YDIM]
        stripVecs = get_strip_vecs(AEROMESH, appendageOptions)
        junctionNodeX = aeroMesh[1, :]

        for inode in eachindex(aeroMesh[:, 1]) # loop aero strips (located at FEM nodes)
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
            nVec = stripVecs[inode, :]

            # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change
            lᵉ = sqrt(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # length of elem
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
                clα = SolverRoutines.do_linear_interp(xDom, clαVec, yⁿ)
                c = SolverRoutines.do_linear_interp(xDom, chordVec, yⁿ)
                ab = SolverRoutines.do_linear_interp(xDom, abVec, yⁿ)
                eb = SolverRoutines.do_linear_interp(xDom, ebVec, yⁿ)
            else
                if appendageOptions["config"] == "t-foil"
                    if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                        # Put negative sign on the linear interp routine bc there is a bug!
                        xDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                        yⁿ = -1 * yⁿ

                        clα = SolverRoutines.do_linear_interp(xDom, clαVec, yⁿ)
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
                        clα = SolverRoutines.do_linear_interp(xDom, strutclαVec, zⁿ)
                        c = SolverRoutines.do_linear_interp(xDom, strutChordVec, zⁿ)
                        ab = SolverRoutines.do_linear_interp(xDom, strutabVec, zⁿ)
                        eb = SolverRoutines.do_linear_interp(xDom, strutebVec, zⁿ)
                        # println("I'm a strut strip")
                    end
                elseif appendageOptions["config"] == "full-wing"
                    if inode <= nElemWing * 2 + 1
                        # Put negative sign on the linear interp routine bc there is a bug!
                        xDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                        yⁿ = -1 * yⁿ

                        clα = SolverRoutines.do_linear_interp(xDom, clαVec, yⁿ)
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
            KLocal = Γ'[1:NDOF, 1:NDOF] * KLocal * Γ[1:NDOF, 1:NDOF]
            CLocal = Γ'[1:NDOF, 1:NDOF] * CLocal * Γ[1:NDOF, 1:NDOF]
            MLocal = Γ'[1:NDOF, 1:NDOF] * MLocal * Γ[1:NDOF, 1:NDOF]

            GDOFIdx::Int64 = NDOF * (inode - 1) + 1

            # Add local AIC to global AIC and remember to multiply by strip width to get the right result
            globalKf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(KLocal) * Δy
            globalKf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(KLocal) * Δy
            globalCf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(CLocal) * Δy
            globalCf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(CLocal) * Δy
            globalMf_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = MLocal * Δy

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

function get_strip_vecs(
    AEROMESH, solverOptions
)
    """
    Parameters
    ----------
    aeroMesh: array
        The aerodynamic mesh
    elemConn: array
        The element connectivity from the structural mesh
    """
    aeroMesh = AEROMESH.mesh
    elemConn = AEROMESH.elemConn

    nStrips = size(aeroMesh)[1]

    stripVecs = zeros(RealOrComplex, nStrips, 3)
    stripVecs_z = Zygote.Buffer(stripVecs)
    stripVecs_z[:, :] = stripVecs
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
            if istrip == nElemWing + 1 || istrip == nElemWing * 2 + 1 || istrip == nElemWing * 2 + nElemStrut + 1
                n1 = elemConn[istrip-1, 1]
                n2 = elemConn[istrip-1, 2]
            end
        end

        nvec = aeroMesh[n2, :] - aeroMesh[n1, :]
        stripVecs_z[istrip, :] = nvec

    end

    # Treat last strip separately
    if size(elemConn)[1] == 3 || size(elemConn)[1] == 2
        stripVecs_z[end, :] = aeroMesh[end, :] - aeroMesh[1, :]
    else
        # if solverOptions["debug"]
        #     println("treating last strip (node) tangent vector as a continuation of previous node")
        # end
        stripVecs_z[end, :] = aeroMesh[end, :] - aeroMesh[end-1, :]
    end

    return copy(stripVecs_z)
end

function compute_genHydroLoadsMatrices(kMax, nk, U∞, b_ref, dim, AEROMESH, Λ, FOIL, elemType)
    """
    Computes the hydrodynamic coefficients for a sweep of reduced frequencies

    Inputs
    ------
        kMax: maximum reduced frequency
        nk: number of reduced frequencies
        U∞: freestream velocity
    """

    linDist = ((LinRange(1, nk, nk)) .- 1) / (nk - 1)
    cubicDist = linDist[2:end] .^ 3 # cubic distribution removing the first point
    kSweep = vcat([1e-13], kMax .* cubicDist)

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
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(AEROMESH, FOIL, dim, Λ, U∞, ω, elemType)

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

    return copy(Mf_sweep_z[:, :, 1]), copy(Cf_r_sweep_z), copy(Cf_i_sweep_z), copy(Kf_r_sweep_z), copy(Kf_i_sweep_z), kSweep
end

function integrate_hydroLoads(
    foilStructuralStates, fullAIC, α₀, rake, dofBlank::Vector{Int64}, downwashAngles::DTYPE, elemType="BT2";
    appendageOptions=Dict(), solverOptions=Dict()
)
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
    DVDict = Dict(
        "alfa0" => α₀,
        "rake" => rake,
        "beta" => 0.0,
    )
    foilTotalStates = SolverRoutines.return_totalStates(foilStructuralStates, DVDict, elemType;
        appendageOptions=appendageOptions, alphaCorrection=downwashAngles)

    # --- Strip theory ---
    # This is the hydro force traction vector
    # The problem is the full AIC matrix build (RHS). These states look good
    # fhydro RHS = -Kf * states
    ForceVector = zeros(DTYPE, length(foilTotalStates))
    ForceVector_z = Zygote.Buffer(zeros(DTYPE, length(foilTotalStates)))
    ForceVector_z[:] = ForceVector
    # Only compute forces not at the blanked BC node
    F = -(fullAIC[1:end.∉[dofBlank], 1:end.∉[dofBlank]] * (foilTotalStates[1:end.∉[dofBlank]]))
    ForceVector_z[1:length(foilTotalStates).∉[dofBlank]] = F
    ForceVector = copy(ForceVector_z)

    nDOF = BeamElement.NDOF
    if elemType == "bend-twist"
        My = ForceVector[nDOF:nDOF:end]
    elseif elemType == "BT2"
        My = ForceVector[3:nDOF:end]
        Lift = ForceVector[1:nDOF:end]
    elseif elemType == "COMP2"
        My = ForceVector[4:nDOF:end]
        Fz = ForceVector[3:nDOF:end]
        Fy = ForceVector[2:nDOF:end]
    else
        error("Invalid element type")
    end

    ChainRulesCore.ignore_derivatives() do
        if solverOptions["debug"]
            config = appendageOptions["config"]
            println("Plotting hydrodynamic loads")
            plot(1:length(Fz), Fz, label="Fz")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Lift (N/m)")
            fname = @sprintf("./DebugOutput/hydroloads_lift.png")
            savefig(fname)

            plot(1:length(My), My, label="Moment")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Moment (Nm/m)")
            fname = @sprintf("./DebugOutput/hydroloads_moments.png")
            savefig(fname)
        end
    end

    # --- Total dynamic hydro force calcs ---
    AbsTotalLift = 0.0
    for secLift in Fz
        AbsTotalLift += abs(secLift)
    end
    AbsTotalMoment = 0.0
    for secMom in My
        AbsTotalMoment += abs(secMom)
    end

    return ForceVector, AbsTotalLift, AbsTotalMoment
end

function apply_BCs(K, C, M, globalDOFBlankingList)
    """
    Applies BCs for nodal displacements
    """

    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newC = C[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]

    return newK, newC, newM
end


end # end module
