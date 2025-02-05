# --- Julia 1.10---
"""
@File          :   GlauertLL.jl
@Date created  :   2024/10/08
@Last modified :   2024/10/08
@Author        :   Galen Ng
@Desc          :   The original Glauert lifting line method used in Akcabay's papers
"""
module GlauertLL

# --- PACKAGES ---
using Statistics
# --- DCFoil modules ---
using ..SolverRoutines
using ..Interpolation

function compute_glauert_circ(
    semispan, chordVec, α₀, U∞;
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
        ftheta = Interpolation.do_linear_interp(xlin, twist, xnew)
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
        # @ignore_derivatives() do
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

    cℓ = Interpolation.do_linear_interp(y, cl, xq)
    cl_α = Interpolation.do_linear_interp(y, clα, xq)
    # If this is fully ventilated, can divide the slope by 4

    # if solverOptions["config"] == "t-foil"
    #     cl_α[end] = 0.0
    #     println("Zeroing out the root vortex strength")
    # end

    # ************************************************
    #     Lift-induced drag computation
    # ************************************************
    downwashDistribution = -U∞ * (sin.(ỹn) * (n .* ã)) ./ sin.(ỹ)
    wy = Interpolation.do_linear_interp(y, downwashDistribution, xq)

    Fx_ind = 0.0
    dy = semispan / (nNodes)
    spanwiseVorticity = Interpolation.do_linear_interp(y, γ, xq)

    for ii in 1:nNodes
        Fx_ind += -rho * spanwiseVorticity[ii] * wy[ii] * dy
    end

    # Assumes half wing drag
    CDi = Fx_ind / (0.5 * rho * U∞^2 * semispan * mean(chordVec))

    # if debug
    #     println("Plotting debug hydro")
    #     layout = @layout [a b; c d]
    #     p1 = plot(y, γ ./ (4 * U∞ * semispan), label="γ(y)/2Us", xlabel="y", ylabel="γ(y)", title="Spanwise distribution")
    #     p2 = plot(y, wy ./ U∞, label="w(y)/Uinf", ylabel="w(y)/Uinf", linecolor=:red)
    #     p3 = plot(y, cl, label="cl(y)", ylabel="cl(y)", linecolor=:green, ylim=(-1.0, 1.0))
    #     p4 = plot(y, cl_α, label="cl_α(y)", ylabel="cl_α(y)", linecolor=:blue)
    #     plot(p1, p2, p3, p4, layout=layout)
    #     savefig("DebugOutput/debug_spanwise_distribution.png")
    # end

    return reverse(cl_α), Fx_ind, CDi
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

    Fnh = U∞ / (√(9.81 * h))
    # Find limiting case
    if real(Fnh) < real(10 / √(h / mean(chordVec)))
        println("Violating high-speed free-surface BC with Fnh*√(h/c) of")
        println(Fnh * √(h / mean(chordVec)))
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


end
