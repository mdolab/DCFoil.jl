# --- Julia ---
"""
@File    :   SolveFlutter.jl
@Time    :   2022/10/07
@Author  :   Galen Ng
@Desc    :   p-k method for flutter analysis
"""

module SolveFlutter
"""
Eigenvalue and eigenvector solution
"""

# --- Public functions ---
export solve

# --- Libraries ---
using LinearAlgebra, Statistics
using JSON
using Zygote
using Profile


# --- DCFoil modules ---
include("../InitModel.jl")
include("../struct/BeamProperties.jl")
include("../struct/FiniteElements.jl")
include("../hydro/Hydro.jl")
include("SolveStatic.jl")
include("../constants/SolutionConstants.jl")
include("./SolverRoutines.jl")
# then use them
using .InitModel, .Hydro, .StructProp
using .FEMMethods
using .SolveStatic
using .SolutionConstants
using .SolverRoutines

function solve(DVDict::Dict, outputDir::String, uSweep::StepRangeLen{Float64,Base.TwicePrecision{Float64}}, fSearch::StepRangeLen{Float64,Base.TwicePrecision{Float64}}; use_freeSurface=false, cavitation=nothing)
    """
    Use p-k method to find roots (p) to the equation
        (-p²[M]-p[C]+[K]){ũ} = {0}
    """

    # ************************************************
    #     Initialize
    # ************************************************
    global FOIL = InitModel.init_dynamic(fSearch, DVDict, uSweep=uSweep)
    nElem = FOIL.neval - 1

    println("====================================================================================")
    println("        BEGINNING FLUTTER SOLUTION")
    println("====================================================================================")
    # ---------------------------
    #   Assemble structure
    # ---------------------------
    elemType = "BT2"
    loadType = "force"

    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)
    globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, FOIL.constitutive)
    FEMMethods.apply_tip_load!(globalF, elemType, loadType)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType, "clamped")
    Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

    # --- Initialize stuff ---
    u = copy(globalF)
    # globalMf = copy(globalMs) * 0
    # globalCf_r = copy(globalKs) * 0
    # globalKf_r = copy(globalKs) * 0
    # globalKf_i = copy(globalKs) * 0
    # globalCf_i = copy(globalKs) * 0
    # extForceVec = copy(F) * 0 # this is a vector excluded the BC nodes
    # extForceVec[end-1] = tipForceMag # this is applying a tip twist
    # LiftDyn = zeros(length(fSweep)) # * 0im
    # MomDyn = zeros(length(fSweep)) # * 0im
    # TipBendDyn = zeros(length(fSweep)) # * 0im
    # TipTwistDyn = zeros(length(fSweep)) # * 0im

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(Ks, Ms, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q

    # ************************************************
    #     For every flow speed, solve for the 'p' roots
    # ************************************************
    q_ctr = 1
    b_ref = Statistics.mean(FOIL.c) # mean semichord
    dim = size(Ks)[1] + length(globalDOFBlankingList)
    for Uinf in uSweep
        println("Solving for dynamic pressure (q): ", round(Uinf^2 * 0.5 * FOIL.ρ_f, digits=3), "Pa (", Uinf, "m/s)")

        # --- Apply the flutter solution method ---
        find_p(Uinf, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList)

        break
    end
end

function find_p(U∞, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList)
    """
    Non-iterative flutter solution following van Zyl

    Inputs
    ------
    Uinf: float
        free-stream velocity for eigenvalue solve
    fSweep: array
        frequency sweep
    structMesh: StructMesh
        mesh object
    FOIL: FOIL  
        foil object
    dim: int
        dimension of hydro matrices
    """

    # --- Initialize stuff ---
    globalMf::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_i::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_i::Matrix{Float64} = zeros(Float64, dim, dim)

    # ************************************************
    #     Loop over search frequencies
    # ************************************************
    for f in FOIL.fSweep

        ω = 2π * f
        k_ref = ω * b_ref / ((U∞ * cos(FOIL.Λ)))

        println("searching for ", f, " Hz")

        # ---------------------------
        #   Set the hydrodynamics
        # ---------------------------
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs!(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, FOIL, U∞, ω, elemType)
        Kf_r, Cf_r, Mf = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
        Kf_i, Cf_i, _ = Hydro.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

        # TODO: I split up the stuff before to separate imag and real math
        Cf = Cf_r + 1im * Cf_i
        Kf = Kf_r + 1im * Kf_i

        # ---------------------------
        #   1st-order eigenvalue solve
        # ---------------------------
        # --- Solve problem ---
        
        
        # --- Check matched point (Im(p) - k_ref) ---
        
        global p = 0.0
    end

    return p

end

end