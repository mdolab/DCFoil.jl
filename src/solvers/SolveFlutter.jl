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

function solve(DVDict, outputDir::String, uSweep; use_freeSurface=false, cavitation=nothing)
    """
    Use p-k method to find roots (p) to the equation
        (-p²[M]-p[C]+[K]){ũ} = {0}
    """

    # ************************************************
    #     Initialize
    # ************************************************
    global FOIL = InitModel.init_dynamic(nothing, DVDict, uSweep=uSweep)
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
    for Uinf in uSweep
        println("Solving for dynamic pressure (q): ", round(Uinf^2 * 0.5 * FOIL.ρ_f, digits=3), "Pa (", Uinf, "m/s)")



    end
end

end