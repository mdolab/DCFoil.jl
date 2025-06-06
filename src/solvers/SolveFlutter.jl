# --- Julia ---
"""
@File    :   SolveFlutter.jl
@Time    :   2022/10/07
@Author  :   Galen Ng with snippets from Eirikur Jonsson
@Desc    :   p-k method for flutter analysis, also contains modal analysis

             NOTE: if flutter solution is failing, there are 3 places to check:
                1. starting k guess
                2. tolerance on real root in 'extract_kCrossings()'
                3. maxK in the 'compute_pkFlutterAnalysis()' 
"""

module SolveFlutter

# --- PACKAGES ---
using LinearAlgebra
using Statistics
using JSON, JLD2
using Plots
using Printf
using FileIO
# Differentiation
using AbstractDifferentiation: AbstractDifferentiation as AD
using FiniteDifferences
using Zygote
using ReverseDiff: ReverseDiff
using ChainRulesCore: ChainRulesCore, @ignore_derivatives # this is an extremely weird bug that none of the code wrapped in @ignore_derivatives is evaluated

# --- DCFoil modules ---
for headerName in [
    "../utils/Utilities",
    "../struct/FEMMethods",
    "../hydro/LiftingLine",
    "../io/MeshIO",
    "../io/TecplotIO",
    "../constants/SolutionConstants",
    "../hydro/HydroStrip",
    "../solvers/SolverRoutines",
    "../constants/DesignConstants",
    "../solvers/DCFoilSolution",
    "../InitModel",
    "../solvers/SolverSetup",
]
    include("$(headerName).jl")
end

using .FEMMethods
using .LiftingLine
using .HydroStrip

# ==============================================================================
#                         MODULE CONSTANTS
# ==============================================================================
const loadType = "force"
const derivMode = "RAD"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solve(structMesh, elemConn, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, idxTip, LLSystem, claVec, FlowCond, debug)
    """
    Use p-k method to find roots (p) to the equation
        (-p²[M]-p[C]+[K]){ũ} = {0}

    Wrapper function to call pk flutter analysis and return the KS aggregated cost function
    This acts like 'subroutine velocitySweepPkNonIter()' from Eirikur's code.
    """

    # --- Primal flutter eigenvalue calculations ---
    appendageOptions = solverOptions["appendageList"][1]
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    p_r, p_i, true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, NTotalModesFound, nFlow = compute_pkFlutterAnalysis(
        uRange,
        structMesh,
        elemConn,
        b_ref,
        Λ,
        chordVec,
        abVec,
        ebVec,
        FOIL,
        dim,
        N_R,
        DOFBlankingList,
        idxTip,
        N_MAX_Q_ITER,
        nModes,
        CONSTANTS.Mmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        CONSTANTS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        CONSTANTS.Cmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        LLSystem, claVec, FlowCond
        ;
        # --- Optional args ---
        # ΔdynP=0.5 * FlowCond.rhof * 1^2,
        Δu=0.4,
        debug=debug,
        solverOptions=solverOptions,
        appendageOptions=appendageOptions
    )

    # --- Store solution in struct ---
    FLUTTERSOL = FlutterSolution(true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, NTotalModesFound, N_MAX_Q_ITER, flowHistory, nFlow, iblank, p_r)


    # ************************************************
    #     AGGREGATE DAMPING USING NON-DIM 'p'
    # ************************************************
    ρKS = solverOptions["rhoKS"]
    obj, pmG = postprocess_damping(FLUTTERSOL.N_MAX_Q_ITER, FLUTTERSOL.flowHistory, FLUTTERSOL.NTotalModesFound, FLUTTERSOL.nFlow, p_r, FLUTTERSOL.iblank, ρKS)


    return obj, pmG, FLUTTERSOL
end # end solve

function setup_solverFromDVs(α₀, Λ, span, c, toc, ab, x_αb, zeta, theta_f, solverOptions::AbstractDict)
    """
    Setup function to be called before solve()
    """

    # ************************************************
    #     Initializations
    # ************************************************
    outputDir = solverOptions["outputDir"]
    uRange = solverOptions["uRange"]
    fRange = solverOptions["fRange"]
    nModes = solverOptions["nModes"]
    debug = solverOptions["debug"]

    # --- Init model structure ---
    if length(solverOptions["appendageList"]) == 1
        foilOptions = solverOptions["appendageList"][1]
        tipMass = foilOptions["use_tipMass"]
        global FOIL, _ = InitModel.init_dynamic(α₀, 0.0, span, c, toc, ab, x_αb, zeta, theta_f,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            foilOptions, solverOptions; uRange=uRange, fRange=fRange)
    else
        error("Only one appendage is supported at the moment")
    end

    # --- Create mesh ---
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_componentMesh(nElem, span; config=foilOptions["config"])
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, c, c, c, c, 0.0, zeros(2, 2))

    println("====================================================================================")
    println("        BEGINNING FLUTTER SOLUTION")
    println("====================================================================================")
    println("Speed range [m/s]: ", uRange)
    if debug
        rm("./DebugOutput/", recursive=true)
        mkpath("./DebugOutput/")
        println("+---------------------------+")
        println("|    Running debug mode!    |")
        println("+---------------------------+")
    end

    # ************************************************
    #     FLUTTER SOLUTION
    # ************************************************
    N_MAX_Q_ITER = solverOptions["maxQIter"]    # TEST VALUE
    N_R = 8                                     # reduced problem size (Nr x Nr)
    abVec = ab
    x_αbVec = x_αb
    chordVec = c
    ebVec = 0.25 * chordVec .+ abVec
    b_ref = sum(chordVec) / FOIL.nNodes         # mean semichord

    # ************************************************
    #     FEM assembly
    # ************************************************
    globalKs, globalMs, globalF = FEMMethods.assemble(FEMESH, x_αbVec, FOIL, ELEMTYPE, FOIL.constitutive)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    globalDOFBlankingList = 0
    ChainRulesCore.ignore_derivatives() do
        globalDOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=foilOptions)
    end
    Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

    # ---------------------------
    #   Get damping
    # ---------------------------
    alphaConst, betaConst = FEMMethods.compute_proportional_damping(Ks, Ms, zeta, solverOptions["nModes"])
    Cs = alphaConst * Ms .+ betaConst * Ks
    globalCs = alphaConst * globalMs .+ betaConst * globalKs

    # ---------------------------
    #   Add any discrete masses
    # ---------------------------
    if tipMass
        bulbMass = 2200 #[kg]
        bulbInertia = 900 #[kg-m²]
        x_αbBulb = -0.1 # [m]
        dR = (structMesh[end, :] - structMesh[end-1, :])
        elemLength = √(dR[XDIM]^2 + dR[YDIM]^2 + dR[ZDIM]^2)
        # transMat = SolverRoutines.get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength, ELEMTYPE)
        transMat = get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength)
        Ms = FEMMethods.apply_tip_mass(Ms, bulbMass, bulbInertia, elemLength, x_αbBulb, transMat, ELEMTYPE)
    end

    # ---------------------------
    #   Constants
    # ---------------------------
    u = copy(globalF)
    dim = size(Ks)[1] + length(globalDOFBlankingList)
    alphaCorrection = zeros(nNodes)
    global SOLVERPARAMS = DCFoilSolverParams(globalKs, globalMs, globalCs, ELEMTYPE, zeros(2, 2), derivMode, 0.0, globalDOFBlankingList, alphaCorrection)

    return structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, SOLVERPARAMS, debug
end

function setup_solverFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions::AbstractDict)
    """
    Setup function to be called before solve()
    """

    # ************************************************
    #     Initializations
    # ************************************************
    uRange = solverOptions["uRange"]
    fRange = solverOptions["fRange"]
    nModes = solverOptions["nModes"]
    debug = solverOptions["debug"]

    # --- Init model structure ---
    if length(solverOptions["appendageList"]) == 1
        appendageOptions = solverOptions["appendageList"][1]
        tipMass = appendageOptions["use_tipMass"]
        FOIL, STRUT, _, FEMESH, LLOutputs, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)
    else
        error("Only one appendage is supported at the moment")
    end

    # --- Create mesh ---
    nElem = FOIL.nNodes - 1
    # structMesh, elemConn = FEMMethods.make_componentMesh(nElem, span; config=appendageOptions["config"])
    # FEMESH = FEMMethods.StructMesh(structMesh, elemConn, c, c, c, c, 0.0, zeros(2, 2))

    println("====================================================================================")
    println("        BEGINNING FLUTTER SOLUTION")
    println("====================================================================================")
    println("Speed range [m/s]: ", uRange)
    if debug
        rm("./DebugOutput/", recursive=true)
        mkpath("./DebugOutput/")
        println("+---------------------------+")
        println("|    Running debug mode!    |")
        println("+---------------------------+")
    end

    # ************************************************
    #     FLUTTER SOLUTION
    # ************************************************
    N_MAX_Q_ITER = solverOptions["maxQIter"]    # TEST VALUE
    N_R = 8                                     # reduced problem size (Nr x Nr)
    abVec = FOIL.ab
    x_αbVec = appendageParams["x_ab"]
    chordVec = FOIL.chord
    ebVec = 0.25 * chordVec .+ abVec
    b_ref = sum(chordVec) / FOIL.nNodes         # mean semichord

    # ************************************************
    #     FEM assembly
    # ************************************************
    globalKs, globalMs, globalF = FEMMethods.assemble(FEMESH, x_αbVec, FOIL, ELEMTYPE, FOIL.constitutive;
        config=appendageOptions["config"])

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    Ks, Ms, F = FEMMethods.apply_BCs(real.(globalKs), real.(globalMs), real.(globalF), ChainRulesCore.ignore_derivatives(DOFBlankingList))

    # ---------------------------
    #   Get structural damping
    # ---------------------------
    # alphaConst, betaConst = FEMMethods.compute_proportional_damping(Ks, Ms, appendageParams["zeta"], solverOptions["nModes"])
    alphaConst = solverOptions["alphaConst"]
    betaConst = solverOptions["betaConst"]
    Cs = alphaConst * Ms .+ betaConst * Ks
    globalCs = alphaConst * globalMs .+ betaConst * globalKs

    # ---------------------------
    #   Add any discrete masses
    # ---------------------------
    if tipMass
        bulbMass = 2200 #[kg]
        bulbInertia = 900 #[kg-m²]
        x_αbBulb = -0.1 # [m]
        dR = (structMesh[end, :] - structMesh[end-1, :])
        elemLength = √(dR[XDIM]^2 + dR[YDIM]^2 + dR[ZDIM]^2)
        # transMat = SolverRoutines.get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength, ELEMTYPE)
        transMat = get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength)
        Ms = FEMMethods.apply_tip_mass(Ms, bulbMass, bulbInertia, elemLength, x_αbBulb, transMat, ELEMTYPE)
    end

    # ---------------------------
    #   Constants
    # ---------------------------
    u = copy(globalF)
    dim = size(Ks)[1] + length(DOFBlankingList)
    alphaCorrection = 0.0
    SOLVERPARAMS = DCFoilSolverParams(globalKs, globalMs, globalCs, zeros(2, 2), 0.0, alphaCorrection)

    return FEMESH, LLSystem, LLOutputs, FlowCond, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, LLSystem.sweepAng, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, SOLVERPARAMS, debug
end

function solve_frequencies(LECoords, TECoords, nodeConn, appendageParams::AbstractDict, solverOptions::AbstractDict, appendageOptions::AbstractDict)
    """
    System natural frequencies
    """

    # ************************************************
    #     Initialize
    # ************************************************
    outputDir = solverOptions["outputDir"]
    nModes = solverOptions["nModes"]
    # tipMass = appendageOptions["use_tipMass"]
    tipMass = false
    debug = solverOptions["debug"]
    # --- Initialize the model ---
    # global FOIL, STRUT, _ = InitModel.init_modelFromDVDict(appendageParams, solverOptions, appendageOptions)
    # global FOIL, STRUT, _, FEMESH, LLOutputs, LLSystem, FlowCond = init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)
    # --- Init model structure ---
    if length(solverOptions["appendageList"]) == 1
        appendageOptions = solverOptions["appendageList"][1]
        tipMass = appendageOptions["use_tipMass"]

        idxTip = LiftingLine.get_tipnode(LECoords)
        midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        # ---------------------------
        #   Hydrodynamics
        # ---------------------------
        α0 = appendageParams["alfa0"]
        β0 = appendageParams["beta"]
        rake = appendageParams["rake"]
        depth0 = appendageParams["depth0"]
        airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
        displCol = zeros(6, npt_wing)
        npt_wing = size(displCol, 2) # overwrite
        LLSystem, FlowCond, _, _, _ = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol, pretwistDist;
            npt_wing=npt_wing,
            npt_airfoil=npt_airfoil,
            rhof=solverOptions["rhof"],
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
            options=options,
        )

    else
        error("Only one appendage is supported at the moment")
    end
    globalKs, globalMs, globalF, _, FEMESH, WingStructModel, StrutStructModel = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    println("====================================================================================")
    println("        BEGINNING MODAL SOLUTION")
    println("====================================================================================")
    if debug
        rm("./DebugOutput/", recursive=true, force=true) # remove the debug folder even if it doesn't exist
        mkpath("./DebugOutput/")
        println("+---------------------------+")
        println("|    Running debug mode!    |")
        println("+---------------------------+")
    end
    # ---------------------------
    #   Assemble structure
    # ---------------------------
    # chordVec = FOIL.chord
    # Λ = DVDict["sweep"]
    # Preprocessing.
    # globalKs_work, globalMs_work, globalF_work = FEMMethods.assemble(FEMESH, x_αbVec, FOIL, ELEMTYPE, FOIL.constitutive; config=appendageOptions["config"])
    # There some weird data type bug here so we need to copy the matrices and make them Float64
    # globalKs = zeros(Float64, size(globalKs_work))
    # globalMs = zeros(Float64, size(globalMs_work))
    # globalF = zeros(Float64, size(globalF_work))
    # globalKs .= globalKs_work
    # globalMs .= globalMs_work
    # globalF .= globalF_work

    if tipMass
        bulbMass = 2200 #[kg]
        bulbInertia = 900 #[kg-m²]
        x_αbBulb = -0.1 # [m]
        dR = (structMesh[end, :] - structMesh[end-1, :])
        elemLength = √(dR[XDIM]^2 + dR[YDIM]^2 + dR[ZDIM]^2)
        # transMat = SolverRoutines.get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength, ELEMTYPE)
        transMat = get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength)
        globalMs = FEMMethods.apply_tip_mass(globalMs, bulbMass, bulbInertia, elemLength, x_αbBulb, transMat, ELEMTYPE)
    end

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    Ks, Ms, _ = FEMMethods.apply_BCs(globalKs, globalMs, globalF, DOFBlankingList)

    # ---------------------------
    #   Initialize stuff
    # ---------------------------
    # u = zeros(length(globalF))
    wetModeShapes_sol = zeros(size(globalF, 1), nModes)
    structModeShapes_sol = zeros(size(globalF, 1), nModes)
    # FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordVec, chordVec, chordVec, chordVec, 0.0, zeros(2, 2)) # dummy inputs
    # alphaCorrection = 0.0
    # global CONSTANTS = DCFoilSolverParams(globalKs, globalMs, real(copy(globalKs)), zeros(2, 2), 0.0, alphaCorrection)

    # ---------------------------
    #   Test eigensolver
    # ---------------------------
    # --- Dry solve ---
    omegaSquared, structModeShapes = FEMMethods.compute_eigsolve(Ks, Ms, nModes)
    structNatFreqs = .√(omegaSquared) / (2π)
    println("+-------------------------------------+")
    println("| Structural natural frequencies [Hz]:")
    println("+-------------------------------------+")
    ctr = 1
    for natFreq in structNatFreqs
        println(@sprintf("mode %i: %.3f\t(%.3f rad/s)", ctr, natFreq, natFreq * 2π))
        ctr += 1
    end
    println("+-------------------------------------+")
    # --- Wetted solve ---
    claVec = 2π * ones(length(globalF) ÷ FEMMethods.NDOF) # this does not matter
    # end
    globalMf, globalCf_r, _, globalKf_r, _ = HydroStrip.compute_AICs(FEMESH, WingStructModel, LLSystem, claVec, FlowCond.rhof, size(globalMs)[1], LLSystem.sweepAng, 0.1, 0.1, FlowCond, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    _, _, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, DOFBlankingList)
    wetOmegaSquared, wetModeShapes = FEMMethods.compute_eigsolve(Ks, Ms .+ Mf, nModes)
    wetNatFreqs = .√(wetOmegaSquared) / (2π)
    println("| Wetted natural frequencies [Hz]:    |")
    println("+-------------------------------------+")
    ctr = 1
    for natFreq in wetNatFreqs
        println(@sprintf("mode %i: %.3f\t(%.3f rad/s)", ctr, natFreq, natFreq * 2π))
        ctr += 1
    end
    println("+-------------------------------------+")

    # --- Put BCs back ---
    for ii in 1:nModes
        structModeShapes_sol[:, ii], _ = FEMMethods.put_BC_back(structModeShapes[:, ii], ELEMTYPE)
        wetModeShapes_sol[:, ii], _ = FEMMethods.put_BC_back(wetModeShapes[:, ii], ELEMTYPE)
    end
    # ************************************************
    #     Write solution out to files
    # ************************************************
    if !(isempty(outputDir))
        write_modalsol(structNatFreqs, structModeShapes_sol, wetNatFreqs, wetModeShapes_sol, outputDir)
        if solverOptions["writeTecplotSolution"]
            write_tecplot_natural(appendageParams, structNatFreqs, structModeShapes_sol, wetNatFreqs, wetModeShapes_sol, FEMESH.mesh, chordVec, outputDir; solverOptions=solverOptions)
        end
    end

    return structNatFreqs, structModeShapes_sol, wetNatFreqs, wetModeShapes_sol
end # end solve_frequencies


# ==============================================================================
#                         Solution writer
# ==============================================================================
function write_solution(FLUTTERSOL, solverOptions, appendageParamsList; callNumber=0)
    outputDir = solverOptions["outputDir"] * @sprintf("/call_%03d/", callNumber)

    println("="^80)
    println("Writing solution files...")
    println("="^80)

    write_sol(FLUTTERSOL, outputDir)

    if solverOptions["writeTecplotSolution"] && solverOptions["run_static"]
        STATSOL = STATSOLLIST[iComp]
        FEMESH = STATSOL.FEMESH
        write_tecplot(appendageParams, FLUTTERSOL, FEMESH.chord, FEMESH.mesh, outputDir; solverOptions=solverOptions)
    end
end

function write_sol(FLUTTERSOL, outputDir="./OUTPUT/")
    """
    Write out the p-k flutter results
    """
    # Store solutions here
    workingOutput = outputDir * "/pkFlutter/"
    mkpath(workingOutput)
    println("Writing out flutter solution to ", workingOutput)

    true_eigs_r = FLUTTERSOL.eigs_r
    true_eigs_i = FLUTTERSOL.eigs_i
    R_eigs_r = FLUTTERSOL.R_eigs_r
    R_eigs_i = FLUTTERSOL.R_eigs_i
    iblank = FLUTTERSOL.iblank
    flowHistory = FLUTTERSOL.flowHistory

    # --- Store eigenvalues ---
    fname = workingOutput * "eigs_r.jld2"
    save(fname, "data", true_eigs_r)
    fname = workingOutput * "eigs_i.jld2"
    save(fname, "data", true_eigs_i)

    # --- Store eigenvectors ---
    fname = workingOutput * "eigenvectors_r.jld2"
    save(fname, "data", R_eigs_r)
    fname = workingOutput * "eigenvectors_i.jld2"
    save(fname, "data", R_eigs_i)

    # --- Store blanking ---
    fname = workingOutput * "iblank.jld2"
    save(fname, "data", iblank)

    # --- Store flow history ---
    fname = workingOutput * "flowHistory.jld2"
    save(fname, "data", flowHistory)

end # end write_sol

function write_modalsol(structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, outputDir="./OUTPUT/")
    """
    Write out the quiescent fluid results
    """

    # Store solutions here
    workingOutput = outputDir * "modal/"
    mkpath(workingOutput)

    # --- Store structural dynamics analysis ---
    fname = workingOutput * "structModal.jld2"
    save(fname, "structNatFreqs", structNatFreqs, "structModeShapes", structModeShapes)

    # --- Store wetted modal analysis ---
    fname = workingOutput * "wetModal.jld2"
    save(fname, "wetNatFreqs", wetNatFreqs, "wetModeShapes", wetModeShapes)

end

function write_tecplot(DVDict, FLUTTERSOL, chords, mesh, outputDir="./OUTPUT/"; solverOptions=Dict())
    """
    General purpose tecplot writer wrapper for flutter solution
    """
    write_hydroelastic_mode(DVDict, FLUTTERSOL, chords, mesh, outputDir, "mode"; appendageOptions=solverOptions["appendageList"][1])

end

function write_tecplot_natural(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, mesh, chords, outputDir="./OUTPUT/"; solverOptions=Dict())
    """
    General purpose tecplot writer wrapper for modal solution
    """
    write_natural_mode(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, mesh, chords, outputDir; solverOptions=solverOptions)

end

# ==============================================================================
#                         Flutter routines
# ==============================================================================
function compute_correlationMatrix(old_r, old_i, new_r, new_i)
    """
    This routine computes the eigenvector correlation matrix needed for associating converged eigenvectors
    This implementation is based on the van Zyl mode tracking method 1992. https://arc.aiaa.org/doi/abs/10.2514/3.46380

        X⋅Y = √(S1² + S2²) / √(S3 * S4)

    Although I think the output here is the square of the above equation.

    **Arguments
    Inputs
    ------
        old_r, old_i - the real/imaginary part of the old eigenvectors from previous iteration. The size is (Mold, Mold)
        new_r, new_i - the real/imaginary part of the new eigenvector from current iteration. The size is (Mnew, Mnew)
    Outputs
    -------
        C - The square correlation matrix. Values range from 0-1 where rows represent old eigenvectors and columns new eigenvectors.
        The size is (M_old, M_new)
    """

    M_old = size(old_r)[1] # nrows
    N_old = size(old_r)[2] # ncols or the number of modes
    M_new = size(new_r)[1]
    N_new = size(new_r)[2]
    C = zeros(M_old, M_new)

    # ---------------------------
    #   Eigenvector norms
    # ---------------------------
    # S3 for mode
    # Norm of each eigenvector for old array is a sum over rows
    normOld = .√(sum(old_r .^ 2 + old_i .^ 2, dims=1))

    # S4 for mode
    # Norm of each eigenvector for new array
    normNew = .√(sum(new_r .^ 2 + new_i .^ 2, dims=1))

    # Old and new eigen vectors
    old = old_r + 1im * old_i
    new = new_r + 1im * new_i
    # S1 S2 numerator
    # Dot product the arrays
    Ctmp = abs.(transpose(conj(old)) * new)


    # Now normalize correlation matrix Ctmp and put into output matrix C
    # --- Set to zero first for Buffer ---
    for jj in 1:M_new # loop cols
        for ii in 1:M_old # loop rows
            C[ii, jj] = 0.0
        end
    end

    for jj in 1:N_new # loop cols
        for ii in 1:N_old # loop rows
            if Ctmp[ii, jj] == 0.0 || normOld[ii] < MEPSLARGE || normNew[jj] < MEPSLARGE # do not allow divide by zero
                C[ii, jj] = 0.0
            else
                C[ii, jj] = Ctmp[ii, jj] / (normOld[ii] * normNew[jj])
            end
        end
    end

    return C

end # end function

function loop_corrMat(C, N_new)
    """
    Function that loops and finds the max element in the C matrix,
    stores the row and column indices in mTmp and corrTmp,
    then zeros out the row and column in the C matrix to continue

    Inputs
    ------
        C - correlation matrix
        N_new - number of eigenvectors in latest iteration
    Outputs
    -------

    """

    # --- Initialize outputs ---
    corrTmp = zeros(N_new)
    mTmp = zeros(Int64, N_new, 2)
    for ii in 1:N_new
        mTmp[ii, 1] = 0
        mTmp[ii, 2] = 0
        corrTmp[ii] = 0.0
    end

    isMaxValZero = true
    C_work = C # copy C matrix to work on
    while isMaxValZero

        # --- Find max value in correlation matrix and location ---
        maxI, maxJ, maxVal = maxLocArr2d(C_work)

        # --- Check if max value is zero ---
        if maxVal == 0.0
            isMaxValZero = false
        else
            # Store the correlation value in its proper location
            corrTmp[maxJ] = maxVal

            # Store where previous eigenvector was in proper spot for current eigenvector
            mTmp[maxJ, 1] = maxI

            # Zero out corresponding row and col in correlation matrix
            for jj in 1:size(C_work)[2] # loop cols
                C_work[maxI, jj] = 0.0
            end
            for ii in 1:size(C_work)[1] # loop rows
                C_work[ii, maxJ] = 0.0
            end

        end
    end # end while

    return mTmp, corrTmp
end

function compute_correlationMetrics(old_r, old_i, new_r, new_i, p_old_i, p_new_i; nFlow=nothing, debug=false)
    """
    Computes the correlation metrics based on previous and current eigenvectors
    between dynamic pressure increments qᵢ and qᵢ₊₁
    It is a modification to the van Zyl method that should handle if new modes appear
    by looking at the frequencies between modes (imag(p))

    Inputs
    ------
        old_r, old_i - the real/imaginary part of the old eigenvectors from previous iteration. The size is (Mold, Nold)
        new_r, new_i - the real/imaginary part of the new eigenvector from current iteration. The size is (Mnew, Nnew)
        p_old_i - imaginary part of eigenvalues from previous iteration. Size is (Nold)
        p_new_i - imaginary part of eigenvalues from previous iteration. Size is (Nnew)

    Outputs
    -------
        corr - correlation metric vector used to determine how well modes correlate, should be between 0-1 where 1 is perfect correlation
        m - array where for each line the first column has the index location of the old eigenvector and the second column has the new eigenvector location
        newModesIdx - holds the indices of any new modes

    Note that Mold == Mnew and Nold and Nnew should be at maximum Mold or MNew
    """

    # Rows and columns of old and new eigenvectors
    # M_old::Int64 = size(old_r)[1] # nrows
    N_old::Int64 = size(old_r)[2] # ncols or the number of modes
    # M_new::Int64 = size(new_r)[1] # TODO: why aren't these used
    N_new::Int64 = size(new_r)[2]

    # --- Initialize outputs ---
    corr = zeros(N_old)
    m = zeros(Int64, N_old, 2)
    newModesIdx = zeros(Int64, N_old)
    for ii in 1:N_old
        m[ii, 1] = 0
        m[ii, 2] = 0
        newModesIdx[ii] = 0
        corr[ii] = 0.0
    end

    # Working matrices
    newModesIdxTmp = zeros(Int64, N_new)
    for ii in 1:N_new
        newModesIdxTmp[ii] = 0
    end

    # --- Compute correlation matrix ---
    C = compute_correlationMatrix(old_r, old_i, new_r, new_i)

    # save correlation matrix
    if debug
        ChainRulesCore.ignore_derivatives() do
            fname = @sprintf("./DebugOutput/corrMatrix-%03i.jld2", nFlow)
            save(fname, "corrMat", C)
        end
    end

    # --- Loop over corr matrix ---
    # This loop finds the max element in the C matrix,
    # stores the row and column indices in `mTmp` and `corrTmp`,
    # then zeros out the row and column in the C matrix
    mTmp, corrTmp = loop_corrMat(C, N_new)

    # --- Add indices for location of new eigenvector (column 2) ---
    for ii in 1:N_new
        mTmp[ii, 2] = ii
    end

    # --- Find zero and nonzero elements in correlation array ---
    # Then find how many each has
    nz::Int64 = 0 # new mode counter
    nCorrelatedModes::Int64 = 0
    for ii in 1:N_new
        if corrTmp[ii] == 0.0 # new mode?
            # If correlation is zero, then it could be a new mode
            nz += 1
            newModesIdxTmp[nz] = mTmp[ii, 2]
        else # store index and correlation
            nCorrelatedModes += 1
            m[nCorrelatedModes, :] = mTmp[ii, :]
            corr[nCorrelatedModes] = corrTmp[ii]
        end
    end

    # --- Lower frequency modes missed? ---
    # Loop over all possible new modes and see if they are lower than some max
    nNewModes = 0
    maxVal = maximum(p_old_i) # maximum frequency from old modes
    # Loop over new modes
    for ii in 1:nz
        if (p_new_i[newModesIdxTmp[ii]] < maxVal / 2)
            nNewModes += 1
            newModesIdx[nNewModes] = newModesIdxTmp[ii]
        end
    end


    return corr, m, newModesIdx, nCorrelatedModes, nNewModes
end # end function

function print_flutter_text(nFlow, dynPTmp, U∞, nCorr, nCorrNewModes, NTotalModesFound, isFailed; printHeader=false)
    if printHeader
        println("+--------+-------------------+----------------+---------+-------+---------------+------------------+----------+")
        println("|  nFlow | Dyn. press. [kPa] | Velocity [m/s] | [knots] | nCorr | nCorrNewModes | NTotalModesFound | flowfail |")
        println("+--------+-------------------+----------------+---------+-------+---------------+------------------+----------+")
    else
        println(@sprintf("|  %04d  | %17.2f | %14.4f | %7.4f |   %03d |        %03d    |            %03d   |       %d  |",
            nFlow, dynPTmp * 1e-3, U∞, U∞ * 1.94384, nCorr, nCorrNewModes, NTotalModesFound, isFailed))
    end

end

function compute_pkFlutterAnalysis(vel, structMesh, elemConn, b_ref, Λ, chordVec, abVec, ebVec,
    FOIL, dim::Int64, Nr::Int64, DOFBlankingList, idxTip::Int64,
    N_MAX_Q_ITER, nModes, Mmat, Kmat, Cmat, LLSystem, claVec, FlowCond;
    ΔdynP=nothing, Δu=nothing, debug=false, solverOptions=Dict(), appendageOptions=Dict()
)
    """
    Non-iterative flutter solution following van Zyl https://arc.aiaa.org/doi/abs/10.2514/2.2806
    Everything from here on is based on the FORTRAN code written by Eirikur Jonsson.
    The docstrings may be exactly the same as Eirikur's code.

    Inputs
    ------
    vel: vector, size(2)
        start and end free-stream velocities for eigenvalue solve (flight conditions)
    structMesh: StructMesh [DESIGN VAR]
        mesh object
    b_ref: float [DESIGN VAR]
        mean semichord
    Λ: float [DESIGN VAR]
        sweep angle [rad]
    FOIL: FOIL
        foil object with all the other parameters
    dim: int
        dimension of hydro matrices
    Nr: int
        size of reduced problem (number of modes to orthogonalize against)
    DOFBlankingList: array, size(# of DOFs blanked)
        list of DOFs to be blanked
    N_MAX_Q_ITER: int
        maximum number of dynamic pressure iterations. DO NOT MAKE TOO LARGE OR THE CODE STALLS
    nModes: int
        number of modes to solve for
    Mmat: array, size(dim, dim) [DESIGN VAR]
        mass matrix
    Kmat: array, size(dim, dim) [DESIGN VAR]
        stiffness matrix
    Cmat: array, size(dim, dim) [DESIGN VAR]
        damping matrix
    ΔdynP: float
        dynamic pressure increment [Pa], either do dynP or vel increments
    Δu: float
        velocity increment [m/s]
    debug: bool
        flag to print debug statements

    Outputs
    -------
    p_r, p_i: array, size(nModes)
        real and imaginary parts of eigenvalues of flutter modes, [-]
    true_eigs_r, true_eigs_i: array, size(3*nModes, N_MAX_Q_ITER)
        real and imaginary parts of eigenvalues of flutter modes, [rad/s]
    R_eigs_r_tmp, R_eigs_i_tmp: array, size(2*dimwithBC, 3*nModes, N_MAX_Q_ITER)
        real and imaginary parts of eigenvectors of flutter modes
    iblank: array, size(3*nModes, N_MAX_Q_ITER)
        array of indices of blanked modes (which indicate failed solution)
    flowHistory: array, size(N_MAX_Q_ITER, 3)
        history of flow conditions [velocity, density, dynamic pressure]
    """
    # ************************************************
    #     Initializations
    # ************************************************
    # ---------------------------
    #   Modal method reduction
    # ---------------------------
    Mr, Kr, Cr, Qr = compute_modalSpace(Mmat, Kmat, Cmat; reducedSize=Nr)
    ChainRulesCore.ignore_derivatives() do
        println(@sprintf("Modal matrix Qr ∈ (%i x %i)", size(Qr)[1], size(Qr)[2]))
        println(@sprintf("Running max q iter: %i", N_MAX_Q_ITER))
    end
    # dimwithBC = Nr

    AEROMESH = FEMMethods.StructMesh(structMesh, elemConn, zeros(2), zeros(2), zeros(2), zeros(2), 0.0, idxTip, zeros(2, 2))

    # --- Outputs ---
    p_r = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    p_i = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    true_eigs_r = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    true_eigs_i = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    R_eigs_r = zeros(Float64, 2 * (dim - length(DOFBlankingList)), 3 * nModes, N_MAX_Q_ITER)
    R_eigs_i = zeros(Float64, 2 * (dim - length(DOFBlankingList)), 3 * nModes, N_MAX_Q_ITER)
    iblank = zeros(Int64, 3 * nModes, N_MAX_Q_ITER) # stores which modes are blanked and therefore have a failed solution
    flowHistory = zeros(Float64, N_MAX_Q_ITER, 3) # stores [velocity, density, dynamic pressure]

    # ---------------------------
    #   Working arrays
    # ---------------------------
    # --- Correlation arrays ---
    # The m matrix stores which mode is correlated with what.
    # The first column stores indices of old m[:,1] and
    # the second column stores indices of the newly found modes m[:,2]
    m = zeros(Int64, nModes * 3, 2)

    # --- Retained eigenvector matrices in the speed sweep ---
    R_eigs_r_tmp = zeros(Float64, 2 * Nr, 3 * nModes, N_MAX_Q_ITER)
    R_eigs_i_tmp = zeros(Float64, 2 * Nr, 3 * nModes, N_MAX_Q_ITER)

    # --- Others ---
    tmp = zeros(3 * dim) # temp array to store eigenvalues deltas between flow speeds
    dynP = 0.5 * solverOptions["rhof"] * vel .^ 2 # vector of dynamic pressures
    # ωSweep = 2π * FOIL.fRange # sweep of circular frequencies
    p_diff_max = 0.1 # max allowed change in roots between dynamic pressure steps 
    # Eirikur had this set to 0.1. If you make this too big, mode coalescence is a problem

    if debug
        # Needed for debugging
        fs = zeros(3 * nModes)
        gs = zeros(3 * nModes)
    end

    # ************************************************
    #     Velocity sweep loop
    # ************************************************
    # --- Initialize loop vars ---
    nFlow::Int64 = 1 # first flow iter
    # Set working fluid values for the loop
    dynPTmp = dynP[1] # set temporary dynamic pressure used in loop to first dynamic pressure
    U∞ = vel[1] # working vel is first velocity
    Umax = vel[end] # max velocity
    dynPMax = dynP[end] # set max dynamic pressure to last value
    nCorr::Int64 = 0
    NTotalModesFound::Int64 = 0 # total number of modes found over the entire simulation
    nCorrNewModes::Int64 = 0 # number of new modes to correlate
    is_failed::Bool = false
    nK::Int64 = 22 # number of k values to sweep
    maxK = 3.5 # max reduced frequency k to search # this is too low for low speed analysis
    maxK = 30.0 # max reduced frequency k to search
    # maxK = 60.0 # max reduced frequency k to search
    # maxK = 90.0 # max reduced frequency k to search
    maxK = 150.0 # max reduced frequency k to search (pesky moth case with high k


    # --- Zygote buffers ---
    p_r_z = Zygote.Buffer(p_r)
    p_i_z = Zygote.Buffer(p_i)
    p_r_z[:, :] = p_r
    p_i_z[:, :] = p_i
    true_eigs_r_z = Zygote.Buffer(true_eigs_r)
    true_eigs_i_z = Zygote.Buffer(true_eigs_i)
    true_eigs_r_z[:, :] = true_eigs_r
    true_eigs_i_z[:, :] = true_eigs_i
    R_eigs_r_tmp_z = Zygote.Buffer(R_eigs_r_tmp)
    R_eigs_i_tmp_z = Zygote.Buffer(R_eigs_i_tmp)
    R_eigs_r_tmp_z[:, :, :] = R_eigs_r_tmp
    R_eigs_i_tmp_z[:, :, :] = R_eigs_i_tmp
    flowHistory_z = Zygote.Buffer(flowHistory)
    flowHistory_z[:, :] = flowHistory
    iblank_z = Zygote.Buffer(iblank)
    iblank_z[:, :] = iblank
    while (nFlow <= N_MAX_Q_ITER)

        # --- Flow condition header printout ---
        ChainRulesCore.ignore_derivatives() do
            if nFlow % 15 == 1 # header every 10 iterations
                print_flutter_text(0, 0, 0, 0, 0, 0, 0; printHeader=true)
            end
        end

        # Set the proper fail flag
        is_failed = false # declare it as global for this scope

        # --- Sweep k and find eigenvalue/vector crossings ---
        # Non-dimensionalization factor
        # tmpFactor = U∞ * cos(Λ) / b_ref
        # div_tmp = 1 / tmpFactor
        # kSweep = ωSweep * div_tmp
        # --- Compute generalized hydrodynamic loads ---
        Mf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, kSweep = HydroStrip.compute_genHydroLoadsMatrices(maxK, nK, U∞, b_ref, dim, AEROMESH, Λ, FOIL, LLSystem, claVec, FlowCond.rhof, FlowCond, ELEMTYPE; appendageOptions=appendageOptions, solverOptions=solverOptions)
        p_cross_r, p_cross_i, R_cross_r, R_cross_i, kCtr = compute_kCrossings(Mf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, dim, kSweep, b_ref, Λ, chordVec, abVec, ebVec, FOIL, U∞, Mr, Kr, Cr, Qr, structMesh, DOFBlankingList; debug=solverOptions["debug"], qiter=nFlow)
        # ---------------------------
        #   Mode correlations
        # ---------------------------
        if (nFlow == 1) # first flight condition
            # Set the number of modes to correlate equal to the number of modes we're solving for
            nCorr = nModes
            NTotalModesFound = nModes

            # --- Print out first flight condition ---
            ChainRulesCore.ignore_derivatives() do
                print_flutter_text(nFlow, dynPTmp, U∞, nCorr, 0, NTotalModesFound, is_failed)
            end

            # Sort eigenvalues based on the frequency (imaginary part)
            idxTmp = sortperm(p_cross_i[1:kCtr])

            # --- Populate correlation matrix ---
            # If there is an indexing error, chances are you did not find enough k crossings
            # Run in debug mode and see where the Im(p) = k line is and adjust maxK
            ChainRulesCore.ignore_derivatives() do
                for ii in 1:size(m)[1]
                    m[ii, 1] = 0
                    m[ii, 2] = 0
                end
                for ii in 1:nModes
                    m[ii, 1] = ii
                    m[ii, 2] = idxTmp[ii]
                end
            end


        else # not first condition; apply mode tracking between flow speeds
            # --- Compute correlation matrix btwn dynP increments ---
            # Old eigenvectors
            old_r = R_eigs_r_tmp_z[:, :, nFlow-1]
            old_i = R_eigs_i_tmp_z[:, :, nFlow-1]
            # New eigenvectors
            new_r = R_cross_r[:, 1:kCtr]
            new_i = R_cross_i[:, 1:kCtr]
            # Eigenvalues
            p_old_i = p_i_z[:, nFlow-1]
            p_new_i = p_cross_i[1:kCtr]
            # Declare local scope var so we can ignore the derivatives
            corr = zeros(size(old_r)[2])
            newModesIdx = zeros(Int64, size(old_r)[2])
            ChainRulesCore.ignore_derivatives() do
                corr, m, newModesIdx, nCorr, nCorrNewModes = compute_correlationMetrics(old_r, old_i, new_r, new_i, p_old_i, p_new_i; nFlow=nFlow, debug=debug)
            end

            # --- Check if eigenvalue jump is too big ---
            # If the jump is too big, we back up
            # We do this by scaling the 'p' to the true eigenvalue
            eigScale = √(dynPTmp / flowHistory_z[nFlow-1, 3]) # This is a velocity scale


            # Compute difference between old and new modes 
            inner = (p_r_z[m[1:nCorr, 1], nFlow-1] - p_cross_r[m[1:nCorr, 2]] .* eigScale) .^ 2 +
                    (p_i_z[m[1:nCorr, 1], nFlow-1] - p_cross_i[m[1:nCorr, 2]] .* eigScale) .^ 2
            tmp_z = Zygote.Buffer(tmp)
            tmp_z[:] = tmp
            tmp_z[1:nCorr] = .√(inner)
            tmp = copy(tmp_z)

            maxVal = maximum(tmp[1:nCorr])

            if (maxVal > p_diff_max)
                ChainRulesCore.ignore_derivatives() do
                    println("FAILED - pkFlutterAnalysis : p_diff -> ", maxVal, " > ", p_diff_max)
                end
                is_failed = true
            end

            # ---------------------------
            #   Disappearing mode check
            # ---------------------------
            # Were there too many iterations w/o progress? Mode probably disappeared
            boolCheck = false # declare it as global for this scope
            iterTol = 0.05 # tolerance for increment
            if !(isnothing(ΔdynP))
                boolCheck = dynPTmp - flowHistory_z[nFlow-1, 3] < ΔdynP * iterTol
            elseif !(isnothing(Δu))
                boolCheck = U∞ - flowHistory_z[nFlow-1, 1] < Δu * iterTol
            end

            if (boolCheck)
                # Let's check the fail flag, if yes then accept
                if (is_failed)
                    # We should keep all modes that have high correlations, drop others
                    nKeep = 0
                    ChainRulesCore.ignore_derivatives() do
                        for ii in 1:nCorr
                            if corr[ii] > 0.5
                                nKeep += 1
                                m[nKeep, :] = m[ii, :]
                            end
                        end
                    end

                    # We only want the first nKeep modes so we need to overwrite nCorr, which is used later in the code for indexing
                    nCorr = nKeep

                    is_failed = false
                    ChainRulesCore.ignore_derivatives() do
                        println("INFO - pkFlutterAnalysis : Mode disappeared")
                    end
                end
            end

            # Now we add in any new lower frequency modes that weren't correlated above
            if (nCorrNewModes > 0) && !is_failed
                # if !is_failed

                # Append new modes' indices to the new 'm' array
                ChainRulesCore.ignore_derivatives() do
                    for ii in 1:nCorrNewModes
                        m[nCorr+ii, 1] = NTotalModesFound + ii
                        m[nCorr+ii, 2] = newModesIdx[ii]
                    end
                end
                # Increment the nCorr variable counter to include the recently added modes
                nCorr += nCorrNewModes

                # Increment the shift variable for the next new mode
                # NOTE: this shift variable has the total number of modes found over the ENTIRE simulation, not the current number of modes found
                NTotalModesFound += nCorrNewModes

                ChainRulesCore.ignore_derivatives() do
                    println("INFO - pkFlutterAnalysis : Adding new modes")
                end
                # end
            end



            # print out correlation data
            if debug
                ChainRulesCore.ignore_derivatives() do
                    lineCtr = 0
                    fname = @sprintf("./DebugOutput/corrmetric-%03i.jld2", nFlow)
                    save(fname, "corr", corr)

                    fname = @sprintf("./DebugOutput/m-%03i.dat", nFlow)
                    open(fname, "w") do io
                        write(io, "m matrix\n")
                        write(io, "old\tnew\n")
                        for ii in 1:size(m)[1]
                            stringData = @sprintf("%03d %03d\n", m[ii, 1], m[ii, 2])
                            write(io, stringData)
                        end
                    end
                end
            end


        end # end if mode correlation

        # ---------------------------
        #   Store solution
        # ---------------------------
        if is_failed # backup dynamic pressure

            dynPTmp = (dynPTmp - flowHistory_z[nFlow-1, 3]) * 0.5 + flowHistory_z[nFlow-1, 3]
            U∞ = √(2 * dynPTmp / flowHistory_z[nFlow-1, 2])

        else # Store solution
            # --- Unpack values ---
            eig_r_work = p_cross_r[m[1:nCorr, 2]]
            eig_i_work = p_cross_i[m[1:nCorr, 2]]
            R_eig_r_work = R_cross_r[:, m[1:nCorr, 2]]
            R_eig_i_work = R_cross_i[:, m[1:nCorr, 2]]

            # --- Non-dimensional Eigenvalues ---
            p_r_z[m[1:nCorr, 1], nFlow] = eig_r_work
            p_i_z[m[1:nCorr, 1], nFlow] = eig_i_work

            # --- Dimensional eigenvalues [rad/s] ---
            # Non-dimensionalization factor
            tmpFactor = U∞ * cos(Λ) / b_ref

            true_eigs_r_z[m[1:nCorr, 1], nFlow] = eig_r_work * tmpFactor
            true_eigs_i_z[m[1:nCorr, 1], nFlow] = eig_i_work * tmpFactor

            # --- Eigenvectors ---
            # These are the eigenvectors of the modal reduced system
            R_eigs_r_tmp_z[:, m[1:nCorr, 1], nFlow] = R_eig_r_work
            R_eigs_i_tmp_z[:, m[1:nCorr, 1], nFlow] = R_eig_i_work

            # --- Store flow history ---
            flowHistory_z[nFlow, 1] = U∞
            flowHistory_z[nFlow, 2] = FlowCond.rhof
            flowHistory_z[nFlow, 3] = dynPTmp

            # Set value to one to store the solution

            for idx in m[1:nCorr, 1]
                iblank_z[idx, nFlow] = 1
            end

            # --- Write values to file for debugging---
            if debug
                ChainRulesCore.ignore_derivatives() do
                    dimensionalization = tmpFactor / 2π
                    gs[m[1:nCorr, 1]] = p_cross_r[m[1:nCorr, 2]] * dimensionalization
                    fs[m[1:nCorr, 1]] = p_cross_i[m[1:nCorr, 2]] * dimensionalization
                    fname = @sprintf("./DebugOutput/eigenvalues-%03i.dat", nFlow)
                    speedString = @sprintf("Flow speed [m/s]: %10.8f\n", U∞)
                    stringData = "g [1/s]     f [Hz]\n"
                    open(fname, "w") do io
                        write(io, speedString)
                        write(io, stringData)
                        for ii in 1:3*nModes
                            stringData = @sprintf("%e %10.2f\n", gs[ii], fs[ii])
                            write(io, stringData)
                        end
                    end

                    # Store eigenvectors as JLD
                    fname = @sprintf("./DebugOutput/sorted-eigenvectors-%03i.jld2", nFlow)
                    save(fname, "evec_r", R_eigs_r_tmp_z[:, :, nFlow], "evec_i", R_eigs_i_tmp_z[:, :, nFlow])

                    fname = @sprintf("./DebugOutput/iblank-%03i.dat", nFlow)
                    stringData = "iblank\n"
                    open(fname, "w") do io
                        write(io, speedString)
                        write(io, stringData)
                        for ii in 1:3*nModes
                            stringData = @sprintf("%i\n", iblank[ii, nFlow])
                            # stringData = @sprintf("%i\n", iblank[ii, nFlow])
                            write(io, stringData)
                        end
                    end
                end
            end

            # --- Increment for next iteration ---
            nFlow += 1
            # Bool check if dynP or U∞ is being used
            if !(isnothing(ΔdynP))
                dynPTmp += ΔdynP
                # Determine flow speed
                U∞ = √(2 * dynPTmp / FlowCond.rhof)
            elseif !(isnothing(Δu))
                U∞ += Δu
                # Determine dynamic pressure
                dynPTmp = 0.5 * FlowCond.rhof * U∞^2
            end

            # --- Print out flight condition ---
            ChainRulesCore.ignore_derivatives() do
                if (nFlow != 1)
                    print_flutter_text(nFlow, dynPTmp, U∞, nCorr, nCorrNewModes, NTotalModesFound, is_failed)
                end # end if
            end

            # --- Check if we're done ---
            if (dynPTmp > dynPMax)
                # We should stop at (or near) the max velocity specified so we should check if we're within some tolerance

                # First subtract previously added increment, then subtract max velocity
                diff = 0.0
                if !(isnothing(ΔdynP))
                    diff = ((dynPTmp - ΔdynP) - dynPMax)
                elseif !(isnothing(Δu))
                    ΔdynP = dynPTmp - flowHistory_z[nFlow-1, 3]
                    diff = ((dynPTmp - ΔdynP) - dynPMax)
                end
                if (diff^2 < MEPSLARGE^2)
                    # Exit the loop
                    break
                else # Try max value
                    # If this fails, the step will be halved anyway and this process should repeat until the exit condition is met
                    ChainRulesCore.ignore_derivatives() do
                        println("End speed sweep. Set max dynamic pressure")
                    end
                    dynPTmp = dynPMax
                end

            end

        end


    end # end flow speed loop

    # --- Copy buffers ---
    p_r = copy(p_r_z)
    p_i = copy(p_i_z)
    true_eigs_r = copy(true_eigs_r_z) # once you make the copy call, it is frozen
    true_eigs_i = copy(true_eigs_i_z)
    R_eigs_r_tmp = copy(R_eigs_r_tmp_z)
    R_eigs_i_tmp = copy(R_eigs_i_tmp_z)
    flowHistory = copy(flowHistory_z)
    iblank = copy(iblank_z)

    # Decrement flow index since it was incremented before exit
    nFlow -= 1

    # ---------------------------
    #   Full eigenvectors
    # ---------------------------
    # We reduced the problem size using
    # {u} ≈ Qr * {q}
    # where {q} was the retained generalized coordinates
    # --- Zygote buffers ---
    R_eigs_r_z = Zygote.Buffer(R_eigs_r)
    R_eigs_i_z = Zygote.Buffer(R_eigs_i)
    R_eigs_r_z[:, :, :] = R_eigs_r
    R_eigs_i_z[:, :, :] = R_eigs_i
    for qq in 1:N_MAX_Q_ITER
        for mm in 1:3*nModes
            # We need to do some magic here because our eigenvectors are actually stacked
            # evec = [ū ; pn * ū]^T
            nBlank = length(DOFBlankingList)
            R_eigs_r_z[1:dim-nBlank, mm, qq] = Qr * R_eigs_r_tmp[1:Nr, mm, qq]
            R_eigs_i_z[1:dim-nBlank, mm, qq] = Qr * R_eigs_i_tmp[1:Nr, mm, qq]
            R_eigs_r_z[dim+1-nBlank:end, mm, qq] = Qr * R_eigs_r_tmp[Nr+1:end, mm, qq]
            R_eigs_i_z[dim+1-nBlank:end, mm, qq] = Qr * R_eigs_i_tmp[Nr+1:end, mm, qq]
        end
    end

    # --- Copy back from zygote buffer ---
    R_eigs_r = copy(R_eigs_r_z)
    R_eigs_i = copy(R_eigs_i_z)

    return p_r, p_i, true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, NTotalModesFound, nFlow

end # end function

function compute_kCrossings(Mf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, dim, kSweep, b_ref, Λ, chordVec, abVec, ebVec, FOIL, U∞, MM, KK, CC, Qr, structMesh, globalDOFBlankingList; debug=false, qiter=1)
    """
    # This routine solves an eigenvalue problem over a range of reduced frequencies k searches for the
    # intersection of each mode with the diagonal line Im(p) = k and then does a linear interpolation
    # for the eigenvalue and eigenvector. This is method of van Zyl https://arc.aiaa.org/doi/abs/10.2514/2.2806
    MM - structural mass matrix (Nr, Nr)
    KK - structural stiffness matrix (Nr, Nr)
    CC - structural damping matrix (Nr, Nr)
    Qr - modal matrix (dimwithBC, Nr)

    Outputs
    -------
        p_cross_r - unsorted eigenvalues [rad/s]
        p_cross_i - unsorted eigenvalues [rad/s]
        R_cross_r - unsorted eigenvectors [-]
        R_cross_i - unsorted eigenvectors [-]
        ctr - number of matched points
    """

    N_MAX_K_ITER = 5000 # max k iterations before code breaks

    # --- Loop over reduced frequency search range to construct lines ---
    # println("Sweeping k crossings")
    p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik =
        sweep_kCrossings(
            Mf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, dim, kSweep,
            b_ref, Λ, chordVec, abVec, ebVec, U∞, MM, KK, CC, Qr, structMesh, FOIL, globalDOFBlankingList, N_MAX_K_ITER
        )

    if (debug)
        ChainRulesCore.ignore_derivatives() do
            nonDimFactor = U∞ * cos(Λ) / b_ref / (2π)
            ######################################
            # PLOT WHERE MODES CROSS Im(p) = k
            ######################################
            marker = false
            plot(k_history[1:ik], p_eigs_i[1, 1:ik], label="mode 1", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[2, 1:ik], label="mode 2", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[3, 1:ik], label="mode 3", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[4, 1:ik], label="mode 4", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[5, 1:ik], label="mode 5", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[6, 1:ik], label="mode 6", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[7, 1:ik], label="mode 7", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[8, 1:ik], label="mode 8", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[9, 1:ik], label="mode 9", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[10, 1:ik], label="mode 10", marker=marker)
            plot!(k_history[1:ik], p_eigs_i[11, 1:ik], label="mode 11", marker=marker)
            plot!(k_history[1:ik], k_history[1:ik], lc=:black, label="Im(p)=k")
            ylims!((-5, 31.0))
            xlims!((-5, 31.0))
            ylims!((-1, 1.0))
            xlims!((-1, 1.0))
            plotTitle = @sprintf("U = %6.3f m/s", U∞)
            title!(plotTitle)
            xlabel!("k")
            ylabel!("Im(p)")
            fname = @sprintf("./DebugOutput/kCross-qiter-%03i.png", qiter)
            # println("Saving figure to: ", fname)
            savefig(fname)
        end
    end

    # --- Extract valid solutions through interpolation ---
    dimwithBC = dim - length(globalDOFBlankingList)
    Nr = size(Qr)[2]
    # dimwithBC = Nr
    p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr = extract_kCrossings(Nr, p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik, N_MAX_K_ITER)

    if debug
        ChainRulesCore.ignore_derivatives() do
            # Plot all extracted kCrossings as a root-locus
            scatter([p_cross_r[1]], [p_cross_i[1]], label="mode 1")
            scatter!([p_cross_r[2]], [p_cross_i[2]], label="mode 2")
            scatter!([p_cross_r[3]], [p_cross_i[3]], label="mode 3")
            scatter!([p_cross_r[4]], [p_cross_i[4]], label="mode 4")
            scatter!([p_cross_r[5]], [p_cross_i[5]], label="mode 5")
            scatter!([p_cross_r[6]], [p_cross_i[6]], label="mode 6")
            scatter!([p_cross_r[7]], [p_cross_i[7]], label="mode 7")
            scatter!([p_cross_r[8]], [p_cross_i[8]], label="mode 8")
            ylims!((-10.0, 100.0))
            xlims!((-20.0, 10.0))
            plotTitle = @sprintf("U = %.3f m/s", U∞)
            title!(plotTitle)
            xlabel!("Re(p)")
            ylabel!("Im(p)")
            fname = @sprintf("./DebugOutput/RL-qiter-%03i.png", qiter)
            savefig(fname)
            # Plot all extracted kCrossings as a V-f
            scatter([U∞], [p_cross_i[1]], label="mode 1")
            scatter!([U∞], [p_cross_i[2]], label="mode 2")
            scatter!([U∞], [p_cross_i[3]], label="mode 3")
            scatter!([U∞], [p_cross_i[4]], label="mode 4")
            scatter!([U∞], [p_cross_i[5]], label="mode 5")
            scatter!([U∞], [p_cross_i[6]], label="mode 6")
            scatter!([U∞], [p_cross_i[7]], label="mode 7")
            scatter!([U∞], [p_cross_i[8]], label="mode 8")
            xlims!((2, 30))
            ylims!((-10.0, 31.0))
            ylims!((-1, 1.0))
            plotTitle = @sprintf("U = %.3f m/s", U∞)
            title!(plotTitle)
            xlabel!("V [m/s]")
            ylabel!("Im(p)")
            fname = @sprintf("./DebugOutput/Vf-qiter-%03i.png", qiter)
            savefig(fname)
            # Plot all extracted kCrossings as a V-g
            scatter([U∞], [p_cross_r[1]], label="mode 1")
            scatter!([U∞], [p_cross_r[2]], label="mode 2")
            scatter!([U∞], [p_cross_r[3]], label="mode 3")
            scatter!([U∞], [p_cross_r[4]], label="mode 4")
            scatter!([U∞], [p_cross_r[5]], label="mode 5")
            scatter!([U∞], [p_cross_r[6]], label="mode 6")
            scatter!([U∞], [p_cross_r[7]], label="mode 7")
            scatter!([U∞], [p_cross_r[8]], label="mode 8")
            xlims!((2, 30))
            ylims!((-10.0, 0.0))
            ylims!((-1, 1.0))
            plotTitle = @sprintf("U = %.3f m/s", U∞)
            title!(plotTitle)
            xlabel!("V [m/s]")
            ylabel!("Re(p)")
            fname = @sprintf("./DebugOutput/Vg-qiter-%03i.png", qiter)
            savefig(fname)
        end
    end

    return p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr
end # end function

function sweep_kCrossings(globalMf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, dim, kSweep, b_ref, Λ,
    chordVec, abVec, ebVec, U∞, MM, KK, CC, Qr, structMesh, FOIL, DOFBlankingList, N_MAX_K_ITER)
    """
    Solve the eigenvalue problem over a range of reduced frequencies (k)

    Inputs
    ------
        globalMf - fluid added mass matrix (dim, dim)
        Cf_r_sweep, Cf_i_sweep - real and imaginary parts of fluid damping matrix for reduced frequency sweep (dim, dim, nK)
        Kf_r_sweep, Kf_i_sweep - real and imaginary part of fluid stiffness matrix for reduced frequency sweep (dim, dim, nK)
        dim - size of problem (nDOF w/o BC) (half the number of flutter modes you're solving for)
        kSweep - sweep of reduced frequencies
        b_ref - reference semichord
        MM - structural mass matrix
        KK - structural stiffness matrix
        CC - structural damping matrix
        Qr - modal matrix (Nr, Nr)
        globalDOFBlankingList - list of DOFs to blank out for the BCs

    Outputs
    -------
        p_eigs_r - unsorted eigenvalues of length ik [rad/s]
        p_eigs_i - unsorted eigenvalues of length ik [rad/s]
        R_eigs_r - unsorted eigenvectors of length ik [-]
        R_eigs_i - unsorted eigenvectors of length ik [-]
        k_history - history of reduced frequencies. Actual k values that we analyzed and accepted
        ik - number of k's that were actually analyzed and accepted

        The result is 'nMode' sets of lines that the eigenvalue problem was solved for

    TODO: visualization of matrix magnitudes
    """

    Nr = size(Qr)[2]

    # ---------------------------
    #   Outputs
    # ---------------------------
    # Eigenvalue and vector matrices
    p_eigs_r = zeros(Float64, 2 * Nr, N_MAX_K_ITER)
    p_eigs_i = zeros(Float64, 2 * Nr, N_MAX_K_ITER)
    R_eigs_r = zeros(Float64, 2 * Nr, 2 * Nr, N_MAX_K_ITER)
    R_eigs_i = zeros(Float64, 2 * Nr, 2 * Nr, N_MAX_K_ITER)
    k_history = zeros(Float64, N_MAX_K_ITER)
    p_eigs_r_z = Zygote.Buffer(p_eigs_r)
    p_eigs_i_z = Zygote.Buffer(p_eigs_i)
    R_eigs_r_z = Zygote.Buffer(R_eigs_r)
    R_eigs_i_z = Zygote.Buffer(R_eigs_i)
    k_history_z = Zygote.Buffer(k_history)


    p_diff_max = 0.2 # maximum allowed change in poles btwn two steps

    # --- Determine delta k (Δk) to step ---
    # based on minimum wetted natural frequency
    _, _, Mff = HydroStrip.apply_BCs(globalMf, globalMf, globalMf, DOFBlankingList)
    # Modal fluid added mass matrix (Cf and Kf handled in loop)
    Mf = Qr' * Mff * Qr
    # --- Determine Δk ---
    Δk = 0.0
    ChainRulesCore.ignore_derivatives() do
        omegaSquared, _ = FEMMethods.compute_eigsolve(KK, MM .+ Mf, Nr)
        # println("Wetted natural frequencies: $(.√(omegaSquared) ./ (2π)) Hz")
        # println(real.(omegaSquared))
        kwetted = .√(omegaSquared) * b_ref / (U∞)
        Δk = minimum(kwetted) * 0.2 # 20% of the minimum wetted natural frequency
        # omegaSquared, _ = FEMMethods.compute_eigsolve(KK, MM, Nr)
        # println("Dry natural frequencies: $(.√(omegaSquared) ./ (2π)) Hz")
    end

    # ************************************************
    #     Perform iterations on k values
    # ************************************************
    keepLooping = true
    # first 'k' guess close to zero as possible
    # NOTE: we do not want too small since AD breaks below 1e-15
    # But we do not want too large since we want to get static div modes
    # k = 1e-12 # don't use this
    k::Float64 = 2e-13 # this is a good value that catches static div modes
    ik::Int64 = 1 # k counter
    # Nk = length(kSweep)
    # pkEqnType = "rodden"
    pkEqnType = "ng"
    failed = false # fail flag on k jump must be reset to false on every k iteration
    while keepLooping
        failed = false

        # ---------------------------
        #   Compute hydrodynamics
        # ---------------------------
        # In Eirikur's code, he interpolates the AIC matrix. We tried computing it exactly 
        # but the cost is still too much even with strip theory
        # --- Interpolate AICs ---
        globalCf_r, globalCf_i = HydroStrip.interpolate_influenceCoeffs(k, kSweep, Cf_r_sweep, Cf_i_sweep, dim, pkEqnType)
        globalKf_r, globalKf_i = HydroStrip.interpolate_influenceCoeffs(k, kSweep, Kf_r_sweep, Kf_i_sweep, dim, pkEqnType)
        # --- Direct AIC computation ---
        # ω = k * U∞ * (cos(Λ)) / b_ref
        Kffull_r, Cffull_r, _ = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, DOFBlankingList) # real
        Kffull_i, Cffull_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, DOFBlankingList) # imag
        # Mode space reduction
        QrT = transpose(Qr)
        Kf_r = QrT * real(Kffull_r * Qr)
        Kf_i = QrT * real(Kffull_i * Qr)
        Cf_r = QrT * real(Cffull_r * Qr)
        Cf_i = QrT * real(Cffull_i * Qr)

        # --- Solve eigenvalue problem ---
        p_r_tmp, p_i_tmp, R_aa_r_tmp, R_aa_i_tmp = solve_eigenvalueProblem(pkEqnType, Nr, b_ref, U∞, Λ, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK, CC)
        # --- Sort eigenvalues from small to large ---
        p_r = sort(real(p_r_tmp))
        idxs = sortperm(real(p_r_tmp))
        p_i = p_i_tmp[idxs]
        R_aa_r = R_aa_r_tmp[:, idxs]
        R_aa_i = R_aa_i_tmp[:, idxs]

        # println("p m2: re ", p_r[2], " im ", p_i[2])

        # --- Mode tracking (prevent mode hopping between k's) ---
        # Don't need mode tracking for the very first step
        # This correlation matrix will be square
        if (ik > 1)
            # van Zyl tracking method: Find correlation matrix btwn current and previous eigenvectors (mode shape)
            # Rows are old eigenvector number and columns are new eigenvector number
            # corr = compute_correlationMatrix(R_eigs_r[:, :, ik-1], R_eigs_i[:, :, ik-1], R_aa_r, R_aa_i)
            ChainRulesCore.ignore_derivatives() do
                corr = compute_correlationMatrix(R_eigs_r_z[:, :, ik-1], R_eigs_i_z[:, :, ik-1], R_aa_r, R_aa_i)
                # Determine location of new eigenvectors
                idxs = argmax2d(transpose(corr))
            end

            # Check if entries are missing/duplicated
            # NOTE: this might not be a problem?

            # --- Order eigenvalues and eigenvectors based on correlation matrix ---
            p_r = p_r_tmp[idxs]
            p_i = p_i_tmp[idxs]

            R_aa_r = R_aa_r_tmp[:, idxs]
            R_aa_i = R_aa_i_tmp[:, idxs]

            # --- If too big of jump, back up on 'k' ---
            tmp_p_diff = .√((p_eigs_r_z[:, ik-1] - p_r) .^ 2 + (p_eigs_i_z[:, ik-1] - p_i) .^ 2)
            p_diff = maximum(real(tmp_p_diff))

            # # MAYBE DEBUG THE k jump here
            # # println("k = ", k, " p_diff = ", p_diff)
            # println("resonance freqs curr: ", p_i)
            # println("res freq prev: ", p_eigs_i_z[:, ik-1])
            # println(" p_diff = ", p_diff, " at mode ", ind)

            if (p_diff > p_diff_max)
                failed = true
                # When doing symmetric foil configs, some modes are really close to each other but of negated frequencies. We want to catch that
                indMax = argmax(real(tmp_p_diff))
                p_im_diff = maximum((p_i[indMax])^2 - (p_eigs_i_z[indMax, ik-1])^2)
                p_real_diff = maximum((p_r[indMax])^2 - (p_eigs_r_z[indMax, ik-1])^2)
                if p_im_diff < p_diff_max && p_real_diff < p_diff_max
                    failed = false
                    # println("Caught symmetric mode flip")
                end
                # breakit
            end
        end

        # ---------------------------
        #   Check solution
        # ---------------------------
        if failed
            # We need to try some new k guesses. Halve the step
            # k = 0.5 * (k - k_history[ik-1]) + k_history[ik-1]
            k = 0.5 * (k - k_history_z[ik-1]) + k_history_z[ik-1]

            # println("Failed at k = ", k)
            # println("kiter: ", ik)
        else # Success
            p_eigs_r_z[:, ik] = p_r
            p_eigs_i_z[:, ik] = p_i
            R_eigs_r_z[:, :, ik] = R_aa_r
            R_eigs_i_z[:, :, ik] = R_aa_i
            k_history_z[ik] = k

            # increment for next k
            ik += 1
            k += Δk

            # Check if we solved the matched Im(p) = k problem
            maxImP = maximum(real(p_i))
            if (maxImP < k_history_z[ik-1]) || (k > kSweep[end])
                # Assumes highest mode does NOT cross Im(p) = k line from below later
                # i.e. continue looping until the highest mode crosses the line or we reach maxK
                keepLooping = false
            end

        end

    end # end while

    ik = ik - 1 # Reduce counter b/c it was incremented in the last iteration

    p_eigs_r = copy(p_eigs_r_z)
    p_eigs_i = copy(p_eigs_i_z)
    R_eigs_r = copy(R_eigs_r_z)
    R_eigs_i = copy(R_eigs_i_z)
    k_history = copy(k_history_z)

    return p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik
end # end function

function extract_kCrossings(dim, p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik, N_MAX_K_ITER)
    """
    Find where solutions intersect Im(p) = k line and interpolate value

    Inputs
    ------
        dim - number of degrees of freedom
        p_eigs_r - unsorted real part of eigenvalues [rad/s]
        p_eigs_i - unsorted imaginary part of eigenvalues [rad/s]
        R_eigs_r - unsorted real part of eigenvectors [-]
        R_eigs_i - unsorted imaginary part of eigenvectors [-]
        k_history - history of k values [rad/s]
        ik - number of k values
        N_MAX_K_ITER - maximum number of k iterations

    Outputs
    -------
        p_cross_r - unsorted eigenvalues that cross Im(p) = k line [rad/s]
        p_cross_i - unsorted eigenvalues that cross Im(p) = k line [rad/s]
        R_cross_r - unsorted eigenvectors that cross Im(p) = k line [-]
        R_cross_i - unsorted eigenvectors that cross Im(p) = k line [-]
        ctr - Number of found matched points
    """
    # --- Initialize outputs ---
    p_cross_r = zeros(Float64, 2 * dim * 5)
    p_cross_i = zeros(Float64, 2 * dim * 5)
    R_cross_r = zeros(Float64, 2 * dim, 2 * dim * 5)
    R_cross_i = zeros(Float64, 2 * dim, 2 * dim * 5)
    # --- Good practice Zygote initializations ---
    p_cross_r_z = Zygote.Buffer(p_cross_r)
    p_cross_i_z = Zygote.Buffer(p_cross_i)
    R_cross_r_z = Zygote.Buffer(R_cross_r)
    R_cross_i_z = Zygote.Buffer(R_cross_i)
    p_cross_r_z[:] = p_cross_r
    p_cross_i_z[:] = p_cross_i
    R_cross_r_z[:, :] = R_cross_r
    R_cross_i_z[:, :] = R_cross_i


    # --- Look for crossing of diagonal line Im(p) = k ---
    ctr::Int64 = 1 # counter for number of found matched points and index for arrays
    # @simd for ii in 1:2*dim # loop over flutter modes (lines)
    # About a sixth of the time is here
    @inbounds begin
        for ii in 1:2*dim # loop over flutter modes (lines)
            for jj in 1:ik # loop over all reduced frequencies (tracing the mode line)

                # Check if we found a real root
                # If your flutter analyses are failing, this is probably why
                if (k_history[jj] < P_IM_TOL) &&
                   (abs(p_eigs_i[ii, jj]) < P_IM_TOL)
                    # There should be another real root coming up or we already processed
                    # one matching the zero freq
                    p_cross_r_z[ctr] = p_eigs_r[ii, jj]
                    p_cross_i_z[ctr] = p_eigs_i[ii, jj]
                    R_cross_r_z[:, ctr] = R_eigs_r[:, ii, jj]
                    R_cross_i_z[:, ctr] = R_eigs_i[:, ii, jj]
                    ctr += 1

                elseif jj < ik # Always true except for last mode
                    # Get left and right points
                    tmpCrossLeft = p_eigs_i[ii, jj] - k_history[jj]
                    tmpCrossRight = p_eigs_i[ii, jj+1] - k_history[jj+1]
                    # Get signs (+/-1 or 0)
                    tmpCrossLeftSign = sign(tmpCrossLeft)
                    tmpCrossRightSign = sign(tmpCrossRight)

                    # --- Find sign change ---
                    if tmpCrossLeftSign != tmpCrossRightSign
                        # Linear interpolation time since we intersected the Im(p)=k line

                        # You want the point at which Im(p) - k = 0.0
                        @fastmath begin
                            factor = (0.0 - (p_eigs_i[ii, jj+1] - k_history[jj+1])) / ((p_eigs_i[ii, jj] - k_history[jj]) - (p_eigs_i[ii, jj+1] - k_history[jj+1]))

                            p_cross_r_z[ctr] = factor * (p_eigs_r[ii, jj] - p_eigs_r[ii, jj+1]) + p_eigs_r[ii, jj+1]
                            p_cross_i_z[ctr] = factor * (p_eigs_i[ii, jj] - p_eigs_i[ii, jj+1]) + p_eigs_i[ii, jj+1]

                            # --- Eigenvectors ---
                            evec_r_jj = R_eigs_r[:, ii, jj]
                            evec_r_jj1 = R_eigs_r[:, ii, jj+1]
                            evec_i_jj = R_eigs_i[:, ii, jj]
                            evec_i_jj1 = R_eigs_i[:, ii, jj+1]
                            # Look at real part of inner product of two eigenvectors m and m+1
                            tmpSum_r = 0.0
                            for ll in 1:2*dim
                                # tmpSum_r += R_eigs_r[ll, ii, jj+1] * R_eigs_r[ll, ii, jj] - (-1) * R_eigs_i[ll, ii, jj+1] * R_eigs_i[ll, ii, jj]
                                tmpSum_r += evec_r_jj1[ll] * evec_r_jj[ll] - (-1) * evec_i_jj1[ll] * evec_i_jj[ll]
                            end
                            if tmpSum_r > 0.0
                                # R_cross_r_z[:, ctr] = factor * (R_eigs_r[:, ii, jj] - R_eigs_r[:, ii, jj+1]) + R_eigs_r[:, ii, jj+1]
                                # R_cross_i_z[:, ctr] = factor * (R_eigs_i[:, ii, jj] - R_eigs_i[:, ii, jj+1]) + R_eigs_i[:, ii, jj+1]
                                R_cross_r_z[:, ctr] = factor * (evec_r_jj - evec_r_jj1) + evec_r_jj1
                                R_cross_i_z[:, ctr] = factor * (evec_i_jj - evec_i_jj1) + evec_i_jj1
                            else
                                # R_cross_r_z[:, ctr] = factor * (-R_eigs_r[:, ii, jj] - R_eigs_r[:, ii, jj+1]) + R_eigs_r[:, ii, jj+1]
                                # R_cross_i_z[:, ctr] = factor * (-R_eigs_i[:, ii, jj] - R_eigs_i[:, ii, jj+1]) + R_eigs_i[:, ii, jj+1]
                                R_cross_r_z[:, ctr] = factor * (-evec_r_jj - evec_r_jj1) + evec_r_jj1
                                R_cross_i_z[:, ctr] = factor * (-evec_i_jj - evec_i_jj1) + evec_i_jj1
                            end
                        end

                        ctr += 1

                    end
                end
            end
        end
    end
    # --- Final Zygote copies ---
    p_cross_r = copy(p_cross_r_z)
    p_cross_i = copy(p_cross_i_z)
    R_cross_r = copy(R_cross_r_z)
    R_cross_i = copy(R_cross_i_z)

    ctr -= 1 # Decrease counter since last successful iteration incremented it

    return p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr

end # end extract_kCrossings

function solve_eigenvalueProblem(pkEqnType, dim, b, U∞, Λ, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK, CC)
    # """
    # This routine solves the following eigenvalue problem.
    #   [ (U/b)^2 * p^2 * M + (U/b) * C + K - F_aero ] * u = 0

    # Form a first order system by introducing I*\dot{u} = I*\dot{u} where I is identity
    # Then the problem is a generalized eigenvalue problem, p*A*u = B*u where p,u are eigen- values/vectors
    # Then we cast ot into standard form p * u = A^{-1} * B * u which then is solved
    # Note: The A and B in the are just general matrices and serve as placeholders, A is NOT the aero loads
    #       as written in the equaions below.

    # Hassig
    #     The Aerodynamic loads are written
    #     F_aero = qinf * A
    #         p * |I       0           | * | u |  =  |    0        I         | * | u |
    #             |0  (Ucos(Λ)/b)^2 * M|   |p*u|     |-(K-q*A)  -Ucos(Λ)/b*C |   |p*u|

    # Rodden
    #     The Aerodynamic loads are written in terms of real (R) and imaginary part (I)
    #     F_aero = qinf * (A^R + i * A^I)
    #         p * |I       0           | * | u |  =  |    0                   I                        | * | u |
    #             |0  (Ucos(Λ)/b)^2 * M|   |p*u|     |-(K-q*A^R)  -(Ucos(Λ)/b*C - qinf/k * A^I)  |   |p*u|
    #     This method is good because the matrices are all real

    # Ng
    #     The Aerodynamic loads are written in terms of generalized hydro matrices which may
    #     only be possible with analytical methods for hydrodynamics
    #     F_aero = -(Mf*uddot + Cf*udot + Kf*u)
    #         p * |I       0                 | * | u |  =  |    0        I               | * | u |
    #             |0  (Ucos(Λ)/b)^2 * (MM+Mf)|   |p*u|     |-(KK+Kf)  -Ucos(Λ)/b*(CC+Cf) |   |p*u|
    #     but note that CC might be a super small structural damping
    # ARGUMENTS
    #       dim - the size of reduced problem
    #       k - the reduced frequency
    #       b - the reference half chord. Scalar
    #       vel - the velocity of the fluid. Scalar
    #       Mf, Cf_r, Cf_i, Kf_r, Kf_i - The AIC (real/imag parts). Array(dim,dim)
    #       MM - structural mass matrix. Array(dim,dim)
    #       CC - structural damping matrix. Array(dim,dim)
    #       KK - structural stiffness matrix. Array(dim,dim)
    # Outputs
    # -------
    #     p_r - real part of flutter eigenvalue (2*dim)
    #     p_i - imag part of flutter eigenvalue (2*dim)
    #     R_aa_r - real part of flutter mode shape (2*dim, 2*dim)
    #     R_aa_i - imag part of flutter mode shape (2*dim, 2*dim)
    # """

    # --- Initialize ---
    iMat = Matrix{Int}(I, dim, dim)
    zeroMat = zeros(dim, dim)
    A_r = zeros(2 * dim, 2 * dim)
    A_i = zeros(2 * dim, 2 * dim) # this will always be zero!
    B_r = zeros(2 * dim, 2 * dim)
    B_i = zeros(2 * dim, 2 * dim)

    # ************************************************
    #     Generalized eigenvalue problem
    # ************************************************
    # Form the matrices that form the generalized eigenvalue problem given by
    #           p[A]{v} + [B]{v} = 0
    # All entries in A and B are real except the ones coming from the AIC
    # NOTE: Ok, so I've actually realized that the 'Hassig' form and 'Rodden' form here are moot points
    # because I have broken up the generalized hydrodynamic forces in mass, damping, and stiffness matrices
    # As a result, the eigenvalue problem is technically in a modified 'Rodden' form that has fluid-added mass.
    nonDimFactor = (U∞ * cos(Λ) / b)

    if pkEqnType == "ng"

        # --- Form A matrix ---
        # A - real part
        firstRow = hcat(iMat, zeroMat)
        secondRow = hcat(zeroMat, nonDimFactor * nonDimFactor .* (Mf + MM))
        A_r = vcat(firstRow, secondRow)
        # A - imag part, do nothing because it is zero!

        # --- Form B matrix ---
        # B - Real part
        firstRow = hcat(zeroMat, iMat)
        secondRow = hcat(-1 * (Kf_r + KK), -1 * nonDimFactor .* (Cf_r + CC))
        B_r = vcat(firstRow, secondRow)

        # B - Imag part
        firstRow = hcat(zeroMat, zeroMat)
        secondRow = hcat(-1 * (Kf_i), -1 * nonDimFactor .* (Cf_i))
        B_i = vcat(firstRow, secondRow)

    elseif pkEqnType == "hassig"

        # Precompute quasi-steady AIC matrices
        qA_r = Cf_r + Kf_r
        qA_i = Cf_i + Kf_i

        # --- Form A matrix ---
        # A - real part
        firstRow = hcat(iMat, zeroMat)
        secondRow = hcat(zeroMat, nonDimFactor * nonDimFactor .* (Mf + MM))
        A_r = vcat(firstRow, secondRow)
        # A - imag part, do nothing because it is zero!

        # --- Form B matrix ---
        # B - Real part
        firstRow = hcat(zeroMat, iMat)
        secondRow = hcat(-KK + qA_r, zeroMat) # no structural damping right now
        B_r = vcat(firstRow, secondRow)
        # B - Imag part
        firstRow = hcat(zeroMat, zeroMat)
        secondRow = hcat(qA_i, zeroMat)

    elseif pkEqnType == "rodden"

        # Precompute quasi-steady AIC matrices
        qA_r = Cf_r + Kf_r
        qA_i = Cf_i + Kf_i

        # --- Form A matrix ---
        # A - real part
        firstRow = hcat(iMat, zeroMat)
        secondRow = hcat(zeroMat, nonDimFactor * nonDimFactor .* (Mf + MM))
        A_r = vcat(firstRow, secondRow)
        # A - imag part, do nothing because it is zero!

        # --- Form B matrix ---
        # B - Real part
        firstRow = hcat(zeroMat, iMat)
        secondRow = hcat(-KK + qA_r, qA_i) # no structural damping right now
        B_r = vcat(firstRow, secondRow)
        # B - Imag part, do nothing because it is zero!

    end


    # --- Invert the complex matrix and solve ---
    A = A_r + 1im * A_i
    B = B_r + 1im * B_i
    Ainv = inv(A)
    FlutterMat = Ainv * B
    FlutterMat_r = real(FlutterMat)
    FlutterMat_i = imag(FlutterMat)
    # Or the broken apart way here
    # Ainv_r, Ainv_i = SolverRoutines.cmplxInverse(A_r, A_i, 2 * dim)
    # FlutterMat_r, FlutterMat_i = SolverRoutines.cmplxMatmult(Ainv_r, Ainv_i, B_r, B_i)

    # --- Compute the eigenvalues ---
    # p_r, p_i, _, _, R_aa_r, R_aa_i = SolverRoutines.cmplxStdEigValProb(FlutterMat_r, FlutterMat_i, 2 * dim)
    # p_r, p_i, _, _, R_aa_r, R_aa_i = SolverRoutines.cmplxStdEigValProb_fad(FlutterMat_r, FlutterMat_i, 2 * dim)
    y = cmplxStdEigValProb2(FlutterMat_r, FlutterMat_i, 2 * dim)
    n = 2 * dim
    p_r = y[1:n]
    p_i = y[n+1:2*n]
    R_aa_r = reshape(y[2*n+1:2*n+n^2], n, n)
    R_aa_i = reshape(y[2*n+n^2+1:end], n, n)

    return p_r, p_i, R_aa_r, R_aa_i
end # end solve_eigenvalueProblem

function compute_modalSpace(Ms, Ks, Cs; reducedSize=20)
    """
    Reduce to modal space to reduce the size of the problem

    Inputs
    ------
    Ms - structural mass matrix after BC blanking. Array(dim,dim)
    Ks - structural stiffness matrix after BC blanking. Array(dim,dim)
    reducedSize - number of modes to keep, Nr. Int

    Outputs
    -------
    Ms_r - reduced mass matrix. Array(Nr,Nr)
    Ks_r - reduced stiffness matrix. Array(Nr,Nr)
    Cs_r - reduced damping matrix. Array(Nr,Nr)
    Qr - modal matrix. Array(dim,Nr)
    """

    # --- Compute the modal space matrix ---
    _, ubar = FEMMethods.compute_eigsolve(Ks, Ms, reducedSize)
    Qr = ubar
    QrT = transpose(Qr)

    # --- Compute the reduced matrices ---
    Ms_r = QrT * real(Ms * Qr) # need this for type stability in AD
    Ks_r = QrT * real(Ks * Qr) # need this for type stability in AD
    Cs_r = QrT * Cs * Qr

    return Ms_r, Ks_r, Cs_r, Qr
end

function postprocess_damping(N_MAX_Q_ITER, flowHistory, NTotalModesFound, nFlow, p_r, iblank, ρKS)
    """
    Function to compute damping on all modes and aggregate into the flutter constraint

    Inputs
    ------
        N_MAX_Q_ITER
        flowHistory - flow history. Array(Nflow, 3)
        NTotalModesFound - total number of modes found. Int
        nFlow - number of flow conditions. Int
        p_r - real part of the eigenvalues across all flow conditions [-]. Array(N_MAX_Q_ITER, Nmodes).
            This has to be the non-dimensional version so the derivatives are correct
        iblank - blanking indices

    Outputs
    -------
    obj - flutter constraint [-]. Float
    pmG - flutter safety margin
    """
    # ************************************************
    #     Initializations
    # ************************************************
    # Array with same size as nFlow
    # idx = [ii for ii = 1:nFlow]
    idx = 1:nFlow

    # This variable stores the aggregated damping for every mode
    # It mutates in the for loop below so we need to use 'Buffer' and work with that variable instead
    ksTmp = zeros(Float64, NTotalModesFound)
    ksTmp_z = Zygote.Buffer(ksTmp)

    # TODO: safety window
    # G = zeros(Float64, nFlow)
    # if useSafetyWindow
    #     G[1:nFound] = compute_safetyWindow(flowHistory[1:nFlow,1], nFlow)
    # end

    pmG = zeros(Float64, NTotalModesFound, nFlow)
    pmG_z = Zygote.Buffer(pmG)
    pmG_z[1:NTotalModesFound, 1:nFlow] = p_r[1:NTotalModesFound, 1:nFlow]
    pmGTmp = zeros(Float64, nFlow) # working pmG for the mode

    # ************************************************
    #     Aggregate damping
    # ************************************************
    # ---------------------------
    #   Aggregate by velocities
    # ---------------------------
    for modeIdx in 1:NTotalModesFound

        # Extract indices from 'idx' using the mask from 'iblank' containing the values for this mode
        subIdx, nFound = ipack1d(idx, iblank[modeIdx, 1:nFlow] .!= 0, nFlow)

        # if useSafetyWindow
        #     pmG[modeIdx, subIdx[1:nFound]] -= G[subIdx[1:nFound]]
        # end

        pmGTmp = copy(pmG_z[modeIdx, subIdx[1:nFound]])

        ksTmp_z[modeIdx] = compute_KS(pmGTmp[1:nFound], ρKS)
    end
    ksTmp = copy(ksTmp_z)

    # ---------------------------
    #   Aggregate over modes
    # ---------------------------
    obj = compute_KS(ksTmp, ρKS)

    # ChainRulesCore.ignore_derivatives() do
    #     # println("ksrho = ", ρKS)
    #     println("ksTmp = ", ksTmp)
    #     println("obj = ", obj)
    # end

    return obj, pmG
end


# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
# function get_sol(DVDict::AbstractDict, solverOptions::AbstractDict)
#     """
#     Wrapper function
#     This does the primal solve and returns the solution
#     It is used in the top-level evalFuncs call in DCFoil 
#     to return the sol struct
#     """

#     # # Setup
#     # structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug =
#     #     setup_solverFromDVs(DVDict["alfa0"], DVDict["sweep"], DVDict["s"], DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_ab"], DVDict["zeta"], DVDict["theta_f"], solverOptions)

#     # # Solve
#     # obj, _, SOL = solve(structMesh, elemConn, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug)
#     DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
#     SOL = cost_funcsFromDVs(DVVec, DVLengths, solverOptions)

#     return SOL
# end

# function compute_solFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions)

#     ptVec, _, _ = Utilities.unpack_coords(LECoords, TECoords)
#     alfa0 = appendageParams["alfa0"]
#     theta_f = appendageParams["theta_f"]
#     toc = appendageParams["toc"]

#     # these structural damping constants are hidden in this dictionary to keep them constant throughout optimization
#     haskey(solverOptions, "alphaConst") || error("solverOptions must contain 'alphaConst'")

#     obj, SOL = cost_funcsFromCoordsDVs(ptVec, nodeConn, alfa0, theta_f, toc, appendageParams, solverOptions; return_all=true)
#     return SOL
# end

# function cost_funcsFromDVs(
#     # α₀, Λ, span, c, toc, ab, x_αb, zeta, theta_f, 
#     DVVec, DVLengths,
#     solverOptions
# )
#     """
#     This does the primal solve but with a function signature compatible with Zygote
#     """

#     DVDict = Utilities.repack_dvdict(DVVec, DVLengths)
#     # Setup
#     structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, debug =
#         setup_solverFromDVs(DVDict["alfa0"], DVDict["sweep"], DVDict["s"], DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_ab"], DVDict["zeta"], DVDict["theta_f"], solverOptions)

#     # Solve
#     obj, _, SOL = solve(structMesh, elemConn, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, debug)

#     return obj
# end

# function cost_funcsFromCoordsDVs(
#     ptVec,
#     nodeConn,
#     alfa0,
#     theta_f,
#     toc,
#     appendageParams,
#     solverOptions;
#     return_all=false
# )
#     """
#     This does the primal solve but with a function signature compatible with Zygote
#     """

#     LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

#     appendageParams["theta_f"] = theta_f
#     appendageParams["toc"] = toc
#     appendageParams["alfa0"] = alfa0

#     # Setup
#     FEMESH, LLSystem, LLOutputs, FlowCond, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, debug =
#         setup_solverFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions)

#     # Solve
#     obj, _, SOL = solve(FEMESH.mesh, FEMESH.elemConn, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, FEMESH.idxTip, LLSystem, LLOutputs, FlowCond, debug)
#     if return_all
#         return obj, SOL
#     else
#         return obj
#     end
# end

function cost_funcsFromDVsOM(ptVec, nodeConn, displacements_col, claVec, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=false)
    """
    """

    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc
    appendageParams["alfa0"] = alfa0

    FEMESH, LLSystem, FlowCond, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, debug = setup_solverOM(displacements_col, LECoords, TECoords, nodeConn, appendageParams, solverOptions)

    obj, _, SOL = solve(FEMESH.mesh, FEMESH.elemConn, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, FEMESH.idxTip, LLSystem, claVec, FlowCond, debug)

    if return_all
        return obj, SOL
    else
        return obj
    end
end

function evalFuncsSens(evalFuncsSensList, appendageParams::AbstractDict, GridStruct, displacementsCol, claVec, solverOptions::AbstractDict; mode="FiDi")
    """
    Wrapper to compute the total sensitivities for this evalFunc

    Outputs
    -------
        funcsSens - sensitivities of the cost funcs. Dict
    """

    println("===================================================================================================")
    println("        FLUTTER SENSITIVITIES: ", mode)
    println("===================================================================================================")

    # Initialize output dictionary

    LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
    ptVec, mm, NPT = LiftingLine.unpack_coords(GridStruct.LEMesh, GridStruct.TEMesh)

    # these structural damping constants are hidden in this dictionary to keep them constant throughout optimization
    haskey(solverOptions, "alphaConst") || error("solverOptions must contain 'alphaConst'")

    funcsSensOut = Dict()

    for evalFuncSensKey in evalFuncsSensList
        println("-"^20)
        println("Cost func: $(evalFuncSensKey)")
        println("-"^20)

        tFunc = @elapsed begin

            funcsSens = Dict()
            if uppercase(mode) == "FIDI" # use finite differences the stupid way
                backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))

                # dksflutterdx, = AD.gradient(backend, (x) -> cost_funcsFromDVs(x, DVLengths, solverOptions), DVVec)
                # NOTE: just make sure the xalfa, xtheta, xtoc are in the same order as `allDesignVariables` list
                # dIdxDV = AD.gradient(backend, (xpt, xalpha, xtheta, xtoc) ->
                #         cost_funcsFromCoordsDVs(xpt, nodeConn, xalpha, xtheta, xtoc, appendageParams, solverOptions), ptVec, appendageParams["alfa0"], appendageParams["theta_f"], appendageParams["toc"])

                dIdxDV = AD.gradient(backend, (xpt, xdispl, xcla, xtheta, xtoc, xalpha) ->
                        cost_funcsFromDVsOM(xpt, nodeConn, xdispl, xcla, xtheta, xtoc, xalpha, appendageParams, solverOptions),
                    ptVec,
                    displacementsCol,
                    claVec,
                    appendageParams["theta_f"],
                    appendageParams["toc"],
                    appendageParams["alfa0"])

            elseif uppercase(mode) == "RAD" # use automatic differentiation via Zygote
                # backend = AD.ZygoteBackend()

                # AD backend was screwing up the output so use Zygote directly
                # dIdxDV = Zygote.gradient((xpt, xalpha, xtheta, xtoc) ->
                #         cost_funcsFromCoordsDVs(xpt, nodeConn, xalpha, xtheta, xtoc, appendageParams, solverOptions), ptVec, appendageParams["alfa0"], appendageParams["theta_f"], appendageParams["toc"])

                dIdxDV = Zygote.gradient((xpt, xdispl, xcla, xtheta, xtoc, xalpha) ->
                        cost_funcsFromDVsOM(xpt, nodeConn, xdispl, xcla, xtheta, xtoc, xalpha, appendageParams, solverOptions),
                    ptVec,
                    displacementsCol,
                    claVec,
                    appendageParams["theta_f"],
                    appendageParams["toc"],
                    appendageParams["alfa0"],
                )

                # --- In the future, reverse diff would be better and faster, but for now it doesn't work ---
                # backend = AD.ReverseDiffBackend()
                # # There's a weird quirk here with ReverseDiff that the input val has to be the copy of it b/c I guess ReverseDiff modifies the dictionary in place if you do not pass in the copy
                # theta_val = copy(appendageParams["theta_f"])
                # toc_val = copy(appendageParams["toc"])
                # alfa0_val = copy(appendageParams["alfa0"])

                # dIdxDV = AD.gradient(backend, (xtheta, xalpha) ->
                #         cost_funcsFromDVsOM(ptVec, nodeConn, displacementsCol, claVec, xtheta, toc_val, xalpha, appendageParams, solverOptions), theta_val, alfa0_val)

            elseif uppercase(mode) == "FAD"
                error("FAD not implemented yet")
            end

        end

        println("Sensitivity time:\t$(tFunc)")

        dfdxParams = Dict()
        DesignVariables = ["displCol", "cla", "theta_f", "toc", "alfa0"] # every DV except ptVec in order
        # for (ii, dvkey) in enumerate(alldesignvariables) # old
        for (ii, dvkey) in enumerate(DesignVariables)
            dfdxParams[dvkey] = dIdxDV[1+ii]
        end

        funcsSens = Dict(
            "mesh" => reshape(dIdxDV[1], 3, NPT),
            "params" => dfdxParams,
        )

        funcsSensOut[evalFuncSensKey] = funcsSens

    end

    return funcsSensOut

end

end # end module