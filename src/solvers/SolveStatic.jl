# --- Julia ---

# @File    :   SolveStatic.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations w/ steady aero/hydrodynamics and compute gradients

module SolveStatic
"""
Static hydroelastic solver
"""

# --- Public functions ---
export solveFromDVs

# --- PACKAGES ---
using AbstractDifferentiation: AbstractDifferentiation as AD
using FiniteDifferences
using Zygote
using ReverseDiff
using ChainRulesCore
using LinearAlgebra
using Statistics
using JSON
using JLD2
using Printf, DelimitedFiles
using Debugger


# --- DCFoil modules ---
using ..DCFoil: RealOrComplex, DTYPE
using ..InitModel
using ..Preprocessing
using ..FEMMethods: FEMMethods, StructMesh
using ..BeamProperties
using ..EBBeam: NDOF, UIND, VIND, WIND, ΦIND, ΨIND, ΘIND
using ..HydroStrip: HydroStrip
using ..SolutionConstants: SolutionConstants, XDIM, YDIM, ZDIM, DCFoilSolverParams
using ..DesignConstants: DesignConstants, SORTEDDVS, DynamicFoil, CONFIGS
using ..SolverRoutines: SolverRoutines
using ..Utilities: Utilities
using ..DCFoilSolution: DCFoilSolution, StaticSolution
using ..TecplotIO: TecplotIO
using ..DesignVariables: allDesignVariables
using ..ComputeFunctions

# ==============================================================================
#                         COMMON TERMS
# ==============================================================================
const ELEMTYPE = "COMP2"
const loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solveFromDVs(
    SOLVERPARAMS::DCFoilSolverParams, FEMESH::StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, DVDictList,
    solverOptions;
    iComp=1, CLMain=0.0
)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    """

    DVDict = DVDictList[iComp]
    appendageOptions = solverOptions["appendageList"][iComp]
    outputDir = solverOptions["outputDir"]

    # Initial guess on unknown deflections (excluding BC nodes)
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    q_ss0 = FEMMethods.solve_structure(SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], fTractions[1:end.∉[DOFBlankingList]])

    if lowercase(solverOptions["res_jacobian"]) == "cs"
        mode = "CS"
    elseif lowercase(solverOptions["res_jacobian"]) == "rad"
        mode = "RAD"
    elseif lowercase(solverOptions["res_jacobian"]) == "analytic"
        mode = "Analytic"
    end

    # Actual solve
    qSol, _ = SolverRoutines.converge_resNonlinear(compute_residualsFromDV, compute_∂r∂uFromDV, q_ss0, DVDictList;
        is_verbose=true,
        solverParams=SOLVERPARAMS,
        appendageOptions=appendageOptions,
        solverOptions=solverOptions,
        mode=mode,
        iComp=iComp,
        CLMain=CLMain,
    )
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE; appendageOptions=appendageOptions)

    # --- Get hydroLoads again on solution ---
    # _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(uSol), structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; 
    # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC


    write_sol(uSol, fHydro, ELEMTYPE, outputDir)

    STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

    return STATSOL
end

function solveFromCoords(
    LECoords, nodeConn, TECoords, SOLVERPARAMS::DCFoilSolverParams, FEMESH::StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, appendageParamsList,
    solverOptions;
    iComp=1, CLMain=0.0
)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    """

    DVDict = appendageParamsList[iComp]
    appendageOptions = solverOptions["appendageList"][iComp]
    outputDir = solverOptions["outputDir"]

    # Initial guess on unknown deflections (excluding BC nodes)
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    q_ss0 = FEMMethods.solve_structure(SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], fTractions[1:end.∉[DOFBlankingList]])

    if lowercase(solverOptions["res_jacobian"]) == "cs"
        mode = "CS"
    elseif lowercase(solverOptions["res_jacobian"]) == "rad"
        mode = "RAD"
    elseif lowercase(solverOptions["res_jacobian"]) == "analytic"
        mode = "Analytic"
    end

    # Hacky way to pass in coords
    x0List = [LECoords, nodeConn, TECoords, appendageParamsList]

    # Actual solve
    qSol, _ = SolverRoutines.converge_resNonlinear(compute_residualsFromCoords, compute_∂r∂u, q_ss0, x0List;
        is_verbose=true,
        solverParams=SOLVERPARAMS,
        appendageOptions=appendageOptions,
        solverOptions=solverOptions,
        mode=mode,
        iComp=iComp,
        CLMain=CLMain,
    )
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE; appendageOptions=appendageOptions)

    # --- Get hydroLoads again on solution ---
    # _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(uSol), structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; 
    # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC


    write_sol(uSol, fHydro, ELEMTYPE, outputDir)

    STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

    return STATSOL
end

function setup_problemFromDVDict(
    DVDictList, appendageOptions::Dict, solverOptions::Dict;
    iComp=1, CLMain=0.0, verbose=false
)
    """
    """
    DVDict = DVDictList[iComp]

    WING, STRUT, _ = InitModel.init_modelFromDVDict(DVDict, solverOptions, appendageOptions)

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if appendageOptions["config"] == "wing"
        STRUT = WING # just to make the code work
    end
    structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"];
        config=appendageOptions["config"], nElStrut=STRUT.nNodes - 1, spanStrut=DVDict["s_strut"], rake=DVDict["rake"])

    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, WING.chord, DVDict["toc"], WING.ab, DVDict["x_ab"], DVDict["theta_f"], zeros(10, 2))

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, DVDict["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, ab_strut=STRUT.ab, x_αb_strut=DVDict["x_ab_strut"], verbose=verbose)
    alphaCorrection::DTYPE = 0.0
    if iComp > 1
        alphaCorrection = HydroStrip.correct_downwash(iComp, CLMain, DVDictList, solverOptions)
    end
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(FEMESH, WING, size(globalM)[1], DVDict["sweep"], WING.U∞, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT)
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
    # K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, DOFBlankingList)
    derivMode = "RAD"
    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, globalK, AIC, planformArea, alphaCorrection)

    return WING, STRUT, SOLVERPARAMS, FEMESH
end

function setup_problemFromCoords(
    LECoords, nodeConn, TECoords, appendageParams, appendageOptions, solverOptions;
    verbose=false
)
    """
    Full setup
    """

    # This part involves a lifting line solution, which takes about 1-2 seconds
    WING, STRUT, _, FEMESH, LLOutputs, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if !(appendageOptions["config"] == "t-foil")
        STRUT = WING # just to make the code work
    elseif !(appendageOptions["config"] in CONFIGS)
        error("Invalid configuration")
    end

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"], verbose=verbose)
    alphaCorrection::DTYPE = 0.0

    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, size(globalM)[1], appendageParams["sweep"], FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, use_nlll=solverOptions["use_nlll"])

    # DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, globalK, AIC, planformArea, alphaCorrection)

    return WING, STRUT, SOLVERPARAMS, FEMESH
end


function write_sol(
    states::Vector{DTYPE}, fHydro::Vector{DTYPE}, elemType="bend", outputDir="./OUTPUT/"
)
    """
    Inputs
    ------
    states: vector of structural states from the [K]{u} = {f}
    """

    # --- Make output directory ---
    workingOutputDir = outputDir * "static/"
    mkpath(workingOutputDir)



    if elemType == "bend"
        nDOF = 2
    elseif elemType == "bend-twist"
        nDOF = 3
        theta = states[nDOF:nDOF:end]
        Moments = fHydro[nDOF:nDOF:end]
    elseif elemType == "BT2"
        nDOF = 4
        theta = states[3:nDOF:end]
        Moments = fHydro[3:nDOF:end]
        W = states[1:nDOF:end]
        Lift = fHydro[1:nDOF:end]
    elseif elemType == "COMP2"
        nDOF = 9
        theta = states[5:nDOF:end]
        Moments = fHydro[5:nDOF:end]
        W = states[3:nDOF:end]
        Lift = fHydro[3:nDOF:end]
    else
        error("Invalid element type")
    end


    # --- Write bending ---
    fname = workingOutputDir * "bending.dat"
    outfile = open(fname, "w")
    # write(outfile, "Bending\n")
    for wⁿ ∈ W
        write(outfile, string(wⁿ, "\n"))
    end
    close(outfile)

    # --- Write twist ---
    if @isdefined(theta)
        fname = workingOutputDir * "twisting.dat"
        outfile = open(fname, "w")
        # write(outfile, "Twist\n")
        for thetaⁿ ∈ theta
            write(outfile, string(thetaⁿ, "\n"))
        end
        close(outfile)
    end

    # --- Write lift and moments ---
    fname = workingOutputDir * "lift.dat"
    outfile = open(fname, "w")
    for Fⁿ in Lift
        write(outfile, string(Fⁿ, "\n"))
    end
    close(outfile)
    fname = workingOutputDir * "moments.dat"
    outfile = open(fname, "w")
    for Mⁿ in Moments
        write(outfile, string(Mⁿ, "\n"))
    end
    close(outfile)

end

function write_tecplot(
    DVDict::Dict, STATICSOL, FEMESH, outputDir="./OUTPUT/";
    appendageOptions=Dict("config" => "wing"), solverOptions=Dict(), iComp=1
)
    """
    General purpose tecplot writer wrapper for flutter solution
    """
    TecplotIO.write_deflections(DVDict, STATICSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)

end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function get_evalFunc(
    evalFunc, states, SOL::StaticSolution, ptVec, nodeConn, DVDict;
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)
    """
    Works for a single cost function
    """

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    costFuncs = compute_funcs(evalFunc, states, SOL, LECoords, TECoords, nodeConn, DVDict;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

    return costFuncs[evalFunc]
end

function compute_funcs(
    evalFunc, states::Vector, SOL::StaticSolution,
    LECoords, TECoords, nodeConn, appendageParams;
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)

    # ************************************************
    #     RECOMPUTE FROM u AND x
    # ************************************************

    # --- Now update the inputs from DVDict ---
    α₀ = appendageParams["alfa0"]
    rake = appendageParams["rake"]

    ptVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)
    WING, STRUT, constants, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, appendageParams, appendageOptions, solverOptions)

    # solverOptions["debug"] = false
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    forces, _, _ = HydroStrip.integrate_hydroLoads(states, constants.AICmat, α₀, rake, DOFBlankingList, constants.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    meanChord = mean(WING.chord)


    # ************************************************
    #     CALCULATE COST FUNCTIONS
    # ************************************************
    # There should be no reason why the density or flow speed is diff 
    # between 'foil' data structures in the multi-appendage case
    # so this line is ok
    qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************
    costFuncs = Dict() # initialize empty costFunc dictionary
    if appendageOptions["config"] == "wing"
        ADIM = constants.planformArea
    elseif appendageOptions["config"] == "t-foil" || appendageOptions["config"] == "full-wing"
        ADIM = 2 * constants.planformArea
    end

    costFuncs["wtip"] = ComputeFunctions.compute_maxtipbend(states)
    costFuncs["psitip"] = ComputeFunctions.compute_maxtiptwist(states)
    costFuncs["lift"], costFuncs["cl"] = ComputeFunctions.compute_lift(forces, qdyn, ADIM)
    costFuncs["moment"], costFuncs["cmy"] = ComputeFunctions.compute_momy(forces, qdyn, ADIM, meanChord)
    costFuncs["cd"], costFuncs["cdi"], costFuncs["cdw"], costFuncs["cdpr"], costFuncs["cdj"], costFuncs["cds"] =
        ComputeFunctions.compute_calmwaterdrag(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, ADIM, costFuncs["cl"], meanChord)
    costFuncs["cofz"], costFuncs["comy"] = ComputeFunctions.compute_centerofforce(forces, FEMESH)

    return costFuncs
end

function compute_funcsFromfhydro()

end

function compute_solFromDVDict(
    DVDictList, solverOptions::Dict, evalFuncs::Vector{String};
    iComp=1, CLMain=0.0
)
    """
    Wrapper function to do primal solve and return solution struct
    """

    appendageOptions = solverOptions["appendageList"][iComp]
    WING, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromDVDict(DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain, verbose=true)

    SOL = solveFromDVs(SOLVERPARAMS, FEMESH, WING, STRUT, DVDictList, solverOptions; iComp=iComp, CLMain=CLMain)

    return SOL
end

function compute_solFromCoords(LECoords, nodeConn, TECoords, appendageParamsList, solverOptions)

    WING, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, appendageParamsList[1], solverOptions["appendageList"][1], solverOptions)
    SOL = solveFromCoords(LECoords, nodeConn, TECoords, SOLVERPARAMS, FEMESH, WING, STRUT, appendageParamsList, solverOptions)

    return SOL
end

function cost_funcsFromDVs(
    DVDict::Dict, iComp::Int64, solverOptions::Dict, evalFuncsList::Vector{String};
    DVDictList=[], CLMain=0.0
)
    """
    Do primal solve with function signature compatible with Zygote
    """

    appendageOptions = solverOptions["appendageList"][iComp]
    # Setup
    DVDictList[iComp] = DVDict
    FOIL, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromDVDict(DVDictList, appendageOptions, solverOptions;
        verbose=false, iComp=iComp, CLMain=CLMain)
    # Solve
    SOL = solveFromDVs(SOLVERPARAMS, FEMESH, FOIL, STRUT, DVDictList, appendageOptions, solverOptions;
        iComp=iComp, CLMain=CLMain)
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    costFuncs = get_evalFunc(
        evalFuncsList, SOL.structStates, SOL, DVVec, DVLengths;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)
    return costFuncs
end

function cost_funcsFromPtVec(
    ptVec, nodeConn, appendageParams, iComp::Int64, solverOptions::Dict, evalFunc::String;
    DVDictList=[], CLMain=0.0
)
    """
    Do primal solve with function signature compatible with Zygote
    """

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    appendageOptions = solverOptions["appendageList"][iComp]

    # Setup
    DVDictList[iComp] = appendageParams
    FOIL, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, appendageParams, appendageOptions, solverOptions; verbose=false)

    # Solve
    SOL = solveFromCoords(LECoords, nodeConn, TECoords, SOLVERPARAMS, FEMESH, FOIL, STRUT, DVDictList, solverOptions;
        iComp=iComp)

    costFuncs = get_evalFunc(
        evalFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

    return costFuncs
end

function evalFuncsSens(
    STATSOLLIST::Vector, evalFuncSens::String, DVDictList::Vector, GridStruct, FEMESHLIST::Vector, solverOptions::Dict;
    mode="FiDi", CLMain=0.0
)
    """
    Wrapper to compute total sensitivities for one cost func!
    pts are ordered LE then TE

    The majority of computational cost comes from the static lifting line setup (nearly 1 sec each time), so avoid this as much as possible
    """
    println("===================================================================================================")
    println("        STATIC SENSITIVITIES: ", mode)
    println("===================================================================================================")

    # Initialize output
    funcsSensList::Vector = []

    solverOptions["debug"] = false
    LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
    ptVec, mm, NPT = Utilities.unpack_coords(GridStruct.LEMesh, GridStruct.TEMesh)

    # --- Loop foils ---
    for iComp in eachindex(DVDictList)

        DVDict = DVDictList[iComp]

        dfdxstruct = zeros(DTYPE, length(allDesignVariables))

        if uppercase(mode) == "FIDI" # use finite differences the stupid way
            # dh = 1e-4
            dh = 1e-5
            idh = 1 / dh
            println("step size: ", dh)

            dfdxPt = zeros(DTYPE, length(ptVec))

            f_i = cost_funcsFromPtVec(ptVec, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                DVDictList=DVDictList, CLMain=CLMain
            )
            for ii in eachindex(ptVec)
                ptVecwork = copy(ptVec)
                ptVecwork[ii] += dh
                f_f = cost_funcsFromPtVec(ptVecwork, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                    DVDictList=DVDictList, CLMain=CLMain
                )
                ptVecwork[ii] -= dh
                dfdxPt[ii] = (f_f - f_i) * idh
            end


            DVDict["theta_f"] += dh
            f_f = cost_funcsFromPtVec(ptVec, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                DVDictList=DVDictList, CLMain=CLMain
            )
            DVDict["theta_f"] -= dh

            dfdxstruct[1] = (f_f - f_i) * idh

            dfdxPt = reshape(dfdxPt, 3, NPT)
            funcsSens = Dict(
                "mesh" => dfdxPt,
                "struct" => dfdxstruct
            )
            println("Finite difference sensitivities for $(evalFuncSens): ", funcsSens)

            writedlm("funcsSens-mesh-$(evalFuncSens)-$(mode).csv", funcsSens["mesh"], ',')
            writedlm("funcsSens-struct-$(evalFuncSens)-$(mode).csv", funcsSens["struct"], ',')

            funcsSensList = push!(funcsSensList, funcsSens)

        elseif uppercase(mode) == "RAD" #RAD the whole thing # BUSTED with mutating arrays :(
            backend = AD.ZygoteBackend()
            @time funcsSens, = AD.gradient(backend, (x) -> cost_funcsFromPtVec(
                    x, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                    DVDictList=DVDictList, CLMain=CLMain),
                ptVec)

        elseif uppercase(mode) == "ADJOINT"

            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

            @time ∂r∂xPt, ∂r∂xStruct = compute_∂r∂x(STATSOL.structStates, DVDictList, LECoords, TECoords, nodeConn;
                # mode="FiDi", # about 981 sec
                # mode="RAD", # about 282 sec
                mode="ANALYTIC", # 10 sec
                appendageOptions=appendageOptions, solverOptions=solverOptions, CLMain=CLMain, iComp=iComp)

            println("Computing ∂r∂u...")
            @time ∂r∂u = compute_∂r∂u(u, LECoords, TECoords, nodeConn, "Analytic";
                appendageParamsList=DVDictList, solverParams=solverParams, solverOptions=solverOptions, appendageOptions=appendageOptions, iComp=iComp)

            # This is correct btwn fidi and rad so this probably isn't the bug
            @time ∂f∂u = compute_∂f∂u(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                # mode="FiDi", # 200 sec
                mode="RAD", # 100 sec
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            @time ∂f∂xPt, ∂f∂xStruct = compute_∂f∂x(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                # mode="RAD", # 100 sec
                mode="FiDi",
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            println("+---------------------------------+")
            println("| Computing adjoint: $(evalFuncSens)")
            println("+---------------------------------+")
            # should be (n_u x n_f) a column vector!
            ∂f∂uT = transpose(∂f∂u)
            psiVec = compute_adjointVec(∂r∂u, ∂f∂uT;
                solverParams=solverParams, appendageOptions=appendageOptions)


            # --- Compute total sensitivities ---
            # dfdxpt = ∂f∂x - transpose(psiMat) * ∂r∂x
            # Transpose the adjoint vector so it's now a row vector
            dfdxPt = ∂f∂xPt - (transpose(psiVec) * ∂r∂xPt)
            dfdxStruct = ∂f∂xStruct - (transpose(psiVec) * ∂r∂xStruct)

            # writedlm("∂f∂ududx.csv", (transpose(psiVec) * ∂r∂xPt), ',')

            funcsSens = Dict(
                "mesh" => reshape(dfdxPt, 3, NPT),
                "struct" => dfdxStruct
            )
            println("Adjoint sensitivities for $(evalFuncSens): ", funcsSens)

            writedlm("funcsSens-mesh-$(evalFuncSens)-$(mode).csv", funcsSens["mesh"], ',')
            writedlm("funcsSens-struct-$(evalFuncSens)-$(mode).csv", funcsSens["struct"], ',')


            push!(funcsSensList, funcsSens)

        elseif uppercase(mode) == "DIRECT"

            println("Computing direct for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

            @time ∂r∂xPt = compute_∂r∂x(STATSOL.structStates, DVDictList, LECoords, TECoords, nodeConn;
                mode="ANALYTIC",
                appendageOptions=appendageOptions, solverOptions=solverOptions, CLMain=CLMain, iComp=iComp)
            @time ∂r∂u = compute_∂r∂u(u, LECoords, TECoords, nodeConn, "Analytic";
                appendageParamsList=DVDictList, solverParams=solverParams, solverOptions=solverOptions, appendageOptions=appendageOptions, iComp=iComp)
            println("+---------------------------------+")
            println("| Computing direct: $(evalFuncSens)")
            println("+---------------------------------+")
            phiMat = compute_directMatrix(∂r∂u, ∂r∂xPt;
                solverParams=solverParams)

            @time ∂f∂u = compute_∂f∂u(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            @time ∂f∂xPt = compute_∂f∂x(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )

            # --- Compute total sensitivities ---
            # funcsSens = ∂f∂x - ∂f∂u * [ϕ]
            dfdxPt = ∂f∂xPt - (∂f∂u[:, 1:end.∉[DOFBlankingList]] * phiMat)
            funcsSens = reshape(dfdxPt, 3, NPT)
            writedlm("funcsSens-$(evalFuncSens)-$(mode).csv", funcsSens, ',')

            push!(funcsSensList, funcsSens)

        else
            error("Invalid mode")
        end


    end

    return funcsSensList
end

function compute_∂f∂x(
    costFunc::String, SOL::StaticSolution, ptVec, nodeConn, appendageParams::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)

    println("Computing ∂f∂x in $(mode)...")
    ∂f∂xPt = zeros(DTYPE, length(ptVec))
    ∂f∂xStruct = zeros(DTYPE, length(allDesignVariables))

    # Force an automatic initial eval so less code duplication
    dh = 1e-5
    f_i = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
    )


    DVDictList[iComp] = appendageParams
    if uppercase(mode) == "ANALYTIC"
        hydromode = "ANALYTIC"
        println("Computing hydro derivatives in $(hydromode)")
        dfstaticdXpt = compute_dfhydrostaticdXpt(u, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode=hydromode)

        # TODO: PICKUP HERE
        ∂f∂fstatic = compute_∂costFunc∂fhydro(appendageOptions, appendageParams, solverOptions)

        # Any function can be written as some function of the fluid or structural states
        ∂f∂xPt = ∂f∂fstatic * dfstaticdXpt

    elseif uppercase(mode) == "RAD" # WORKS BUT BUGGING WITH NANs prob from lifting line
        backend = AD.ZygoteBackend()
        ∂f∂xPt, = AD.gradient(
            backend,
            x -> get_evalFunc(costFunc, SOL.structStates, SOL, x, nodeConn, appendageParams;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain),
            ptVec,
        )
    elseif uppercase(mode) == "FIDI"

        println("step size: ", dh)


        for ii in eachindex(ptVec)
            ptVec[ii] += dh
            f_f = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            ptVec[ii] -= dh

            ∂f∂xPt[ii] = (f_f - f_i) / dh
        end


    end

    # --- Struct ---
    appendageParams["theta_f"] += dh
    f_f = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
    )
    appendageParams["theta_f"] -= dh
    ∂f∂xStruct = (f_f - f_i) / dh

    println("writing ∂f∂xPt-$(mode).csv")
    writedlm("∂f∂xPt-$(mode).csv", ∂f∂xPt, ',')

    ∂f∂xPt = reshape(∂f∂xPt, 1, length(∂f∂xPt))

    return ∂f∂xPt, ∂f∂xStruct
end

function compute_∂costFunc∂fhydro(appendageOptions, appendageParams, solverOptions)

end

function compute_∂f∂u(
    costFunc::String, SOL::StaticSolution, ptVec, nodeConn, appendageParams::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)
    """
    Compute the gradient of the cost functions with respect to the structural states
    SOL is the solution struct at the current design point
    """

    println("Computing ∂f∂u...")
    ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
    DVDictList[iComp] = appendageParams
    # Do analytic
    if uppercase(mode) == "ANALYTIC"
        # TODO: PICKUP HERE
        ∂f∂fstatic = compute_∂costFunc∂fhydro()

        # ∂f∂udirect = costfuncsdpending directly on the structural states (tip bend, etc.)
        ∂f∂u = ∂f∂fstatic * dfstaticdu + ∂f∂udirect
    elseif uppercase(mode) == "RAD" # works
        backend = AD.ZygoteBackend()
        ∂f∂u, = AD.gradient(
            backend,
            u -> get_evalFunc(
                costFunc, u, SOL, ptVec, nodeConn, appendageParams;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            ),
            SOL.structStates,
        )
        ∂f∂u = reshape(∂f∂u, 1, length(∂f∂u))
    elseif uppercase(mode) == "FIDI" # Finite difference
        dh = 1e-4
        idh = 1 / dh
        println("step size:", dh)

        for ii in eachindex(SOL.structStates)
            r_i = SolveStatic.get_evalFunc(
                costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] += dh
            r_f = SolveStatic.get_evalFunc(
                costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] -= dh

            ∂f∂u[1, ii] = (r_f - r_i) * idh
        end
    else
        error("Invalid mode")
    end

    # Save matrix for debugging?
    # writedlm("∂f∂u-$(mode).csv", ∂f∂u, ',')

    return ∂f∂u
end

function compute_∂r∂x(
    allStructStates, appendageParamsList, LECoords, TECoords, nodeConn;
    mode="FiDi", appendageOptions=nothing,
    solverOptions=nothing, CLMain, iComp=1,
)
    """
    Partial derivatives of residuals with respect to design variables w/o reconverging the solution
    """

    println("Computing ∂r∂x in $(mode) mode...")
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    u = allStructStates[1:end.∉[DOFBlankingList]]

    ptVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)
    appendageParams = appendageParamsList[iComp]

    ∂r∂xPt = zeros(DTYPE, length(u), length(ptVec))

    if uppercase(mode) == "FIDI" # Finite difference

        backend = AD.FiniteDifferencesBackend()
        ∂r∂xPt, = AD.jacobian(
            backend,
            x -> SolveStatic.compute_residualsFromCoords(
                u,
                x,
                nodeConn,
                appendageParamsList,
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
            ),
            ptVec, # compute deriv at this DV
        )

        # ************************************************
        #     Struct
        # ************************************************
        dh = 1e-4
        f_i = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] += dh
        f_f = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] -= dh
        ∂r∂xStruct = (f_f - f_i) / 1e-4

    elseif uppercase(mode) == "CS" # works but slow (4 sec)
        dh = 1e-100
        println("step size: ", dh)

        ∂r∂xPt = zeros(DTYPE, length(u), nn)

        ∂r∂xPt = perturb_coordsForResid(∂r∂xPt, 1im * dh, u, LECoords, TECoords, nodeConn, appendageParamsList, appendageOptions, solverOptions, iComp)

    elseif uppercase(mode) == "ANALYTIC"

        # println("struct partials")
        # This seems good
        ∂Kss∂x_u = compute_∂KssU∂x(u, ptVec, nodeConn, appendageOptions, appendageParamsList[1], solverOptions)
        # writedlm("dKssdx.csv", ∂Kss∂x_u, ',')

        # println("hydro partials")
        # It is strange that this is zero
        # hydromode = "FiDi" # This appears to be right
        hydromode = "ANALYTIC"
        println("Computing hydro derivatives in $(hydromode)")
        dKffdXpt_u = compute_dfhydrostaticdXpt(u, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode=hydromode)

        # ∂r∂x = u ∂K∂X - dfdX
        ∂r∂xPt = (∂Kss∂x_u - dKffdXpt_u)

        # ************************************************
        #     Struct
        # ************************************************
        dh = 1e-4
        f_i = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] += dh
        f_f = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] -= dh
        ∂r∂xStruct = (f_f - f_i) / 1e-4


    elseif uppercase(mode) == "FAD" # this is fked
        error("Not implemented")
    elseif uppercase(mode) == "RAD" # WORKS
        backend = AD.ZygoteBackend()
        ∂r∂xPt, = AD.jacobian(
            backend,
            x -> SolveStatic.compute_residualsFromCoords(
                u,
                x,
                nodeConn,
                appendageParamsList;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
            ),
            ptVec, # compute deriv at this DV
        )
    else
        error("Invalid mode")
    end

    writedlm("drdxPt-$(mode).csv", ∂r∂xPt, ',')
    return ∂r∂xPt, ∂r∂xStruct
end

function compute_∂KssU∂x(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
    """
    Derivative of structural stiffness matrix with respect to design variables times structural states
    """

    ∂KssU∂x = zeros(DTYPE, length(structStates), length(ptVec))

    # CS is faster than fidi, but RAD will probably be best later...? 2.4sec
    dh = 1e-100
    ptVecWork = complex(ptVec)
    for ii in eachindex(ptVec)
        ptVecWork[ii] += 1im * dh
        f_f = compute_KssU(structStates, ptVecWork, nodeConn, appendageOptions, appendageParams, solverOptions)
        ptVecWork[ii] -= 1im * dh
        ∂KssU∂x[:, ii] = imag(f_f) / dh
    end

    return ∂KssU∂x
end

function compute_KssU(u, xVec, nodeConn, appendageOptions, appendageParams, solverOptions)
    LECoords, TECoords = Utilities.repack_coords(xVec, 3, length(xVec) ÷ 3)

    midchords, chordLengths, spanwiseVectors = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

    if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
        print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

        α₀ = appendageParams["alfa0"]
        sweepAng = appendageParams["sweep"]
        rake = appendageParams["rake"]
        span = appendageParams["s"] * 2
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]
        depth0 = appendageParams["depth0"]

        toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
    else
        α₀ = appendageParams["alfa0"]
        sweepAng = appendageParams["sweep"]
        rake = appendageParams["rake"]
        span = appendageParams["s"] * 2
        toc::Vector{RealOrComplex} = appendageParams["toc"]
        ab::Vector{RealOrComplex} = appendageParams["ab"]
        x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        toc_strut = appendageParams["toc_strut"]
        ab_strut = appendageParams["ab_strut"]
        x_ab_strut = appendageParams["x_ab_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]
        depth0 = appendageParams["depth0"]
    end

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, zeros(10, 2))

    WING, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    globalK, _, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"])

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    Kmat = globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    f = Kmat * u
    return f
end

function compute_dfhydrostaticdXpt(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="ANALYTIC")

    dfstaticdXpt = zeros(DTYPE, length(structStates), length(ptVec))

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, ELEMTYPE; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(allStructuralStates, appendageParams, ELEMTYPE;
        appendageOptions=appendageOptions)

    if uppercase(mode) == "ANALYTIC"
        # ************************************************
        #     dcla/dX
        # ************************************************
        # This is a matrix
        # ncla x nX
        hydromode = "IMPLICIT" #
        # hydromode = "FiDi" # slow 2024-11-20 IBoundary nodes pollute the derivative
        #  but there may also be another bug further in this chain
        println("Computing dcladXpt in $(hydromode) mode...")
        dcladXpt = HydroStrip.compute_dcladX(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode=hydromode)

        # writedlm("dcladXpt-$(hydromode).csv", dcladXpt, ",")

        # ************************************************
        #     dKff/dcla
        # ************************************************
        # This is a multidimensional array
        # nu x nu x ncla
        DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
        dim = length(structStates) + length(DOFBlankingList)

        LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
        midchords, chordLengths, spanwiseVectors = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

        structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, appendageParams, appendageOptions)
        if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
            print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

            α₀ = appendageParams["alfa0"]
            sweepAng = appendageParams["sweep"]
            rake = appendageParams["rake"]
            span = appendageParams["s"] * 2
            zeta = appendageParams["zeta"]
            theta_f = appendageParams["theta_f"]
            beta = appendageParams["beta"]
            s_strut = appendageParams["s_strut"]
            c_strut = appendageParams["c_strut"]
            theta_f_strut = appendageParams["theta_f_strut"]
            depth0 = appendageParams["depth0"]

            toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
        else
            α₀ = appendageParams["alfa0"]
            sweepAng = appendageParams["sweep"]
            rake = appendageParams["rake"]
            span = appendageParams["s"] * 2
            toc::Vector{RealOrComplex} = appendageParams["toc"]
            ab::Vector{RealOrComplex} = appendageParams["ab"]
            x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
            zeta = appendageParams["zeta"]
            theta_f = appendageParams["theta_f"]
            beta = appendageParams["beta"]
            s_strut = appendageParams["s_strut"]
            c_strut = appendageParams["c_strut"]
            toc_strut = appendageParams["toc_strut"]
            ab_strut = appendageParams["ab_strut"]
            x_ab_strut = appendageParams["x_ab_strut"]
            theta_f_strut = appendageParams["theta_f_strut"]
            depth0 = appendageParams["depth0"]
        end
        AEROMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, zeros(10, 2))
        FOIL, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

        dKffdcla = HydroStrip.compute_∂Kff∂cla(AEROMESH, FOIL, STRUT, dim, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FIDI")

        ∂Kff∂Xpt = HydroStrip.compute_∂Kff∂Xpt(dim, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FIDI")

        #  d K_ff     ∂ K_ff    d cl_a     ∂ K_ff
        # -------  = --------- --------- + ------
        #  d   X      ∂ cl_a    d  X       ∂ X
        dKffdXpt = dKffdcla * dcladXpt + ∂Kff∂Xpt


        # Over all design vars
        for ii in eachindex(ptVec)
            ∇Kff = reshape(dKffdXpt[:, ii], dim, dim)

            dfstaticdXpt[:, ii] = ∇Kff[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] * foilTotalStates[1:end.∉[DOFBlankingList]]
        end

    elseif uppercase(mode) == "FIDI" # This seems to fix things!
        dh = 1e-5

        Kff_i, cla_i = compute_Kff(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

        dKffdX = zeros(DTYPE, length(structStates) + 9, length(structStates) + 9, length(ptVec))
        dcladXpt = zeros(DTYPE, 40, length(ptVec))
        dKffdXpt = zeros(DTYPE, (length(structStates) + 9)^2, length(ptVec))
        for ii in eachindex(ptVec)

            ptVec[ii] += dh

            Kff_f, cla_f = compute_Kff(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

            ptVec[ii] -= dh

            # ∂fstatic∂X[:, ii] = (f_f - f_i) / dh
            dcladXpt[:, ii] = (cla_f - cla_i) / dh


            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            dKfu = (Kff_f - Kff_i) * foilTotalStates
            dKffdX[:, :, ii] = (Kff_f - Kff_i) / dh
            dfstaticdXpt[:, ii] = dKfu[1:end.∉[DOFBlankingList]] / dh
            # println(maximum(dKffdX[:, :, ii]))

            dKffdXpt[:, ii] = (vec(Kff_f) - vec(Kff_i)) / dh

        end

    end

    return dfstaticdXpt
end

function compute_KffU(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    midchords, chordLengths, spanwiseVectors = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

    if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
        print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

        α₀ = appendageParams["alfa0"]
        sweepAng = appendageParams["sweep"]
        rake = appendageParams["rake"]
        span = appendageParams["s"] * 2
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]
        depth0 = appendageParams["depth0"]

        toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
    else
        α₀ = appendageParams["alfa0"]
        sweepAng = appendageParams["sweep"]
        rake = appendageParams["rake"]
        span = appendageParams["s"] * 2
        toc::Vector{RealOrComplex} = appendageParams["toc"]
        ab::Vector{RealOrComplex} = appendageParams["ab"]
        x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        toc_strut = appendageParams["toc_strut"]
        ab_strut = appendageParams["ab_strut"]
        x_ab_strut = appendageParams["x_ab_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]
        depth0 = appendageParams["depth0"]
    end

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, zeros(10, 2))

    LLOutputs, LLSystem, FlowCond = InitModel.init_staticHydro(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    statWingStructModel, statStrutStructModel = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    WING = DesignConstants.DynamicFoil(
        statWingStructModel.mₛ, statWingStructModel.Iₛ, statWingStructModel.EIₛ, statWingStructModel.EIIPₛ, statWingStructModel.GJₛ, statWingStructModel.Kₛ, statWingStructModel.Sₛ, statWingStructModel.EAₛ,
        statWingStructModel.eb, statWingStructModel.ab, statWingStructModel.chord, statWingStructModel.nNodes, statWingStructModel.constitutive,
        [0, 1.0], [0, 1.0]
    )

    STRUT = nothing

    dim = NDOF * (size(elemConn)[1] + 1)

    _, _, _, AIC, _, _ = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, dim, appendageParams["sweep"], FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, use_nlll=solverOptions["use_nlll"])

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, ELEMTYPE; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(allStructuralStates, appendageParams, ELEMTYPE; appendageOptions=appendageOptions,)

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    fFull = -AIC * foilTotalStates
    f = fFull[1:end.∉[DOFBlankingList]]

    return f, -AIC
end

function compute_Kff(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    midchords, chordLengths, spanwiseVectors = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

    toc::Vector{RealOrComplex} = appendageParams["toc"]
    ab::Vector{RealOrComplex} = appendageParams["ab"]
    x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    toc_strut = appendageParams["toc_strut"]
    ab_strut = appendageParams["ab_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, zeros(10, 2))

    LLOutputs, LLSystem, FlowCond = InitModel.init_staticHydro(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    statWingStructModel, _ = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    WING = DesignConstants.DynamicFoil(
        statWingStructModel.mₛ, statWingStructModel.Iₛ, statWingStructModel.EIₛ, statWingStructModel.EIIPₛ, statWingStructModel.GJₛ, statWingStructModel.Kₛ, statWingStructModel.Sₛ, statWingStructModel.EAₛ,
        statWingStructModel.eb, statWingStructModel.ab, statWingStructModel.chord, statWingStructModel.nNodes, statWingStructModel.constitutive,
        [0, 1.0], [0, 1.0]
    )

    STRUT = nothing

    dim = NDOF * (size(elemConn)[1] + 1)

    _, _, _, AIC, _, _ = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, dim, appendageParams["sweep"], FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, use_nlll=solverOptions["use_nlll"])


    return -AIC, LLOutputs.cla
end

function compute_∂r∂u(
    structuralStates, xLE, xTE, nodeConn, mode="CS";
    appendageParamsList=[], solverParams=nothing, appendageOptions=Dict(), solverOptions=Dict(), iComp=1, CLMain=0.0
)
    """
    Jacobian of residuals with respect to structural states
    EXCLUDING BC NODES

    u - structural states
    """

    if uppercase(mode) == "FIDI" # Finite difference
        # First derivative using 3 stencil points
        backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
        ∂r∂u, = AD.jacobian(
            backend,
            x -> compute_residualsFromCoords(
                x, xLE, xTE, nodeConn, appendageParamsList;
                appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp
            ),
            structuralStates)
        # ∂r∂u = FiniteDifferences.jacobian(forward_fdm(2, 1), compute_residuals, structuralStates)

    elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        # NOTE: a little slow but it is accurate
        # This is a tuple
        backend = AD.ZygoteBackend()
        ∂r∂u, = AD.jacobian(
            backend,
            x -> compute_residualsFromCoords(
                x, xLE, xTE, nodeConn, appendageParamsList;
                appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp
            ),
            structuralStates)
        # ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

        # elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        #     @time ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)

    elseif uppercase(mode) == "CS" # Complex step

        dh = 1e-100
        # println("step size: ", dh)

        ∂r∂u = zeros(DTYPE, length(structuralStates), length(structuralStates))

        # create a complex copy of the structural states
        structuralStatesCS = complex(copy(structuralStates))
        for ii in eachindex(structuralStates)
            structuralStatesCS[ii] += dh * 1im
            r_f = compute_residualsFromCoords(
                structuralStatesCS, xLE, xTE, nodeConn, appendageParamsList;
                appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp
            )
            structuralStatesCS[ii] -= dh * 1im

            ∂r∂u[:, ii] = imag(r_f) / dh
        end

    elseif uppercase(mode) == "ANALYTIC"
        # NOTES:
        # TODO: there is now a bug when using T-FOIL GEOMETRY THAT IT DOES NOT CONVERGE for basic solution
        # TODO: longer term, the AIC is influenced by the structural states b/c of the twist distribution
        # In the case of a linear elastic beam under static fluid loading, 
        # dr/du = Ks + Kf
        # NOTE Kf = AIC matrix
        # where AIC * states = forces on RHS (external)
        # _, _, solverParams = setup_problem(DVDict, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)
        DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
        ∂r∂u_struct = solverParams.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
        ∂r∂u_fluid = solverParams.AICmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
        ∂r∂u::Matrix{DTYPE} = ∂r∂u_struct + ∂r∂u_fluid
        # The behavior of the analytic derivatives is interesting since it takes about 6 NL iterations to 
        # converge to the same solution as the RAD, which only takes 2 NL iterations.
    else
        error("Invalid mode")
    end

    return ∂r∂u
end

function compute_residualsFromCoords(
    structStates, xVec, nodeConn, appendageParamsList;
    appendageOptions=Dict(), solverOptions=Dict(), iComp=1
)
    """
    Compute residual for every node that is not the clamped root node

        r(u, x) = [K]{u} - {f(u)}

        where f(u) is the force vector from the current solution

        Inputs
        ------
        structuralStates : array
            State vector with nodal DOFs and deformations EXCLUDING BCs
        x : dict
            Design variables
    """

    DVDict = appendageParamsList[iComp]
    LECoords, TECoords = Utilities.repack_coords(xVec, 3, length(xVec) ÷ 3)

    # There is probably a nicer way to restructure the code so the order of calls is
    # 1. top level solve call to applies the newton raphson
    # 2.    compute_residuals
    # solverOptions["debug"] = false
    # THIS IS SLOW
    _, _, SOLVERPARAMS = setup_problemFromCoords(LECoords, nodeConn, TECoords, DVDict, appendageOptions, solverOptions)

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, ELEMTYPE; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(
        allStructuralStates, DVDict, ELEMTYPE;
        appendageOptions=appendageOptions, alphaCorrection=SOLVERPARAMS.downwashAngles
    )


    # --- Outputs ---
    F = -SOLVERPARAMS.AICmat * foilTotalStates
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    FOut = F[1:end.∉[DOFBlankingList]]


    # --- Stack them ---
    Knew = SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Felastic = Knew * structStates
    resVec = Felastic - FOut


    # println("u states: ", structStates[end-NDOF:end])
    # println("hydroforces: ", FOut[end-NDOF:end])
    # println("elastic forces: ", Felastic[end-NDOF:end])
    # println("residuals: ", resVec[end-NDOF:end])

    return resVec
end

function compute_directMatrix(
    ∂r∂u::Matrix, ∂r∂x::Matrix;
    solverParams=nothing)
    """
    Computes direct vector
    """

    ϕ = ∂r∂u \ ∂r∂x

    return ϕ
end

function compute_adjointVec(∂r∂u, ∂f∂uT; solverParams=nothing, appendageOptions=nothing)
    """
    Computes adjoint vector
    If ∂f∂uT is a vector, it might be the same as passing in ∂f∂u
    """

    println("WARNING: REMOVING CLAMPED NODE CONTRIBUTION (There's no hydro force at the clamped node...)")
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    ∂f∂uT = ∂f∂uT[1:end.∉[DOFBlankingList]]
    ψ = transpose(∂r∂u) \ ∂f∂uT

    return ψ
end

end # end module