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
using ReverseDiff, ForwardDiff
using ChainRulesCore
using LinearAlgebra
using Statistics
using JSON
using JLD2
using Printf, DelimitedFiles
# using Debugger


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
# function solveFromDVs(
#     SOLVERPARAMS::DCFoilSolverParams, FEMESH::StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, DVDictList,
#     solverOptions;
#     iComp=1, CLMain=0.0
# )
#     """
#     Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
#     """

#     DVDict = DVDictList[iComp]
#     appendageOptions = solverOptions["appendageList"][iComp]
#     outputDir = solverOptions["outputDir"]

#     # Initial guess on unknown deflections (excluding BC nodes)
#     DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
#     fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
#         appendageOptions=appendageOptions, solverOptions=solverOptions)
#     q_ss0 = FEMMethods.solve_structure(SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]], fTractions[1:end.∉[DOFBlankingList]])

#     if lowercase(solverOptions["res_jacobian"]) == "cs"
#         mode = "CS"
#     elseif lowercase(solverOptions["res_jacobian"]) == "rad"
#         mode = "RAD"
#     elseif lowercase(solverOptions["res_jacobian"]) == "analytic"
#         mode = "Analytic"
#     end

#     # Actual solve
#     qSol, _ = SolverRoutines.converge_resNonlinear(compute_residualsFromDV, compute_∂r∂uFromDV, q_ss0, DVDictList;
#         is_verbose=true,
#         solverParams=SOLVERPARAMS,
#         appendageOptions=appendageOptions,
#         solverOptions=solverOptions,
#         mode=mode,
#         iComp=iComp,
#         CLMain=CLMain,
#     )
#     # qSol = q # just use pre-solve solution
#     uSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE; appendageOptions=appendageOptions)

#     # --- Get hydroLoads again on solution ---
#     # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
#     fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
#         appendageOptions=appendageOptions, solverOptions=solverOptions)
#     # global Kf = AIC


#     write_sol(uSol, fHydro, ELEMTYPE, outputDir)

#     STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

#     return STATSOL
# end

function solveFromCoords(
    LECoords, nodeConn, TECoords, SOLVERPARAMS::DCFoilSolverParams, FEMESH::StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, appendageParamsList,
    solverOptions;
    iComp=1, CLMain=0.0
)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    """

    appendageParams = appendageParamsList[iComp]
    appendageOptions = solverOptions["appendageList"][iComp]
    outputDir = solverOptions["outputDir"]

    # Initial guess on unknown deflections (excluding BC nodes)
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, appendageParams["alfa0"], appendageParams["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
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
    # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, appendageParams["alfa0"], appendageParams["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC


    write_sol(uSol, fHydro, ELEMTYPE, outputDir)

    STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

    return STATSOL
end

# function setup_problemFromDVDict(
#     DVDictList, appendageOptions::Dict, solverOptions::Dict;
#     iComp=1, CLMain=0.0, verbose=false
# )
#     """
#     """
#     DVDict = DVDictList[iComp]

#     WING, STRUT, _ = InitModel.init_modelFromDVDict(DVDict, solverOptions, appendageOptions)

#     nNodes = WING.nNodes
#     nElem = nNodes - 1
#     if appendageOptions["config"] == "wing"
#         STRUT = WING # just to make the code work
#     end
#     structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"];
#         config=appendageOptions["config"], nElStrut=STRUT.nNodes - 1, spanStrut=DVDict["s_strut"], rake=DVDict["rake"])

#     FEMESH = FEMMethods.StructMesh(structMesh, elemConn, WING.chord, DVDict["toc"], WING.ab, DVDict["x_ab"], DVDict["theta_f"], zeros(10, 2))

#     globalK, globalM, _ = FEMMethods.assemble(FEMESH, DVDict["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, ab_strut=STRUT.ab, x_αb_strut=DVDict["x_ab_strut"], verbose=verbose)
#     alphaCorrection::DTYPE = 0.0
#     if iComp > 1
#         alphaCorrection = HydroStrip.correct_downwash(iComp, CLMain, DVDictList, solverOptions)
#     end
#     DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
#     # K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, DOFBlankingList)
#     derivMode = "RAD"
#     SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, globalK, AIC, planformArea, alphaCorrection)

#     return WING, STRUT, SOLVERPARAMS, FEMESH
# end

function setup_problemFromCoords(
    LECoords, nodeConn, TECoords, appendageParams, appendageOptions, solverOptions;
    verbose=false
)
    """
    Full setup
    """

    # This part involves a lifting line solution, which takes about 1-2 seconds
    WING, STRUT, _, FEMESH, LLOutputs, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)
    sweepAng = LLSystem.sweepAng

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if !(appendageOptions["config"] == "t-foil")
        STRUT = WING # just to make the code work
    elseif !(appendageOptions["config"] in CONFIGS)
        error("Invalid configuration")
    end

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"], verbose=verbose)
    alphaCorrection::DTYPE = 0.0

    _, _, _, AIC, _ = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, size(globalM)[1], sweepAng, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)
    # areaRef = HydroStrip.compute_areas(FEMESH, WING; appendageOptions=appendageOptions, STRUT=STRUT)
    areaRef = Preprocessing.compute_areas(LECoords, TECoords, nodeConn)

    # DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, copy(Float64.(globalK)), AIC, areaRef, alphaCorrection)

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
    appendageParams::Dict, STATICSOL, FEMESH, outputDir="./OUTPUT/";
    appendageOptions=Dict("config" => "wing"), solverOptions=Dict(), iComp=1
)
    """
    General purpose tecplot writer wrapper for flutter solution
    """
    TecplotIO.write_deflections(appendageParams, STATICSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)

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

    costFuncs = compute_funcs(evalFunc, states, SOL, ptVec, nodeConn, DVDict;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

    return costFuncs
end

function compute_funcs(evalFunc, states::Vector, SOL::StaticSolution,
    ptVec, nodeConn, appendageParams;
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)

    # ************************************************
    #     RECOMPUTE FROM u AND x
    # ************************************************
    FEMESH, forces, WING, areaRef = precompute_funcsux(states, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    meanChord = mean(WING.chord)
    rootChord = WING.chord[1]
    qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************

    fout = compute_funcsFromfhydro(evalFunc, states, forces, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, FEMESH)

    return fout
end

function precompute_funcsux(states, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    # --- Now update the inputs from DVDict ---
    α₀ = appendageParams["alfa0"]
    rake = appendageParams["rake"]

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    WING, STRUT, constants, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, appendageParams, appendageOptions, solverOptions)

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    forces, _, _ = HydroStrip.integrate_hydroLoads(states, constants.AICmat, α₀, rake, DOFBlankingList, constants.downwashAngles, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    return FEMESH, forces, WING, constants.areaRef
end

function compute_funcsFromfhydro(costFunc, states, forces, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, FEMESH)
    """
    states is the structural states with BC
    """

    if costFunc == "wtip"
        wtip = ComputeFunctions.compute_maxtipbend(states)
        fout = wtip
    end
    if costFunc == "psitip"
        psitip = ComputeFunctions.compute_maxtiptwist(states)
        fout = psitip
    end
    if costFunc in ["lift", "cl", "cd"]
        lift, cl = ComputeFunctions.compute_lift(forces, qdyn, areaRef)
        if costFunc == "cl"
            fout = cl
        elseif costFunc == "lift"
            fout = lift
        end
    end
    if costFunc == "kscl"
        kscl = ComputeFunctions.compute_kscl(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        fout = kscl
    end
    if costFunc in ["moment", "cmy"]
        moment, cmy = ComputeFunctions.compute_momy(forces, qdyn, areaRef, meanChord)
        if costFunc == "cmy"
            fout = cmy
        elseif costFunc == "moment"
            fout = moment
        end
    end
    if costFunc in ["cd", "cdi", "cdw", "cdpr", "cdj", "cds"]

        cdi, Di = ComputeFunctions.compute_vortexdrag(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)


        LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
        midchords, chordLengths, _, _ = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, FEMESH.idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)
        aeroSpan = Preprocessing.compute_aeroSpan(midchords, FEMESH.idxTip)
        cdw, cdpr, cdj, cds, dw, dpr, dj, ds = ComputeFunctions.compute_calmwaterdragbuildup(appendageParams, appendageOptions, solverOptions, qdyn, areaRef, aeroSpan, cl, meanChord, rootChord, chordLengths)
        cd = cdi + cdw + cdpr + cdj + cds
        fout = [cd, cdi, cdw, cdpr, cdj, cds]
    end
    if costFunc in ["cofz", "comy"]
        cofz, comy = ComputeFunctions.compute_centerofforce(forces, FEMESH)
        if costFunc == "cofz"
            fout = cofz
        elseif costFunc == "comy"
            fout = comy
        end
    end

    return fout
end

# function compute_solFromDVDict(
#     DVDictList, solverOptions::Dict, evalFuncs::Vector{String};
#     iComp=1, CLMain=0.0
# )
#     """
#     Wrapper function to do primal solve and return solution struct
#     """

#     appendageOptions = solverOptions["appendageList"][iComp]
#     WING, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromDVDict(DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain, verbose=true)

#     SOL = solveFromDVs(SOLVERPARAMS, FEMESH, WING, STRUT, DVDictList, solverOptions; iComp=iComp, CLMain=CLMain)

#     return SOL
# end

function compute_solFromCoords(LECoords, nodeConn, TECoords, appendageParamsList, solverOptions)

    WING, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, appendageParamsList[1], solverOptions["appendageList"][1], solverOptions)
    SOL = solveFromCoords(LECoords, nodeConn, TECoords, SOLVERPARAMS, FEMESH, WING, STRUT, appendageParamsList, solverOptions)

    return SOL
end

# function cost_funcsFromDVs(
#     DVDict::Dict, iComp::Int64, solverOptions::Dict, evalFuncsList::Vector{String};
#     DVDictList=[], CLMain=0.0
# )
#     """
#     Do primal solve with function signature compatible with Zygote
#     """

#     appendageOptions = solverOptions["appendageList"][iComp]
#     # Setup
#     DVDictList[iComp] = DVDict
#     FOIL, STRUT, SOLVERPARAMS, FEMESH = setup_problemFromDVDict(DVDictList, appendageOptions, solverOptions;
#         verbose=false, iComp=iComp, CLMain=CLMain)
#     # Solve
#     SOL = solveFromDVs(SOLVERPARAMS, FEMESH, FOIL, STRUT, DVDictList, appendageOptions, solverOptions;
#         iComp=iComp, CLMain=CLMain)
#     DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
#     costFuncs = get_evalFunc(
#         evalFuncsList, SOL.structStates, SOL, DVVec, DVLengths;
#         appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)
#     return costFuncs
# end

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
    STATSOLLIST::Vector, evalFuncSensList, DVDictList::Vector, GridStruct, FEMESHLIST::Vector, solverOptions::Dict;
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
    funcsSensOut = Dict()

    solverOptions["debug"] = false
    LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
    ptVec, mm, NPT = Utilities.unpack_coords(GridStruct.LEMesh, GridStruct.TEMesh)

    # --- Loop foils ---
    # for iComp in eachindex(DVDictList)
    iComp = 1

    DVDict = DVDictList[iComp]

    dfdxstruct = Dict()

    if uppercase(mode) == "FIDI" # use finite differences the stupid way
        # dh = 1e-2
        # dh = 1e-3
        # dh = 1e-4
        dh = 1e-5
        # dh = 1e-6
        idh = 1 / dh
        println("step size: ", dh)
        for evalFuncSensKey in evalFuncSensList

            dfdxPt = zeros(DTYPE, length(ptVec))

            f_i = cost_funcsFromPtVec(ptVec, nodeConn, DVDict, iComp, solverOptions, evalFuncSensKey;
                DVDictList=DVDictList, CLMain=CLMain
            )
            if !solverOptions["onlyStructDerivs"]

                for ii in eachindex(ptVec)
                    ptVecwork = copy(ptVec)
                    ptVecwork[ii] += dh
                    f_f = cost_funcsFromPtVec(ptVecwork, nodeConn, DVDict, iComp, solverOptions, evalFuncSensKey;
                        DVDictList=DVDictList, CLMain=CLMain
                    )
                    ptVecwork[ii] -= dh
                    if evalFuncSensKey == "cd"
                        # cdi -2 [OK with adjoint], cdw -3 [Not OK], cdpr - 4 [OK], cdj -5, cds -6
                        dfdxPt[ii] = (f_f[1] - f_i[1]) * idh
                    else
                        dfdxPt[ii] = (f_f - f_i) * idh
                    end
                end
            else
                println("Only structural derivatives")
            end


            DVDict["theta_f"] += dh
            f_f = cost_funcsFromPtVec(ptVec, nodeConn, DVDict, iComp, solverOptions, evalFuncSensKey;
                DVDictList=DVDictList, CLMain=CLMain
            )
            DVDict["theta_f"] -= dh

            if evalFuncSensKey == "cd"
                dfdxstruct["theta_f"] = (f_f[1] - f_i[1]) * idh
            else
                dfdxstruct["theta_f"] = (f_f - f_i) * idh
            end

            dfdxstruct["toc"] = zeros(DTYPE, 1, length(DVDict["toc"]))
            for ii in eachindex(DVDict["toc"])
                DVDict["toc"][ii] += dh
                f_f = cost_funcsFromPtVec(ptVec, nodeConn, DVDict, iComp, solverOptions, evalFuncSensKey;
                    DVDictList=DVDictList, CLMain=CLMain
                )
                DVDict["toc"][ii] -= dh
                if evalFuncSensKey == "cd"
                    dfdxstruct["toc"][1, ii] = (f_f[1] - f_i[1]) * idh
                else
                    dfdxstruct["toc"][1, ii] = (f_f - f_i) * idh
                end
            end

            dfdxPt = reshape(dfdxPt, 3, NPT)
            funcsSens = Dict(
                "mesh" => dfdxPt,
                "fiber" => dfdxstruct["theta_f"],
                "toc" => dfdxstruct["toc"],
            )
            # println("Finite difference sensitivities for $(evalFuncSensKey): ", funcsSens)

            println("writing funcsens to file")
            writedlm("funcsSens-mesh-$(evalFuncSensKey)-$(mode).csv", funcsSens["mesh"], ',')
            # writedlm("funcsSens-struct-$(evalFuncSensKey)-$(mode).csv", funcsSens["struct"], ',')

            funcsSensOut[evalFuncSensKey] = funcsSens
        end

    elseif uppercase(mode) == "RAD" #RAD the whole thing # BUSTED with mutating arrays :(
        backend = AD.ZygoteBackend()
        funcsSens, = AD.gradient(backend, (x) -> cost_funcsFromPtVec(
                x, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                DVDictList=DVDictList, CLMain=CLMain),
            ptVec)

    elseif uppercase(mode) == "ADJOINT"

        STATSOL = STATSOLLIST[iComp]
        solverParams = STATSOL.SOLVERPARAMS
        appendageOptions = solverOptions["appendageList"][iComp]
        DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
        u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

        tX = @elapsed begin
            ∂r∂xPt, ∂r∂xParams = compute_∂r∂x(STATSOL.structStates, DVDictList, LECoords, TECoords, nodeConn;
                # mode="FiDi", # about 981 sec
                # mode="RAD", # about 282 sec
                mode="ANALYTIC", # 10 sec
                appendageOptions=appendageOptions, solverOptions=solverOptions, CLMain=CLMain, iComp=iComp)
        end

        tStiff = @elapsed begin
            ∂r∂u = compute_∂r∂u(u, LECoords, TECoords, nodeConn, "Analytic";
                appendageParamsList=DVDictList, solverParams=solverParams, solverOptions=solverOptions, appendageOptions=appendageOptions, iComp=iComp)
        end

        println("∂r∂X:\t$(tX) sec\n∂r∂u:\t$(tStiff) sec")
        for evalFuncSensKey in evalFuncSensList
            tFuncU = @elapsed begin
                ∂f∂u = compute_∂f∂u(evalFuncSensKey, STATSOL, ptVec, nodeConn, DVDict;
                    # mode="FiDi", # 200 sec
                    # mode="RAD", # 100 sec
                    mode="ANALYTIC", # 2 sec
                    appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
                )
            end
            tFuncX = @elapsed begin
                ∂f∂xPt, ∂f∂xParams = compute_∂f∂x(evalFuncSensKey, STATSOL, ptVec, nodeConn, DVDict;
                    # mode="RAD", # 100 sec
                    # mode="FiDi", # 79
                    mode="ANALYTIC", # 15 sec
                    appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
                )
            end
            println("∂f∂u:\t$(tFuncU) sec\n∂f∂x:\t$(tFuncX) sec")

            println("+---------------------------------+")
            println("| Computing adjoint: $(evalFuncSensKey)")
            println("+---------------------------------+")
            # should be (n_u x n_f) a column vector!
            ∂f∂uT = reshape(∂f∂u, 1, length(∂f∂u)) # we only do one cost func at a time
            tadjoint = @elapsed begin
                psiVec = compute_adjointVec(∂r∂u, ∂f∂uT; solverParams=solverParams, appendageOptions=appendageOptions)
            end
            println("Ψ solve:\t$(tadjoint) sec")

            # println("∂f∂xPt", ∂f∂xPt)
            # println("∂r∂xPt",  ∂r∂xPt)
            # println("psiVec", psiVec)

            # --- Compute total sensitivities ---
            # dfdxpt = ∂f∂x - transpose(psiMat) * ∂r∂x
            # Transpose the adjoint vector so it's now a row vector
            dfdxPt = ∂f∂xPt - (transpose(psiVec) * ∂r∂xPt)
            # println("shape: ", size(psiVec), size(∂r∂xStruct))
            # println("shape: ", size(∂f∂xStruct))
            dfdxParams = Dict()
            for designVar in allDesignVariables
                dfdxParams[designVar] = ∂f∂xParams[designVar] - (transpose(psiVec) * ∂r∂xParams[designVar])
            end

            # writedlm("∂f∂ududx.csv", (transpose(psiVec) * ∂r∂xPt), ',')

            funcsSens = Dict(
                "mesh" => reshape(dfdxPt, 3, NPT),
                "params" => dfdxParams,
            )
            # println("Adjoint sensitivities for $(evalFuncSensKey): ", funcsSens)

            # writedlm("funcsSens-mesh-$(evalFuncSensKey)-$(mode).csv", funcsSens["mesh"], ',')
            # writedlm("funcsSens-struct-$(evalFuncSens)-$(mode).csv", funcsSens["struct"], ',')

            funcsSensOut[evalFuncSensKey] = funcsSens
            # push!(funcsSensOut, funcsSens)
        end


    elseif uppercase(mode) == "DIRECT"

        println("Computing direct for component ", iComp)
        STATSOL = STATSOLLIST[iComp]
        solverParams = STATSOL.SOLVERPARAMS
        appendageOptions = solverOptions["appendageList"][iComp]
        DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
        u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

        ∂r∂xPt = compute_∂r∂x(STATSOL.structStates, DVDictList, LECoords, TECoords, nodeConn;
            mode="ANALYTIC",
            appendageOptions=appendageOptions, solverOptions=solverOptions, CLMain=CLMain, iComp=iComp)
        ∂r∂u = compute_∂r∂u(u, LECoords, TECoords, nodeConn, "Analytic";
            appendageParamsList=DVDictList, solverParams=solverParams, solverOptions=solverOptions, appendageOptions=appendageOptions, iComp=iComp)
        println("+---------------------------------+")
        println("| Computing direct: $(evalFuncSens)")
        println("+---------------------------------+")
        phiMat = compute_directMatrix(∂r∂u, ∂r∂xPt;
            solverParams=solverParams)

        for evalFuncSens in evalFuncSensList
            ∂f∂u = compute_∂f∂u(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            ∂f∂xPt = compute_∂f∂x(evalFuncSens, STATSOL, ptVec, nodeConn, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )

            # --- Compute total sensitivities ---
            # funcsSens = ∂f∂x - ∂f∂u * [ϕ]
            dfdxPt = ∂f∂xPt - (∂f∂u[:, 1:end.∉[DOFBlankingList]] * phiMat)
            funcsSens = reshape(dfdxPt, 3, NPT)
            # writedlm("funcsSens-$(evalFuncSens)-$(mode).csv", funcsSens, ',')

            push!(funcsSensOut, funcsSens)
        end

    else
        error("Invalid mode")
    end


    # end

    return funcsSensOut
end

function compute_∂f∂x(
    costFunc::String, SOL::StaticSolution, ptVec, nodeConn, appendageParams::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)

    # println("Computing ∂f∂x in $(mode)...")
    ∂f∂xPt = zeros(DTYPE, length(ptVec))
    ∂f∂xParams = Dict()

    # Force an automatic initial eval so less code duplication
    dh = 1e-5
    f_i = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
    )


    DVDictList[iComp] = appendageParams
    if uppercase(mode) == "ANALYTIC"
        hydromode = "ANALYTIC"
        # println("Computing hydro derivatives in $(hydromode)")

        if costFunc ∉ ["cdi", "cdw", "cdpr", "cdj", "cds"] # parts of the drag build up
            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            u = SOL.structStates[1:end.∉[DOFBlankingList]]
            dfstaticdXpt = compute_dfhydrostaticdXpt(u, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode=hydromode)

            meanChord = mean(SOL.FOIL.chord)
            rootChord = SOL.FOIL.chord[1]
            qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2
            # areaRef = HydroStrip.compute_areas(SOL.FEMESH, SOL.FOIL; appendageOptions=appendageOptions, STRUT=SOL.STRUT)
            LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
            areaRef = Preprocessing.compute_areas(LECoords, TECoords, nodeConn)

            if costFunc != "cd"
                ∂f∂fstatic = compute_∂costFunc∂fhydro(costFunc, SOL, ptVec, nodeConn, qdyn, areaRef, meanChord, rootChord, appendageOptions, appendageParams, solverOptions)
                ∂f∂fstatic = ∂f∂fstatic[1:end.∉[DOFBlankingList]]
            else
                ∂f∂fstatic = zeros(length(u))
            end

            # TODO: get a reverse mode working with this
            ∂f∂xPtdirect = compute_∂costFunc∂Xpt(costFunc, SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

            # Any function can be written as some function of the fluid or structural states
            ∂f∂xPt = reshape(∂f∂fstatic, 1, length(∂f∂fstatic)) * dfstaticdXpt + reshape(∂f∂xPtdirect, 1, length(∂f∂xPtdirect))
        else
            error("Why are you here? $(costFunc)")
        end

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

    # ************************************************
    #     PARAMS DERIVATIVES
    # ************************************************
    appendageParams["theta_f"] += dh
    f_f = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
    )
    appendageParams["theta_f"] -= dh
    if costFunc != "cd"
        ∂f∂xParams["theta_f"] = (f_f - f_i) / dh
    else
        # When drag coeff is the cost func, pick out the first component
        ∂f∂xParams["theta_f"] = (f_f[1] - f_i[1]) / dh
    end
    ∂f∂xParams["toc"] = zeros(DTYPE, 1, length(appendageParams["toc"]))
    for ii in eachindex(appendageParams["toc"])
        appendageParams["toc"][ii] += dh
        f_f = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
            appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
        )
        appendageParams["toc"][ii] -= dh
        if costFunc != "cd"
            ∂f∂xParams["toc"][1, ii] = (f_f - f_i) / dh
        else
            ∂f∂xParams["toc"][1, ii] = (f_f[1] - f_i[1]) / dh
        end
    end
    appendageParams["alfa0"] += dh
    f_f = get_evalFunc(costFunc, SOL.structStates, SOL, ptVec, nodeConn, appendageParams;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
    )
    appendageParams["alfa0"] -= dh
    if costFunc != "cd"
        ∂f∂xParams["alfa0"] = (f_f - f_i) / dh
    else
        # When drag coeff is the cost func, pick out the first component
        ∂f∂xParams["alfa0"] = (f_f[1] - f_i[1]) / dh
    end


    ∂f∂xPt = reshape(∂f∂xPt, 1, length(∂f∂xPt))

    # println("writing ∂f∂xPt-$(mode).csv")
    # writedlm("∂f∂xPt-$(mode).csv", ∂f∂xPt, ',')

    return ∂f∂xPt, ∂f∂xParams
end

function compute_∂costFunc∂Xpt(costFunc, SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    idxTip = SOL.FEMESH.idxTip

    function costFuncFromXpt(costFunc, SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

        toc::Vector{RealOrComplex} = appendageParams["toc"]
        ab::Vector{RealOrComplex} = appendageParams["ab"]
        x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]

        WING, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, appendageParams["theta_f"], appendageParams["toc_strut"], appendageParams["ab_strut"], appendageParams["theta_f_strut"], appendageParams, appendageOptions, solverOptions)

        midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)

        FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, appendageParams["theta_f"], idxTip, zeros(10, 2))

        # areaRef = HydroStrip.compute_areas(FEMESH, WING; appendageOptions=appendageOptions, STRUT=STRUT)
        areaRef = Preprocessing.compute_areas(LECoords, TECoords, nodeConn)
        meanChord = sum(chordLengths) / length(chordLengths)
        rootChord = chordLengths[1]

        qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2

        fout = compute_funcsFromfhydro(costFunc, SOL.structStates, SOL.fHydro, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, FEMESH)

        return fout
    end

    function dragCostFuncFromXpt(SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

        toc::Vector{RealOrComplex} = appendageParams["toc"]
        ab::Vector{RealOrComplex} = appendageParams["ab"]
        x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]

        WING, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, appendageParams["theta_f"], appendageParams["toc_strut"], appendageParams["ab_strut"], appendageParams["theta_f_strut"], appendageParams, appendageOptions, solverOptions)

        midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)

        FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, appendageParams["theta_f"], idxTip, zeros(10, 2))

        # areaRef = HydroStrip.compute_areas(FEMESH, WING; appendageOptions=appendageOptions, STRUT=STRUT)
        areaRef = Preprocessing.compute_areas(LECoords, TECoords, nodeConn)
        meanChord = sum(chordLengths) / length(chordLengths)
        rootChord = chordLengths[1]

        qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2
        _, cl = ComputeFunctions.compute_lift(SOL.fHydro, qdyn, areaRef)

        aeroSpan = Preprocessing.compute_aeroSpan(midchords, FEMESH.idxTip)
        cdw, cdpr, cdj, cds, dw, dpr, dj, ds = ComputeFunctions.compute_calmwaterdragbuildup(appendageParams, appendageOptions, solverOptions, qdyn, areaRef, aeroSpan, cl, meanChord, rootChord, chordLengths)

        return vec([cdw, cdpr, cdj, cds, dw, dpr, dj, ds])
    end

    # backend = AD.FiniteDifferencesBackend() # this works alright
    # ∂f∂xPt, = AD.gradient(backend, x -> costFuncFromXpt(
    #         costFunc, SOL, x, nodeConn, appendageParams, appendageOptions, solverOptions),
    #     ptVec)
    ∂f∂xPt = zeros(DTYPE, length(ptVec))
    if costFunc != "cd"
        f_i = costFuncFromXpt(costFunc, SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        dh = 1e-5
        for ii in eachindex(ptVec)
            ptVec[ii] += dh
            f_f = costFuncFromXpt(costFunc, SOL, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
            ptVec[ii] -= dh

            ∂f∂xPt[ii] = (f_f - f_i) / dh
        end
    elseif costFunc == "cd"

        #dcdidXpt = HydroStrip.compute_dcdidXpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")
        dcdidXpt = HydroStrip.compute_dcdidXpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="ADJOINT") # DIRECT and ADJOINT agree and are good
        #dcdidXpt = HydroStrip.compute_dcdidXpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="DIRECT")
        # backend = AD.ForwardDiffBackend()
        backend = AD.ZygoteBackend()

        ddragbuildupdxpt, = AD.jacobian(backend, x -> dragCostFuncFromXpt(
                SOL, x, nodeConn, appendageParams, appendageOptions, solverOptions),
            ptVec)
        # println(" cdw grad\n", reshape(ddragbuildupdxpt[1, :], 3, length(ptVec) ÷ 3))
        # println(" cdpr grad\n", reshape(ddragbuildupdxpt[2, :], 3, length(ptVec) ÷ 3))
        # println(" cdj grad\n", reshape(ddragbuildupdxpt[3, :], 3, length(ptVec) ÷ 3))
        # println(" cds grad\n", reshape(ddragbuildupdxpt[4, :], 3, length(ptVec) ÷ 3))

        dcddxpt = ddragbuildupdxpt[1, :] + ddragbuildupdxpt[2, :] + ddragbuildupdxpt[3, :] + ddragbuildupdxpt[4, :]

        ∂f∂xPt = dcdidXpt + reshape(dcddxpt, 1, length(dcddxpt))
    else
        error("Why are you here?")
    end


    return ∂f∂xPt
end

function compute_∂costFunc∂fhydro(costFunc, SOL, ptVec, nodeConn, qdyn, areaRef, meanChord, rootChord, appendageOptions, appendageParams, solverOptions)
    """
    Compute the gradient of the cost functions with respect to the hydrodynamic forces
    """

    # backend = AD.ReverseDiffBackend()
    backend = AD.ForwardDiffBackend() # works
    # backend = AD.FiniteDifferencesBackend() #SUPER SLOW
    # backend = AD.ZygoteBackend()

    if costFunc != "cd"
        ∂f∂fhydro, = AD.gradient(backend, x -> compute_funcsFromfhydro(
                costFunc, SOL.structStates, x, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, SOL.FEMESH),
            SOL.fHydro)
    else
        ∂f∂fhydro, = AD.gradient(backend, x -> compute_funcsFromfhydro(
                costFunc, SOL.structStates, x, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, SOL.FEMESH)[1],
            SOL.fHydro)
    end

    return ∂f∂fhydro
end

function compute_∂costFunc∂udirect(costFunc, SOL, ptVec, nodeConn, qdyn, areaRef, meanChord, rootChord, appendageOptions, appendageParams, solverOptions)
    """
    Compute the gradient of the cost functions with respect to the hydrodynamic forces
    """

    backend = AD.ForwardDiffBackend() # works
    # backend = AD.ZygoteBackend()

    if costFunc != "cd"
        ∂f∂udirect, = AD.gradient(backend, x -> compute_funcsFromfhydro(
                costFunc, x, SOL.fHydro, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, SOL.FEMESH),
            SOL.structStates)
    else
        ∂f∂udirect, = AD.gradient(backend, x -> compute_funcsFromfhydro(
                costFunc, x, SOL.fHydro, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, qdyn, areaRef, meanChord, rootChord, SOL.FEMESH)[1],
            SOL.structStates)
    end

    return ∂f∂udirect
end


function compute_∂f∂u(
    costFunc::String, SOL::StaticSolution, ptVec, nodeConn, appendageParams::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)
    """
    Compute the gradient of the cost functions with respect to the structural states
    SOL is the solution struct at the current design point
    """

    # println("Computing ∂f∂u...")
    ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
    DVDictList[iComp] = appendageParams

    if uppercase(mode) == "ANALYTIC" # works, but the root portion may be slightly off compared to finite

        # if costFunc ∉ ["cd", "cdi", "cdw", "cdpr", "cdj", "cds"]
        _, _, WING, areaRef = precompute_funcsux(SOL.structStates, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        meanChord = mean(WING.chord)
        rootChord = WING.chord[1]
        qdyn = 0.5 * solverOptions["rhof"] * solverOptions["Uinf"]^2

        ∂f∂fstatic = compute_∂costFunc∂fhydro(costFunc, SOL, ptVec, nodeConn, qdyn, areaRef, meanChord, rootChord, appendageOptions, appendageParams, solverOptions)
        dfstaticdu = -SOL.SOLVERPARAMS.AICmat # RHS type

        # ∂f∂udirect = costfuncsdpending directly on the structural states (tip bend, etc.)
        ∂f∂udirect = compute_∂costFunc∂udirect(costFunc, SOL, ptVec, nodeConn, qdyn, areaRef, meanChord, rootChord, appendageOptions, appendageParams, solverOptions)

        # println("∂f∂u:\n", reshape(∂f∂fstatic, 1, length(∂f∂fstatic)) * dfstaticdu)
        ∂f∂u = reshape(∂f∂fstatic, 1, length(∂f∂fstatic)) * dfstaticdu + reshape(∂f∂udirect, 1, length(∂f∂udirect))

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

    # println("Computing ∂r∂x in $(mode) mode...")
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    u = allStructStates[1:end.∉[DOFBlankingList]]

    ptVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)
    appendageParams = appendageParamsList[iComp]

    ∂r∂xPt = zeros(DTYPE, length(u), length(ptVec))
    ∂r∂xParams = Dict()

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
        ∂r∂xParams = (f_f - f_i) / 1e-4

    elseif uppercase(mode) == "CS" # works but slow (4 sec)
        dh = 1e-100
        println("step size: ", dh)

        ∂r∂xPt = zeros(DTYPE, length(u), nn)

        ∂r∂xPt = perturb_coordsForResid(∂r∂xPt, 1im * dh, u, LECoords, TECoords, nodeConn, appendageParamsList, appendageOptions, solverOptions, iComp)

    elseif uppercase(mode) == "ANALYTIC"

        # println("struct partials")
        # This seems good
        # tstruct = @elapsed begin
        ∂Kss∂x_u = compute_∂KssU∂x(u, ptVec, nodeConn, appendageOptions, appendageParamsList[1], solverOptions;
            mode="CS" # 4 sec but accurate
            # mode="FAD" # 5 sec
            # mode="FiDi" # 2.8 sec
            # mode="RAD" # broken
        )
        # end
        # println("∂Kss∂x:\t$(tstruct) sec")
        # writedlm("dKssdx.csv", ∂Kss∂x_u, ',')


        # It is strange that this is zero
        # hydromode = "FiDi" # This appears to be right
        hydromode = "ANALYTIC"
        # println("Computing hydro derivatives in $(hydromode)")
        dKffdXpt_u = compute_dfhydrostaticdXpt(u, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode=hydromode)

        # ∂r∂x = u ∂K∂X - dfdX
        ∂r∂xPt = (∂Kss∂x_u - dKffdXpt_u)

        # ************************************************
        #     Params derivatives
        # ************************************************
        dh = 1e-4
        f_i = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] += dh
        f_f = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["theta_f"] -= dh
        ∂r∂xParams["theta_f"] = (f_f - f_i) / dh

        ∂r∂xParams["toc"] = zeros(DTYPE, length(f_i), length(appendageParamsList[iComp]["toc"]))
        for ii in eachindex(appendageParamsList[iComp]["toc"])
            appendageParamsList[iComp]["toc"][ii] += dh
            f_f = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
            appendageParamsList[iComp]["toc"][ii] -= dh
            ∂r∂xParams["toc"][:, ii] = (f_f - f_i) / dh
        end
        appendageParamsList[iComp]["alfa0"] += dh
        f_f = SolveStatic.compute_residualsFromCoords(u, ptVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
        appendageParamsList[iComp]["alfa0"] -= dh
        ∂r∂xParams["alfa0"] = (f_f - f_i) / dh


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

    # writedlm("drdxPt-$(mode).csv", ∂r∂xPt, ',')
    return ∂r∂xPt, ∂r∂xParams
end

function compute_∂KssU∂x(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="CS")
    """
    Derivative of structural stiffness matrix with respect to design variables times structural states
    """

    ∂KssU∂x = zeros(DTYPE, length(structStates), length(ptVec))
    LECoords, _ = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = Preprocessing.get_tipnode(LECoords)

    if uppercase(mode) == "CS"
        # CS is faster than fidi automated, but RAD will probably be best later...? 2.4sec
        dh = 1e-100
        ptVecWork = complex(ptVec)
        for ii in eachindex(ptVec)
            ptVecWork[ii] += 1im * dh
            f_f = compute_KssU(structStates, ptVecWork, nodeConn, idxTip, appendageOptions, appendageParams, solverOptions)
            ptVecWork[ii] -= 1im * dh
            ∂KssU∂x[:, ii] = imag(f_f) / dh
        end
    elseif uppercase(mode) == "FIDI"
        dh = 1e-5
        f_i = compute_KssU(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
        for ii in eachindex(ptVec)
            ptVec[ii] += dh
            f_f = compute_KssU(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
            ptVec[ii] -= dh
            ∂KssU∂x[:, ii] = (f_f - f_i) / dh
        end
    elseif uppercase(mode) == "FAD"
        backend = AD.ForwardDiffBackend()
        ∂KssU∂x, = AD.jacobian(
            backend,
            x -> compute_KssU(structStates, x, nodeConn, appendageOptions, appendageParams, solverOptions),
            ptVec,
        )
    elseif uppercase(mode) == "RAD"
        backend = AD.ReverseDiffBackend()
        ∂KssU∂x, = AD.jacobian(
            backend,
            x -> compute_KssU(structStates, x, nodeConn, appendageOptions, appendageParams, solverOptions),
            ptVec,
        )
    end

    return ∂KssU∂x
end

function compute_KssU(u, xVec, nodeConn, idxTip, appendageOptions, appendageParams, solverOptions)
    LECoords, TECoords = Utilities.repack_coords(xVec, 3, length(xVec) ÷ 3)

    midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
        print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

        α₀ = appendageParams["alfa0"]
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
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
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
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

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), @ignore_derivatives(idxTip), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))

    WING, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    globalK, _, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"])

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    Kmat = globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    f = Kmat * u
    return f
end

function compute_dfhydrostaticdXpt(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="ANALYTIC")
    """
    Derivative of steady hydro forces wrt mesh points
    """

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
        # println("Computing dcladXpt in $(hydromode) mode...")
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
        idxTip = Preprocessing.get_tipnode(LECoords)
        midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)
        if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
            print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

            α₀ = appendageParams["alfa0"]
            rake = appendageParams["rake"]
            # span = appendageParams["s"] * 2
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
            rake = appendageParams["rake"]
            # span = appendageParams["s"] * 2
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
        AEROMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))
        FOIL, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

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

        DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
        nblank = length(DOFBlankingList)

        dKffdX = zeros(DTYPE, length(structStates) + nblank, length(structStates) + nblank, length(ptVec))
        dcladXpt = zeros(DTYPE, 40, length(ptVec))
        dKffdXpt = zeros(DTYPE, (length(structStates) + nblank)^2, length(ptVec))
        for ii in eachindex(ptVec)

            ptVec[ii] += dh

            Kff_f, cla_f = compute_Kff(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

            ptVec[ii] -= dh

            # ∂fstatic∂X[:, ii] = (f_f - f_i) / dh
            dcladXpt[:, ii] = (cla_f - cla_i) / dh


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

    midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
        print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

        α₀ = appendageParams["alfa0"]
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
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
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
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

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), @ignore_derivatives(idxTip), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))

    LLOutputs, LLSystem, FlowCond = InitModel.init_staticHydro(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    statWingStructModel, statStrutStructModel = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    sweepAng = LLSystem.sweepAng

    WING = DesignConstants.DynamicFoil(
        statWingStructModel.mₛ, statWingStructModel.Iₛ, statWingStructModel.EIₛ, statWingStructModel.EIIPₛ, statWingStructModel.GJₛ, statWingStructModel.Kₛ, statWingStructModel.Sₛ, statWingStructModel.EAₛ,
        statWingStructModel.eb, statWingStructModel.ab, statWingStructModel.chord, statWingStructModel.nNodes, statWingStructModel.constitutive,
        [0, 1.0], [0, 1.0]
    )

    STRUT = nothing

    dim = NDOF * (size(elemConn)[1] + 1)

    _, _, _, AIC, _ = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, dim, sweepAng, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, ELEMTYPE; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(allStructuralStates, appendageParams, ELEMTYPE; appendageOptions=appendageOptions,)

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    fFull = -AIC * foilTotalStates
    f = fFull[1:end.∉[DOFBlankingList]]

    return f, -AIC
end

function compute_Kff(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    toc::Vector{RealOrComplex} = appendageParams["toc"]
    ab::Vector{RealOrComplex} = appendageParams["ab"]
    x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    toc_strut = appendageParams["toc_strut"]
    ab_strut = appendageParams["ab_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]

    structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn), @ignore_derivatives(idxTip), appendageParams, appendageOptions)
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))

    LLOutputs, LLSystem, FlowCond = InitModel.init_staticHydro(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    statWingStructModel, _ = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

    sweepAng = LLSystem.sweepAng

    WING = DesignConstants.DynamicFoil(
        statWingStructModel.mₛ, statWingStructModel.Iₛ, statWingStructModel.EIₛ, statWingStructModel.EIIPₛ, statWingStructModel.GJₛ, statWingStructModel.Kₛ, statWingStructModel.Sₛ, statWingStructModel.EAₛ,
        statWingStructModel.eb, statWingStructModel.ab, statWingStructModel.chord, statWingStructModel.nNodes, statWingStructModel.constitutive,
        [0, 1.0], [0, 1.0]
    )

    STRUT = nothing

    dim = NDOF * (size(elemConn)[1] + 1)

    _, _, _, AIC, _ = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, dim, sweepAng, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)


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
        #     ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)

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