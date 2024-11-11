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
using ..DesignConstants: SORTEDDVS, DynamicFoil, CONFIGS
using ..SolverRoutines: SolverRoutines
using ..Utilities: Utilities
using ..DCFoilSolution: DCFoilSolution, StaticSolution
using ..TecplotIO: TecplotIO

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
    qSol, _ = SolverRoutines.converge_resNonlinear(compute_residualsFromCoords, compute_∂r∂uFromCoords, q_ss0, x0List;
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
    """

    # THIS IS WHERE most cost comes from
    WING, STRUT, _, FEMESH, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if !(appendageOptions["config"] == "t-foil")
        STRUT = WING # just to make the code work
    elseif !(appendageOptions["config"] in CONFIGS)
        error("Invalid configuration")
    end

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"], verbose=verbose)
    alphaCorrection::DTYPE = 0.0
    # if iComp > 1
    #     alphaCorrection = HydroStrip.correct_downwash(iComp, CLMain, DVDictList, solverOptions)
    # end
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(FEMESH, WING, size(globalM)[1], appendageParams["sweep"], WING.U∞, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, LLSystem=LLSystem, use_nlll=solverOptions["use_nlll"])

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
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
    evalFunc::String, states::Vector, SOL::StaticSolution, LECoords, TECoords, nodeConn, DVDict;
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)
    """
    Works for a single cost function
    """

    costFuncs = compute_funcs(evalFunc, states, SOL, LECoords, TECoords, nodeConn, DVDict;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

    return costFuncs[evalFunc]
end

function compute_funcs(
    evalFunc, states::Vector, SOL::StaticSolution,
    LECoords, TECoords, nodeConn, DVDict;
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)

    # DVDict = Utilities.repack_dvdict(DVVec, DVLengths)
    # ************************************************
    #     RECOMPUTE FROM u AND x
    # ************************************************
    # --- First unpack inputs ---
    WING = SOL.FOIL
    STRUT = SOL.STRUT
    # nnodes = WING.nNodes
    # solconstants = SOL.SOLVERPARAMS

    # --- Now update the inputs from DVDict ---
    α₀ = DVDict["alfa0"]
    rake = DVDict["rake"]

    # WING, STRUT, constants, FEMESH = setup_problemFromDVDict(DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)
    WING, STRUT, constants, FEMESH = setup_problemFromCoords(LECoords, nodeConn, TECoords, DVDict, appendageOptions, solverOptions)

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
    qdyn = 0.5 * WING.ρ_f * WING.U∞^2

    theta = states[ΘIND:NDOF:end]
    Moments = forces[ΘIND:NDOF:end]
    W = states[WIND:NDOF:end]
    Lift = forces[WIND:NDOF:end]

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************
    costFuncs = Dict() # initialize empty costFunc dictionary
    if appendageOptions["config"] == "wing"
        ADIM = constants.planformArea
    elseif appendageOptions["config"] == "t-foil" || appendageOptions["config"] == "full-wing"
        ADIM = 2 * constants.planformArea
    end

    if "wtip" in [evalFunc]
        w_tip = W[end]
        costFuncs["wtip"] = w_tip
    end
    if "psitip" in [evalFunc]
        psi_tip = theta[end]
        costFuncs["psitip"] = psi_tip
    end
    TotalLift = sum(Lift)
    if "lift" in [evalFunc]
        costFuncs["lift"] = TotalLift
    end
    # Moment about mid-chord (where the finite element is)
    TotalMoment = sum(Moments)
    if "moment" in [evalFunc]
        costFuncs["moment"] = TotalMoment
    end
    if "cl" in [evalFunc]
        CL = TotalLift / (qdyn * ADIM)
        costFuncs["cl"] = CL
    end
    # Coefficient of moment about the mid-chord
    if "cmy" in [evalFunc]
        CM = TotalMoment / (qdyn * ADIM * meanChord)
        costFuncs["cmy"] = CM
    end
    if "cd" in [evalFunc]
        CD = 0.0
        costFuncs["cd"] = CD
    end
    if "cdi" in [evalFunc] || "fxi" in [evalFunc]
        wingTwist = theta[1:appendageOptions["nNodes"]]

        _, F, CDi = HydroStrip.compute_hydroLLProperties(
            DVDict["s"],
            WING.chord,
            deg2rad(DVDict["alfa0"]),
            0.0, # rake
            0.0,# sweep
            0.0,# depth
            ;
            solverOptions=solverOptions
        )
        Fxi = F[XDIM]
        # _, Fxi, CDi = HydroStrip.compute_glauert_circ(
        #     DVDict["s"],
        #     WING.chord,
        #     deg2rad(DVDict["alfa0"]),
        #     solverOptions["Uinf"];
        #     h=DVDict["depth0"],
        #     useFS=solverOptions["use_freeSurface"],
        #     rho=solverOptions["rhof"],
        #     twist=wingTwist,
        #     debug=solverOptions["debug"],
        #     config=appendageOptions["config"],
        # )

        if "cdi" in [evalFunc]
            costFuncs["cdi"] = CDi
        end
        if "fxi" in [evalFunc]
            Fxi = CDi * qdyn * ADIM
            costFuncs["fxi"] = Fxi
        end
    end
    # From Hörner Chapter 8
    if "cdj" in [evalFunc] || "fxj" in [evalFunc]
        tocbar = 0.5 * (DVDict["toc"][1] + DVDict["toc_strut"][1])
        CDt = 17 * (tocbar)^2 - 0.05
        dj = CDt * (qdyn * (tocbar * DVDict["c"][1])^2)
        CDj = dj / (qdyn * ADIM)
        costFuncs["cdj"] = CDj
        costFuncs["fxj"] = dj
    end
    if "cds" in [evalFunc] || "fxs" in [evalFunc]
        t = DVDict["toc_strut"][end] * DVDict["c_strut"][end]
        # --- Hörner CHapter 10 ---
        # CDts = 0.24
        # ds = CDts * (qdyn * (t)^2)
        # CDs = ds / (qdyn * ADIM)
        # Chapman 1971 assuming x/c = 0.35
        CDs = 0.009 + 0.013 * DVDict["toc_strut"][end]
        ds = CDs * qdyn * t * DVDict["c_strut"][end]
        costFuncs["cds"] = CDs
        costFuncs["fxs"] = ds
    end
    # # TODO:
    # Wave drag
    # if "cdw" in [evalFunc]s || "fxw" in [evalFunc]s
    #     # rws = EQ 6.149 FALTINSEN thickness effect on wave resistance
    #     # rwgamma = EQ 6.145 FALTINSEN wave resistance due to lift
    # end
    if "cdpr" in [evalFunc] || "fxpr" in [evalFunc] # profile drag
        if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
            WSA = 2 * ADIM # both sides
        elseif appendageOptions["config"] == "t-foil"
            WSA = 2 * ADIM + 2 * DVDict["s_strut"] * mean(DVDict["c_strut"])
        end
        println("I'm not debugged")
        # TODO: MAKE WSA AND DRAG A VECTORIZED STRIPWISE CALCULATION
        NU = 1.1892E-06 # kinematic viscosity of seawater at 15C
        Re = solverOptions["Uinf"] * meanChord / NU
        # Ma = solverOptions["Uinf"] / 1500
        cfittc = 0.075 / (log10(Re) - 2)^2 # flat plate friction coefficient ITTC 1957
        xcmax = 0.3 # chordwise position of the maximum thickness
        # # --- Raymer equation 12.30 ---
        # FF = (1 .+ 0.6 ./ (xcmax) .* DVDict["toc"] + 100 .* DVDict["toc"].^4) * (1.34*Ma^0.18 * cos(DVDict["sweep"])^0.28)
        # --- Torenbeek 1990 ---
        # First term is increase in skin friction due to thickness and quartic is separation drag
        FF = 1 .+ 2.7 .* DVDict["toc"] .+ 100 .* DVDict["toc"] .^ 4
        FF = mean(FF)
        Df = qdyn * WSA * cfittc
        Dpr = Df * FF
        costFuncs["fxpr"] = Dpr
        costFuncs["cdpr"] = Dpr / (qdyn * ADIM)
    end
    # --- Center of forces ---
    # These calculations are in local appendage frame
    if "cofz" in [evalFunc] # center of forces in z direction
        xcenter = sum(Lift .* FEMESH.mesh[:, XDIM]) / sum(Lift)
        ycenter = sum(Lift .* FEMESH.mesh[:, YDIM]) / sum(Lift)
        zcenter = sum(Lift .* FEMESH.mesh[:, ZDIM]) / sum(Lift)
        costFuncs["cofz"] = [xcenter, ycenter, zcenter]
    end

    if "comy" in [evalFunc] # center of moments about y axis
        xcenter = sum(Moments .* FEMESH.mesh[:, XDIM]) / sum(Moments)
        ycenter = sum(Moments .* FEMESH.mesh[:, YDIM]) / sum(Moments)
        zcenter = sum(Moments .* FEMESH.mesh[:, ZDIM]) / sum(Moments)
        costFuncs["comy"] = [xcenter, ycenter, zcenter]
    end

    return costFuncs
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
        evalFunc, SOL.structStates, SOL, LECoords, TECoords, nodeConn, appendageParams;
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


        if uppercase(mode) == "FIDI" # use finite differences the stupid way
            dh = 1e-4
            idh = 1 / dh
            println("step size: ", dh)

            # dfdx = zeros(DTYPE, sum(DVLengths))
            dfdxpt = zeros(DTYPE, length(ptVec))
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
                dfdxpt[ii] = (f_f - f_i) * idh
            end


            funcsSens = reshape(dfdxpt, 3, NPT)
            println("Finite difference sensitivities for $(evalFuncSens): ", funcsSens)

            funcsSensList = push!(funcsSensList, funcsSens)

        elseif uppercase(mode) == "RAD" #RAD the whole thing # BUSTED with mutating arrays :(
            backend = AD.ZygoteBackend()
            @time funcsSens, = AD.gradient(backend, (x) -> cost_funcsFromPtVec(
                    x, nodeConn, DVDict, iComp, solverOptions, evalFuncSens;
                    DVDictList=DVDictList, CLMain=CLMain),
                ptVec)

        elseif uppercase(mode) == "ADJOINT"
            println("TOC adjoint derivative is wrong")

            println("Computing adjoint for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

            @time ∂r∂x = compute_∂r∂x(STATSOL.structStates, DVDict, LECoords, TECoords, nodeConn;
                # mode="FiDi",
                # mode="RAD",
                mode="ANALYTIC",
                # mode="CS",
                appendageOptions=appendageOptions, solverOptions=solverOptions, CLMain=CLMain, iComp=iComp)

            println("Computing ∂r∂u...")
            @time ∂r∂u = compute_∂r∂uFromCoords(u, LECoords, TECoords, nodeConn, "Analytic";
                appendageParamsList=DVDictList, solverParams=solverParams, solverOptions=solverOptions, appendageOptions=appendageOptions, iComp=iComp)

            dfdxpt = zeros(DTYPE, length(ptVec))

            @time ∂f∂u = compute_∂f∂u(costFunc, STATSOL, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            @time ∂f∂x = compute_∂f∂x(costFunc, STATSOL, DVDict;
                mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )
            println("+---------------------------------+")
            println(@sprintf("| Computing adjoint: %s ", costFunc))
            println("+---------------------------------+")
            ∂f∂uT = transpose(∂f∂u)
            # println(size(∂f∂uT)) # should be (n_u x n_f) a column vector!
            psiVec = compute_adjointVec(∂r∂u, ∂f∂uT;
                solverParams=solverParams, appendageOptions=appendageOptions)

            # --- Compute total sensitivities ---
            # dfdxpt = ∂f∂x - transpose(psiMat) * ∂r∂x
            # Transpose the adjoint vector so it's now a row vector
            dfdxpt = ∂f∂x - (transpose(psiVec) * ∂r∂x)

        elseif uppercase(mode) == "DIRECT"

            println("Computing direct for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
            u = STATSOL.structStates[1:end.∉[DOFBlankingList]]

            @time ∂r∂x = compute_∂r∂x(
                STATSOL.structStates, DVDict, LECoords, TECoords, nodeConn;
                mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )

            @time ∂r∂u = compute_∂r∂uFromDV(
                u,
                solverOptions["res_jacobian"];
                solverParams=solverParams,
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList,
                CLMain=CLMain,
                iComp=iComp
            )
            println("+---------------------------------+")
            println(@sprintf("| Computing direct: "))
            println("+---------------------------------+")
            phiMat = compute_directMatrix(∂r∂u, ∂r∂x;
                solverParams=solverParams)

            for (ifunc, costFunc) in enumerate(evalFuncSens)
                ∂f∂u = compute_∂f∂u(costFunc, STATSOL, DVDict;
                    mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
                )
                ∂f∂x = compute_∂f∂x(costFunc, STATSOL, DVDict;
                    mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

                # --- Compute total sensitivities ---
                # funcsSens[costFunc] = ∂f∂x - ∂f∂u * [ϕ]
                dfdx = ∂f∂x - (∂f∂u[:, 1:end.∉[DOFBlankingList]] * phiMat)
                giDV = 1
                for (iiDV, dvkey) in enumerate(SORTEDDVS)
                    ndv = DVLengths[iiDV]

                    # --- Pack sensitivities into dictionary ---
                    funcsSens = Utilities.pack_funcsSens(funcsSens, costFunc, dvkey, dfdx[giDV:giDV+ndv-1])

                    giDV += ndv # starting value for next set
                end
            end
        else
            error("Invalid mode")
        end

        push!(funcsSensList, funcsSens)

    end

    # save(@sprintf("funcsSensList-%s.jld2", mode), "derivs", funcsSensList)

    # # ************************************************
    # #     Sort the sensitivities by key (alphabetical)
    # # ************************************************
    # sorted_keys = sort(collect(keys(funcsSens)))
    # sorted_dict = Dict()
    # for key in sorted_keys
    #     sorted_dict[key] = funcsSens[key]
    # end
    # funcsSens = sorted_dict

    return funcsSensList
end

function compute_∂f∂x(
    costFunc::String, SOL::StaticSolution, DVDict::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)

    println("Computing ∂f∂x...")
    # ∂f∂u = zeros(Float64, length(DVDict))
    DVDictList[iComp] = DVDict
    if uppercase(mode) == "ANALYTIC"
        println("Haven't done this yet")
    elseif uppercase(mode) == "RAD" # WORKS
        backend = AD.ZygoteBackend()
        DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
        ∂f∂x, = AD.gradient(
            backend,
            x -> get_evalFunc(costFunc, SOL.structStates, SOL, x, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain),
            DVVec, # compute deriv at this DV
        )
        ∂f∂x = reshape(∂f∂x, 1, length(∂f∂x))
    elseif uppercase(mode) == "FIDI"
        dh = 1e-3
        println("step size: ", dh)
        DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
        ∂f∂x = zeros(DTYPE, length(DVVec))
        for ii in eachindex(DVVec)
            f_i = get_evalFunc(costFunc, SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            DVVec[ii] += dh
            f_f = get_evalFunc(costFunc, SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            DVVec[ii] -= dh

            ∂f∂x[ii] = (f_f - f_i) / dh
        end
        ∂f∂x = reshape(∂f∂x, 1, length(∂f∂x))
    end
    return ∂f∂x
end

function compute_∂f∂u(
    costFunc::String, SOL::StaticSolution, DVDict::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict(), DVDictList=[], CLMain=0.0, iComp=1
)
    """
    Compute the gradient of the cost functions with respect to the structural states
    SOL is the solution struct at the current design point
    """

    println("Computing ∂f∂u...")
    ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
    DVDictList[iComp] = DVDict
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    # Do analytic
    if uppercase(mode) == "ANALYTIC"

    elseif uppercase(mode) == "RAD" # works
        backend = AD.ZygoteBackend()
        ∂f∂u, = AD.gradient(
            backend,
            u -> get_evalFunc(
                costFunc, u, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            ),
            SOL.structStates,
        )
        ∂f∂u = reshape(∂f∂u, 1, length(∂f∂u))
    elseif uppercase(mode) == "FIDI" # Finite difference
        dh = 1e-4
        idh = 1 / dh
        println("step size:", dh)
        ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
        for ii in eachindex(SOL.structStates)
            r_i = SolveStatic.get_evalFunc(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] += dh
            r_f = SolveStatic.get_evalFunc(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] -= dh

            ∂f∂u[1, ii] = (r_f[costFunc] - r_i[costFunc]) * idh
        end
        # # First derivative using 3 stencil points
        # ∂f∂u, = FiniteDifferences.jacobian(forward_fdm(2, 1),
        #     u -> evalFuncs([costFunc], u, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions),
        #     SOL.structStates,
        # )
        # ∂f∂u = reshape(∂f∂u, 1, length(∂f∂u))
    else
        error("Invalid mode")
    end

    return ∂f∂u
end

function compute_∂r∂x(
    allStructStates, appendageParams, LECoords, TECoords, nodeConn;
    mode="FiDi", appendageOptions=nothing,
    solverOptions=nothing, iComp=1, CLMain=0.0
)
    """
    Partial derivatives of residuals with respect to design variables w/o reconverging the solution
    """

    println("Computing ∂r∂x in $(mode) mode...")
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)
    u = allStructStates[1:end.∉[DOFBlankingList]]

    DVVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)

    if uppercase(mode) == "FIDI" # Finite difference

        # This is the manual FD
        # dh = 1e-4
        # ∂r∂x = zeros(DTYPE, length(u), nn)
        # println("step size: ", dh)
        backend = AD.FiniteDifferencesBackend()
        ∂r∂x = AD.jacobian(
            backend,
            x -> SolveStatic.compute_residualsFromCoords(
                u,
                x,
                nodeConn,
                appendageParams,
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
            ),
            DVVec, # compute deriv at this DV
        )

    elseif uppercase(mode) == "CS" # works but slow (4 sec)
        dh = 1e-100
        println("step size: ", dh)

        ∂r∂x = zeros(DTYPE, length(u), nn)

        ∂r∂x = perturb_coordsForResid(∂r∂x, 1im * dh, u, LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions, iComp)

    elseif uppercase(mode) == "ANALYTIC"

        @time ∂Kss∂x_u = compute_∂KssU∂x(u, DVVec, nodeConn, appendageOptions, appendageParams, solverOptions)

        @time ∂Kff∂x_u = compute_∂KffU∂x(u, DVVec, nodeConn, appendageOptions, appendageParams, solverOptions)

        ∂r∂x = (∂Kss∂x_u - ∂Kff∂x_u)
    elseif uppercase(mode) == "FAD" # this is fked
        error("Not implemented")
    elseif uppercase(mode) == "RAD" # WORKS
        backend = AD.ZygoteBackend()
        ∂r∂x, = AD.jacobian(
            backend,
            x -> SolveStatic.compute_residualsFromCoords(
                u,
                x,
                nodeConn,
                appendageParams;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
            ),
            DVVec, # compute deriv at this DV
        )
    else
        error("Invalid mode")
    end
    return ∂r∂x
end

function compute_∂KssU∂x(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
    """
    Derivative of structural stiffness matrix with respect to design variables times structural states
    """

    ∂KssU∂x = zeros(DTYPE, length(structStates), length(ptVec))

    # backend = AD.ZygoteBackend() # bugging
    # backend = AD.ReverseDiffBackend()

    # backend = AD.FiniteDifferencesBackend()
    # ∂Kss∂x_u, = AD.jacobian(
    #     backend, x -> compute_KssU(structStates, x, nodeConn, appendageOptions, appendageParams, solverOptions),
    #     ptVec
    # )

    # CS is faster than fidi, but RAD will probably be best later.
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

    WING, STRUT = InitModel.init_staticStruct(chordLengths, toc, ab, zeta, theta_f, c_strut, toc_strut, ab_strut, theta_f_strut, appendageOptions, solverOptions)

    globalK, _, _ = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"])

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    Kmat = globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    f = Kmat * u
    return f
end

function compute_∂KffU∂x(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

    ∂KffU∂x = zeros(DTYPE, length(structStates), length(ptVec))

    # backend = AD.ZygoteBackend()
    # ∂KffU∂x, = AD.gradient(
    #     backend, x -> compute_KffU(structStates, x, nodeConn, appendageOptions, appendageParams, solverOptions),
    #     ptVec
    # )

    # dh = 1e-4
    # idh = 1 / dh

    # f_i = compute_KffU(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
    # for ii in eachindex(ptVec)
    #     ptVec[ii] += dh
    #     f_f = compute_KffU(structStates, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)
    #     ptVec[ii] -= dh
    #     ∂KffU∂x[:, ii] = (f_f - f_i) * idh
    # end

    return ∂KffU∂x

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

    WING, STRUT, LLSystem, FlowCond = InitModel.init_staticHydro(α₀, sweepAng, rake, span, chordLengths, ab, zeta, beta, s_strut, c_strut, toc_strut, ab_strut, theta_f_strut, depth0, appendageOptions, solverOptions)

    dim = NDOF * (size(elemConn)[1] + 1)
    _, _, _, AIC, _, _ = HydroStrip.compute_AICs(FEMESH, WING, dim, appendageParams["sweep"], FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, LLSystem=LLSystem, use_nlll=solverOptions["use_nlll"])

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, ELEMTYPE; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(allStructuralStates, appendageParams, ELEMTYPE; appendageOptions=appendageOptions,)

    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    fFull = -AIC * foilTotalStates
    f = fFull[1:end.∉[DOFBlankingList]]

    return f
end
# function perturb_coordsForResid(∂r∂x, step, u, LECoords, TECoords, nodeConn, DVDictList, appendageOptions, solverOptions, iComp)
#     """
#     step can be real or imag
#     """
#     LECoordsVec, mm, nn = Utilities.unpack_coords(LECoords)
#     TECoordsVec, _, _ = Utilities.unpack_coords(TECoords)
#     lengthLE = length(LECoordsVec)
#     lengthTE = length(TECoordsVec)

#     if !isreal(step)
#         LECoordsVec = complex(copy(LECoordsVec))
#         TECoordsVec = complex(copy(TECoordsVec))
#     end

#     for ii in 1:lengthLE # LE bumps
#         LECoordsVec[ii] += step
#         LECoordsWork = Utilities.repack_coords(LECoordsVec, mm, nn)
#         r_f = compute_residualsFromCoords(
#             u, LECoordsWork, TECoords, nodeConn, DVDictList;
#             appendageOptions=appendageOptions,
#             solverOptions=solverOptions,
#             iComp=iComp,
#         )
#         LECoordsVec[ii] -= step
#         ∂r∂x[:, ii] = imag((r_f)) / abs(step)
#     end
#     for ii in 1:lengthTE # TE bumps
#         TECoordsVec[ii] += step
#         TECoordsWork = Utilities.repack_coords(TECoordsVec, mm, nn)
#         r_f = compute_residualsFromCoords(
#             u, LECoords, TECoordsWork, nodeConn, DVDictList;
#             appendageOptions=appendageOptions,
#             solverOptions=solverOptions,
#             iComp=iComp,
#         )
#         TECoordsVec[ii] -= step
#         ∂r∂x[:, ii+lengthLE] = imag((r_f)) / abs(step)
#     end

# end

function compute_∂r∂uFromDV(
    structuralStates, mode="CS";
    DVDictList=[], solverParams=nothing, appendageOptions=Dict(), solverOptions=Dict(), iComp=1, CLMain=0.0
)
    """
    Jacobian of residuals with respect to structural states
    EXCLUDING BC NODES

    u - structural states
    """

    DVDict = DVDictList[iComp]
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)

    if uppercase(mode) == "FIDI" # Finite difference
        # First derivative using 3 stencil points
        backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
        ∂r∂u, = AD.jacobian(
            backend,
            x -> compute_residualsFromDV(x, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain),
            structuralStates,
        )
        # ∂r∂u = FiniteDifferences.jacobian(forward_fdm(2, 1), compute_residuals, structuralStates)

    elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        # NOTE: a little slow but it is accurate
        # This is a tuple
        backend = AD.ZygoteBackend()
        ∂r∂u, = AD.jacobian(
            backend,
            x -> compute_residualsFromDV(x, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain),
            structuralStates,
        )
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
            r_f = compute_residualsFromDV(
                structuralStatesCS, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain
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
        ∂r∂u::Matrix{DTYPE} = solverParams.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
        +solverParams.AICmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
        # The behavior of the analytic derivatives is interesting since it takes about 6 NL iterations to 
        # converge to the same solution as the RAD, which only takes 2 NL iterations.
    else
        error("Invalid mode")
    end

    return ∂r∂u
end

function compute_∂r∂uFromCoords(
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

function compute_residualsFromDV(
    structStates, DVs::Vector, DVLengths::Vector{Int64};
    appendageOptions=Dict(), solverOptions=Dict(), iComp=1, CLMain=0.0, DVDictList=[]
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

    DVDict = Utilities.repack_dvdict(DVs, DVLengths)

    # Need to put DVDict in the iComp spot of the DVDictListWork array
    indicesNotMatchingiComp = 1:length(DVDictList) .!= iComp
    # DVDictWork::Vector = copy(DVDictList[indicesNotMatchingiComp]) # this is a vector...
    # TODO: make this better
    if length(DVDictList) > 1
        if iComp == 1
            DVDictListWork = vcat([DVDict], [DVDictList[2]])
        else
            DVDictListWork = vcat([DVDictList[1]], [DVDict])
        end
    else
        DVDictListWork = [DVDict]
    end
    # There is probably a nicer way to restructure the code so the order of calls is
    # 1. top level solve call to applies the newton raphson
    # 2.    compute_residuals
    # solverOptions["debug"] = false
    # THIS IS SLOW
    _, _, SOLVERPARAMS = setup_problemFromDVDict(DVDictListWork, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)

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
    resVec = SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] * structStates - FOut

    return resVec
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

# function ChainRulesCore.rrule(::typeof(compute_residualsFromCoords))

# end

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