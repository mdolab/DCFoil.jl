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
export solve

# --- PACKAGES ---
using AbstractDifferentiation: AbstractDifferentiation as AD
using FiniteDifferences
using Zygote
using ChainRulesCore
using LinearAlgebra
using Statistics
using JSON
using JLD2
using Printf, DelimitedFiles

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex, DTYPE
using ..InitModel
using ..FEMMethods: FEMMethods, StructMesh
using ..BeamProperties
using ..EBBeam: NDOF, UIND, VIND, WIND, ΦIND, ΨIND, ΘIND
using ..HydroStrip: HydroStrip
using ..SolutionConstants: SolutionConstants, XDIM, YDIM, ZDIM, DCFoilSolverParams
using ..DesignConstants: SORTEDDVS, DynamicFoil
using ..SolverRoutines: SolverRoutines
using ..Utilities: Utilities
using ..DCFoilSolution: DCFoilSolution, StaticSolution
using ..TecplotIO: TecplotIO

# ==============================================================================
#                         COMMON TERMS
# ==============================================================================
const elemType = "COMP2"
const loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solve(
    SOLVERPARAMS::DCFoilSolverParams, FEMESH::StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, DVDictList::Vector,
    appendageOptions::Dict, solverOptions::Dict;
    iComp=1, CLMain=0.0
)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    """

    DVDict = DVDictList[iComp]
    outputDir = solverOptions["outputDir"]

    # Initial guess on unknown deflections (excluding BC nodes)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], SOLVERPARAMS.dofBlank, SOLVERPARAMS.downwashAngles, elemType;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    q_ss0 = FEMMethods.solve_structure(SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]], SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]], fTractions[1:end.∉[SOLVERPARAMS.dofBlank]])

    if lowercase(solverOptions["res_jacobian"]) == "cs"
        mode = "CS"
    elseif lowercase(solverOptions["res_jacobian"]) == "rad"
        mode = "RAD"
    elseif lowercase(solverOptions["res_jacobian"]) == "analytic"
        mode = "Analytic"
    end

    # Actual solve
    qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q_ss0, DVDictList;
        is_verbose=true,
        solverParams=SOLVERPARAMS,
        appendageOptions=appendageOptions,
        solverOptions=solverOptions,
        mode=mode,
        iComp=iComp,
        CLMain=CLMain,
    )
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, SOLVERPARAMS.elemType; appendageOptions=appendageOptions)

    # --- Get hydroLoads again on solution ---
    # _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(uSol), structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; 
    # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, DVDict["alfa0"], DVDict["rake"], SOLVERPARAMS.dofBlank, SOLVERPARAMS.downwashAngles, SOLVERPARAMS.elemType;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC


    write_sol(uSol, fHydro, SOLVERPARAMS.elemType, outputDir)

    STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

    return STATSOL
end

function setup_problem(
    DVDictList, appendageOptions::Dict, solverOptions::Dict;
    iComp=1, CLMain=0.0, verbose=false
)
    """
    """
    DVDict = DVDictList[iComp]

    WING, STRUT, _ = InitModel.init_model_wrapper(DVDict, solverOptions, appendageOptions)

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if appendageOptions["config"] == "wing"
        STRUT = WING # just to make the code work
    end
    structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"];
        config=appendageOptions["config"], nElStrut=STRUT.nNodes - 1, spanStrut=DVDict["s_strut"], rake=DVDict["rake"])

    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, WING.chord, DVDict["toc"], WING.ab, DVDict["x_ab"], DVDict["theta_f"], zeros(10, 2))

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, WING.ab, DVDict["x_ab"], WING, elemType, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, ab_strut=STRUT.ab, x_αb_strut=DVDict["x_ab_strut"], verbose=verbose)
    alphaCorrection::DTYPE = 0.0
    if iComp > 1
        alphaCorrection = HydroStrip.correct_downwash(iComp, CLMain, DVDictList, solverOptions)
    end
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(FEMESH, WING, size(globalM)[1], DVDict["sweep"], WING.U∞, 0.0, elemType; appendageOptions=appendageOptions, STRUT=STRUT)
    DOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
    # K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, DOFBlankingList)
    derivMode = "RAD"
    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, globalK, elemType, AIC, derivMode, planformArea, DOFBlankingList, alphaCorrection)

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
function evalFuncs(
    evalFuncsList::Vector{String}, states::Vector, SOL::StaticSolution, DVVec::Vector, DVLengths::Vector{Int64};
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)
    """
    Given {u} and the forces, compute the cost functions
    """

    costFuncs = compute_funcs(evalFuncsList, states, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)


    return costFuncs
end

function evalFuncs(
    evalFuncs::String, states::Vector, SOL::StaticSolution, DVVec::Vector, DVLengths::Vector{Int64};
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)
    """
    Allow this to work for a single cost function too
    """

    costFuncs = compute_funcs([evalFuncs], states, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

    return costFuncs[evalFuncs]
end

function compute_funcs(
    evalFuncsList::Vector, states::Vector, SOL::StaticSolution, DVVec::Vector, DVLengths::Vector{Int64};
    appendageOptions=nothing, solverOptions=nothing, DVDictList=[], iComp=1, CLMain=0.0
)

    DVDict = Utilities.repack_dvdict(DVVec, DVLengths)
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
    WING, STRUT, constants, FEMESH = setup_problem(DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)
    # solverOptions["debug"] = false
    forces, _, _ = HydroStrip.integrate_hydroLoads(states, constants.AICmat, α₀, rake, constants.dofBlank, constants.downwashAngles, constants.elemType;
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

    if "wtip" in evalFuncsList
        w_tip = W[end]
        costFuncs["wtip"] = w_tip
    end
    if "psitip" in evalFuncsList
        psi_tip = theta[end]
        costFuncs["psitip"] = psi_tip
    end
    TotalLift = sum(Lift)
    if "lift" in evalFuncsList
        costFuncs["lift"] = TotalLift
    end
    # Moment about mid-chord (where the finite element is)
    TotalMoment = sum(Moments)
    if "moment" in evalFuncsList
        costFuncs["moment"] = TotalMoment
    end
    if "cl" in evalFuncsList
        CL = TotalLift / (qdyn * ADIM)
        costFuncs["cl"] = CL
    end
    # Coefficient of moment about the mid-chord
    if "cmy" in evalFuncsList
        CM = TotalMoment / (qdyn * ADIM * meanChord)
        costFuncs["cmy"] = CM
    end
    if "cd" in evalFuncsList
        CD = 0.0
        costFuncs["cd"] = CD
    end
    if "cdi" in evalFuncsList || "fxi" in evalFuncsList
        wingTwist = theta[1:appendageOptions["nNodes"]]

        _, Fxi, CDi = HydroStrip.compute_glauert_circ(
            DVDict["s"],
            WING.chord,
            deg2rad(DVDict["alfa0"]),
            solverOptions["U∞"], appendageOptions["nNodes"];
            h=DVDict["depth0"],
            useFS=solverOptions["use_freeSurface"],
            rho=solverOptions["ρ_f"],
            twist=wingTwist,
            debug=solverOptions["debug"],
            config=appendageOptions["config"],
        )

        if "cdi" in evalFuncsList
            costFuncs["cdi"] = CDi
        end
        if "fxi" in evalFuncsList
            Fxi = CDi * qdyn * ADIM
            costFuncs["fxi"] = Fxi
        end
    end
    # From Hörner Chapter 8
    if "cdj" in evalFuncsList || "fxj" in evalFuncsList
        tocbar = 0.5 * (DVDict["toc"][1] + DVDict["toc_strut"][1])
        CDt = 17 * (tocbar)^2 - 0.05
        dj = CDt * (qdyn * (tocbar * DVDict["c"][1])^2)
        CDj = dj / (qdyn * ADIM)
        costFuncs["cdj"] = CDj
        costFuncs["fxj"] = dj
    end
    if "cds" in evalFuncsList || "fxs" in evalFuncsList
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
    # if "cdw" in evalFuncs || "fxw" in evalFuncs
    #     # rws = EQ 6.149 FALTINSEN thickness effect on wave resistance
    #     # rwgamma = EQ 6.145 FALTINSEN wave resistance due to lift
    # end
    if "cdpr" in evalFuncsList || "fxpr" in evalFuncsList # profile drag
        if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
            WSA = 2 * ADIM # both sides
        elseif appendageOptions["config"] == "t-foil"
            WSA = 2 * ADIM + 2 * DVDict["s_strut"] * mean(DVDict["c_strut"])
        end
        println("I'm not debugged")
        # TODO: MAKE WSA AND DRAG A VECTORIZED STRIPWISE CALCULATION
        NU = 1.1892E-06 # kinematic viscosity of seawater at 15C
        Re = solverOptions["U∞"] * meanChord / NU
        # Ma = solverOptions["U∞"] / 1500
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
    if "cofz" in evalFuncsList # center of forces in z direction
        xcenter = sum(Lift .* FEMESH.mesh[:, XDIM]) / sum(Lift)
        ycenter = sum(Lift .* FEMESH.mesh[:, YDIM]) / sum(Lift)
        zcenter = sum(Lift .* FEMESH.mesh[:, ZDIM]) / sum(Lift)
        costFuncs["cofz"] = [xcenter, ycenter, zcenter]
    end

    if "comy" in evalFuncsList # center of moments about y axis
        xcenter = sum(Moments .* FEMESH.mesh[:, XDIM]) / sum(Moments)
        ycenter = sum(Moments .* FEMESH.mesh[:, YDIM]) / sum(Moments)
        zcenter = sum(Moments .* FEMESH.mesh[:, ZDIM]) / sum(Moments)
        costFuncs["comy"] = [xcenter, ycenter, zcenter]
    end

    return costFuncs
end

function get_sol(
    DVDictList, solverOptions::Dict, evalFuncs::Vector{String};
    iComp=1, CLMain=0.0
)
    """
    Wrapper function to do primal solve and return solution struct
    """

    DVDict = DVDictList[iComp]
    appendageOptions = solverOptions["appendageList"][iComp]
    WING, STRUT, SOLVERPARAMS, FEMESH = setup_problem(DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain, verbose=true)

    SOL = solve(SOLVERPARAMS, FEMESH, WING, STRUT, DVDictList, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)

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
    FOIL, STRUT, SOLVERPARAMS, FEMESH = setup_problem(DVDictList, appendageOptions, solverOptions;
        verbose=false, iComp=iComp, CLMain=CLMain)
    # Solve
    SOL = solve(SOLVERPARAMS, FEMESH, FOIL, STRUT, DVDictList, appendageOptions, solverOptions;
        iComp=iComp, CLMain=CLMain)
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    costFuncs = evalFuncs(
        evalFuncsList, SOL.structStates, SOL, DVVec, DVLengths;
        appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)
    return costFuncs
end

function evalFuncsSens(
    STATSOLLIST::Vector, evalFuncsSensList::Vector{String}, DVDictList::Vector, FEMESHLIST::Vector, solverOptions::Dict;
    mode="FiDi", CLMain=0.0
)
    """
    Wrapper to compute total sensitivities
    """
    println("===================================================================================================")
    println("        STATIC SENSITIVITIES: ", mode)
    println("===================================================================================================")

    # Initialize output
    funcsSensList::Vector = []

    solverOptions["debug"] = false

    # --- Loop foils ---
    for iComp in eachindex(DVDictList)

        DVDict = DVDictList[iComp]
        FEMESH = FEMESHLIST[iComp]
        _, DVLengths = Utilities.unpack_dvdict(DVDict)

        DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
        funcsSens = Dict()
        # for (iDV, dvkey) in enumerate(SORTEDDVS)
        #     funcsSens[dvkey] = zeros(DTYPE, length(evalFuncsSensList), DVLengths[iDV])
        # end

        if uppercase(mode) == "FIDI" # use finite differences the stupid way
            dh = 1e-4
            println("step size: ", dh)
            dfdx = zeros(DTYPE, length(evalFuncsSensList), sum(DVLengths))
            DVVecMod = copy(DVVec)
            for ii in eachindex(DVVec)
                f_i = SolveStatic.cost_funcsFromDVs(DVDict, iComp, solverOptions, evalFuncsSensList;
                    DVDictList=DVDictList, CLMain=CLMain
                )
                DVVecMod[ii] += dh
                DVDict = Utilities.repack_dvdict(DVVecMod, DVLengths)
                f_f = SolveStatic.cost_funcsFromDVs(DVDict, iComp, solverOptions, evalFuncsSensList;
                    DVDictList=DVDictList, CLMain=CLMain
                )
                DVVecMod[ii] -= dh
                DVDict = Utilities.repack_dvdict(DVVecMod, DVLengths)

                for ifunc in eachindex(evalFuncsSensList)
                    dfdx[ifunc, ii] = (f_f[evalFuncsSensList[ifunc]] - f_i[evalFuncsSensList[ifunc]]) / dh
                end
            end

            for (ifunc, costFunc) in enumerate(evalFuncsSensList)
                giDV = 1
                for (iiDV, dvkey) in enumerate(SORTEDDVS)
                    ndv = DVLengths[iiDV]
                    funcsSens[dvkey][ifunc, :] = dfdx[ifunc, giDV:giDV+ndv-1]
                    giDV += ndv # starting value for next set
                end
            end

        elseif uppercase(mode) == "ADJOINT"
            println("TOC adjoint derivative is wrong")
            # TODO: DEBUGGIN WHERE TOC influence is lost :(

            println("Computing adjoint for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            u = STATSOL.structStates[1:end.∉[solverParams.dofBlank]]

            @time ∂r∂x = compute_∂r∂x(STATSOL.structStates, DVDict;
                mode="FiDi", SOLVERPARAMS=solverParams, appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp)

            println("Computing ∂r∂u...")
            @time ∂r∂u = compute_∂r∂u(
                u,
                solverOptions["res_jacobian"];
                solverParams=solverParams,
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList,
                CLMain=CLMain,
                iComp=iComp
            )

            # --- Loop over cost functions ---
            for (ifunc, costFunc) in enumerate(evalFuncsSensList)
                funcsSens[costFunc] = Dict()

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
                    solverParams=solverParams)

                # --- Compute total sensitivities ---
                # funcsSens[costFunc] = ∂f∂x - transpose(psiMat) * ∂r∂x
                # Transpose the adjoint vector so it's now a row vector
                dfdx = ∂f∂x - (transpose(psiVec) * ∂r∂x)
                giDV = 1

                for (iiDV, dvkey) in enumerate(SORTEDDVS)
                    ndv = DVLengths[iiDV]
                    
                    # --- Pack sensitivities into dictionary ---
                    funcsSens = Utilities.pack_funcsSens(funcsSens, costFunc, dvkey, dfdx[giDV:giDV+ndv-1])

                    giDV += ndv # starting value for next set
                end
            end

        elseif uppercase(mode) == "DIRECT"

            println("Computing direct for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            u = STATSOL.structStates[1:end.∉[solverParams.dofBlank]]

            @time ∂r∂x = compute_∂r∂x(
                STATSOL.structStates, DVDict;
                mode="FiDi", SOLVERPARAMS=solverParams, appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, CLMain=CLMain, iComp=iComp
            )

            @time ∂r∂u = compute_∂r∂u(
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

            for (ifunc, costFunc) in enumerate(evalFuncsSensList)
                ∂f∂u = compute_∂f∂u(costFunc, STATSOL, DVDict;
                    mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
                )
                ∂f∂x = compute_∂f∂x(costFunc, STATSOL, DVDict;
                    mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

                # --- Compute total sensitivities ---
                # funcsSens[costFunc] = ∂f∂x - ∂f∂u * [ϕ]
                dfdx = ∂f∂x - (∂f∂u[:, 1:end.∉[solverParams.dofBlank]] * phiMat)
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
            x -> evalFuncs(costFunc, SOL.structStates, SOL, x, DVLengths;
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
            f_i = evalFuncs(costFunc, SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            DVVec[ii] += dh
            f_f = evalFuncs(costFunc, SOL.structStates, SOL, DVVec, DVLengths;
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
            u -> evalFuncs(
                costFunc, u, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            ),
            SOL.structStates,
        )
        ∂f∂u = reshape(∂f∂u, 1, length(∂f∂u))
    elseif uppercase(mode) == "FIDI" # Finite difference
        dh = 1e-4
        println("step size:", dh)
        ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
        for ii in eachindex(SOL.structStates)
            r_i = SolveStatic.evalFuncs(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] += dh
            r_f = SolveStatic.evalFuncs(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList, iComp=iComp, CLMain=CLMain
            )
            SOL.structStates[ii] -= dh

            ∂f∂u[1, ii] = (r_f[costFunc] - r_i[costFunc]) / dh
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
    allStructStates, DVDict::Dict;
    mode="FiDi", SOLVERPARAMS=nothing, appendageOptions=nothing, 
    solverOptions=nothing, iComp=1, CLMain=0.0, DVDictList=[]
)
    """
    Partial derivatives of residuals with respect to design variables w/o reconverging the solution
    """

    println("Computing ∂r∂x...")
    u = allStructStates[1:end.∉[SOLVERPARAMS.dofBlank]]
    DVDictList[iComp] = DVDict
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    if uppercase(mode) == "FIDI" # Finite difference

        # This is the manual FD
        dh = 1e-4
        ∂r∂x = zeros(DTYPE, length(u), length(DVVec))
        println("step size: ", dh)
        for ii in eachindex(DVVec)
            r_i = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
                CLMain=CLMain,
                DVDictList=DVDictList,
            )
            DVVec[ii] += dh
            r_f = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                DVDictList=DVDictList,
                iComp=iComp,
                CLMain=CLMain,
            )
            DVVec[ii] -= dh

            ∂r∂x[:, ii] = (r_f - r_i) ./ dh
        end

    elseif uppercase(mode) == "CS" # TODO: THIS WOULD BE THE BEST APPROACH BUT DOESN'T WORK RIGHT NOW
        dh = 1e-100
        ∂r∂x = zeros(DTYPE, length(u), length(DVVec))

        println("step size: ", dh)
        
        # create a complex copy of the design variables
        DVVecCS = complex(copy(DVVec))
        for ii in eachindex(DVVec)
            DVVecCS[ii] += 1im * dh
            r_f = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
                CLMain=CLMain,
                DVDictList=DVDictList,
            )
            DVVecCS[ii] -= 1im * dh

            ∂r∂x[:, ii] = imag((r_f)) / dh
        end

    elseif uppercase(mode) == "ANALYTIC"

    elseif uppercase(mode) == "FAD" # this is fked
        error("Not implemented")
    elseif uppercase(mode) == "RAD" # WORKS
        backend = AD.ZygoteBackend()
        ∂r∂x, = AD.jacobian(
            backend,
            x -> SolveStatic.compute_residuals(
                u,
                x,
                DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
                CLMain=CLMain,
                DVDictList=DVDictList,
            ),
            DVVec, # compute deriv at this DV
        )
    else
        error("Invalid mode")
    end
    return ∂r∂x
end

function compute_∂r∂u(
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
            x -> compute_residuals(x, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain),
            structuralStates,
        )
        # ∂r∂u = FiniteDifferences.jacobian(forward_fdm(2, 1), compute_residuals, structuralStates)

    elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        # NOTE: a little slow but it is accurate
        # This is a tuple
        backend = AD.ZygoteBackend()
        ∂r∂u, = AD.jacobian(
            backend,
            x -> compute_residuals(x, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain),
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

            r_i = compute_residuals(
                structuralStatesCS, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain
            )
            structuralStatesCS[ii] += dh * 1im
            r_f = compute_residuals(
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
        ∂r∂u = solverParams.Kmat[1:end.∉[solverParams.dofBlank], 1:end.∉[solverParams.dofBlank]]
        +solverParams.AICmat[1:end.∉[solverParams.dofBlank], 1:end.∉[solverParams.dofBlank]]
        # The behavior of the analytic derivatives is interesting since it takes about 6 NL iterations to 
        # converge to the same solution as the RAD, which only takes 2 NL iterations.
    else
        error("Invalid mode")
    end

    return ∂r∂u
end

function compute_residuals(
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
    _, _, SOLVERPARAMS = setup_problem(DVDictListWork, appendageOptions, solverOptions; iComp=iComp, CLMain=CLMain)

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, elemType; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(
        allStructuralStates, DVDict, elemType;
        appendageOptions=appendageOptions, alphaCorrection=SOLVERPARAMS.downwashAngles
    )


    # --- Outputs ---
    F = -SOLVERPARAMS.AICmat * foilTotalStates
    FOut = F[1:end.∉[SOLVERPARAMS.dofBlank]]


    # --- Stack them ---
    resVec = SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]] * structStates - FOut

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

function compute_adjointVec(∂r∂u, ∂f∂uT; solverParams=nothing)
    """
    Computes adjoint vector
    If ∂f∂uT is a vector, it might be the same as passing in ∂f∂u
    """

    println("WARNING: REMOVING CLAMPED NODE CONTRIBUTION (There's no hydro force at the clamped node...)")
    ∂f∂uT = ∂f∂uT[1:end.∉[solverParams.dofBlank]]
    ψ = transpose(∂r∂u) \ ∂f∂uT

    return ψ
end

end # end module