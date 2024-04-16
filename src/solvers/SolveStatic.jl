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
using ..FEMMethods
using ..BeamProperties
using ..EBBeam: NDOF, UIND, VIND, WIND, ΦIND, ΨIND, ΘIND
using ..HydroStrip
using ..SolutionConstants
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..DesignConstants: SORTEDDVS
using ..SolverRoutines: SolverRoutines
using ..Utilities: Utilities
using ..DCFoilSolution
using ..TecplotIO: TecplotIO

# ==============================================================================
#                         COMMON TERMS
# ==============================================================================
const elemType = "COMP2"
const loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solve(SOLVERPARAMS, FEMESH, FOIL, STRUT, DVDict::Dict,
    appendageOptions::Dict, solverOptions::Dict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    """

    outputDir = solverOptions["outputDir"]

    # Initial guess on unknown deflections (excluding BC nodes)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(zeros(length(SOLVERPARAMS.Kmat[1, :])), SOLVERPARAMS.AICmat, DVDict["α₀"], DVDict["rake"], SOLVERPARAMS.dofBlank, elemType;
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    q_ss0 = FEMMethods.solve_structure(SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]], SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]], fTractions[1:end.∉[SOLVERPARAMS.dofBlank]])

    # Actual solve
    qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q_ss0, DVDict;
        is_verbose=true, solverParams=SOLVERPARAMS, appendageOptions=appendageOptions, solverOptions=solverOptions)
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, SOLVERPARAMS.elemType; appendageOptions=appendageOptions)

    # --- Get hydroLoads again on solution ---
    # _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(uSol), structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; 
    # appendageOptions=appendageOptions, STRUT=STRUT, strutChordVec=strutChordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(uSol, SOLVERPARAMS.AICmat, DVDict["α₀"], DVDict["rake"], SOLVERPARAMS.dofBlank, SOLVERPARAMS.elemType; appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC


    write_sol(uSol, fHydro, SOLVERPARAMS.elemType, outputDir)

    STATSOL = DCFoilSolution.StaticSolution{DTYPE}(uSol, fHydro, FEMESH, SOLVERPARAMS, FOIL, STRUT)

    return STATSOL
end

function setup_problem(DVDict::Dict, appendageOptions::Dict, solverOptions::Dict; iComp=1, verbose=false)
    """
    """

    WING, STRUT, _ = InitModel.init_model_wrapper(DVDict, solverOptions, appendageOptions)

    nNodes = WING.nNodes
    nElem = nNodes - 1
    if appendageOptions["config"] == "wing"
        STRUT = WING # just to make the code work
    end
    structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"];
        config=appendageOptions["config"], nElStrut=STRUT.nNodes - 1, spanStrut=DVDict["s_strut"], rake=DVDict["rake"])

    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, WING.chord, DVDict["toc"], WING.ab, DVDict["x_αb"], DVDict["θ"], zeros(10, 2))

    globalK, globalM, _ = FEMMethods.assemble(FEMESH, WING.ab, DVDict["x_αb"], WING, elemType, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, ab_strut=STRUT.ab, x_αb_strut=DVDict["x_αb_strut"], verbose=verbose)
    if iComp == 2
        println("Computing downstream flow effects")
    end
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(FEMESH, WING, size(globalM)[1], DVDict["Λ"], WING.U∞, 0.0, elemType; appendageOptions=appendageOptions, STRUT=STRUT)
    DOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped"; appendageOptions=appendageOptions, verbose=verbose)
    # K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, DOFBlankingList)
    derivMode = "RAD"
    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalK, globalK, globalK, elemType, AIC, derivMode, planformArea, DOFBlankingList)

    return WING, STRUT, SOLVERPARAMS, FEMESH
end

function write_sol(states, fHydro, elemType="bend", outputDir="./OUTPUT/")
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

function write_tecplot(DVDict, STATICSOL, FEMESH, outputDir="./OUTPUT/"; appendageOptions=Dict("config" => "wing"), solverOptions=Dict(), iComp=1)
    """
    General purpose tecplot writer wrapper for flutter solution
    """
    TecplotIO.write_deflections(DVDict, STATICSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)

end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function evalFuncs(
    evalFuncsList::Vector{String}, states::Vector, SOL, DVVec::Vector, DVLengths::Vector{Int64};
    appendageOptions=nothing, solverOptions=nothing
)
    """
    Given {u} and the forces, compute the cost functions
    """

    costFuncs = compute_funcs(evalFuncsList, states, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions)


    return costFuncs
end

function evalFuncs(
    evalFuncs::String, states::Vector, SOL, DVVec::Vector, DVLengths::Vector{Int64};
    appendageOptions=nothing, solverOptions=nothing
)
    """
    Given {u} and the forces, compute the cost functions
    """

    costFuncs = compute_funcs([evalFuncs], states, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions)

    return costFuncs[evalFuncs]
end

function compute_funcs(evalFuncsList::Vector, states, SOL, DVVec, DVLengths; appendageOptions=nothing, solverOptions=nothing)

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
    α₀ = DVDict["α₀"]
    rake = DVDict["rake"]
    WING, STRUT, constants, FEMESH = setup_problem(DVDict, appendageOptions, solverOptions)
    solverOptions["debug"] = false
    forces, _, _ = HydroStrip.integrate_hydroLoads(states, constants.AICmat, α₀, rake, constants.dofBlank, constants.elemType; appendageOptions=appendageOptions, solverOptions=solverOptions)
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
    if "lift" in evalFuncsList
        TotalLift = sum(Lift)
        costFuncs["lift"] = TotalLift
    end
    # Moment about mid-chord (where the finite element is)
    if "moment" in evalFuncsList
        TotalMoment = sum(Moments)
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
        twist = theta[1:globappendageOptions["nNodes"]]

        clalpha, Fxi, CDi = HydroStrip.compute_glauert_circ(DVDict["s"], chordVec, deg2rad(DVDict["α₀"]), solverOptions["U∞"], appendageOptions["nNodes"];
            h=DVDict["depth0"],
            useFS=solverOptions["use_freeSurface"],
            rho=solverOptions["ρ_f"],
            twist=twist,
            debug=solverOptions["debug"],
            solverOptions=solverOptions, # TODO: this should probably happen on the solve mode too
        )

        if "cdi" in evalFuncsList
            costFuncs["cdi"] = CDi
        end
        if "fxi" in evalFuncsList
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
    # if "cdw" in evalFuncs || "fxw" in evalFuncs
    #     # rws = EQ 6.149 FALTINSEN thickness effect on wave resistance
    #     # rwgamma = EQ 6.145 FALTINSEN wave resistance due to lift
    # end
    if "cdpr" in evalFuncsList || "fxpr" in evalFuncsList
        if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
            WSA = 2 * ADIM # both sides
        elseif appendageOptions["config"] == "t-foil"
            WSA = 2 * ADIM + 2 * DVDict["s_strut"] * mean(DVDict["c_strut"])
        end
        println("I'm not debugged")
        # TODO: MAKE WSA AND DRAG A VECTORIZED STRIPWISE CALCULATION
        NU = 1.1892E-06 # kinematic viscosity of seawater at 15C
        Re = globsolverOptions["U∞"] * mean(chordVec) / NU
        Ma = globsolverOptions["U∞"] / 1500
        cfittc = 0.075 / (log10(Re) - 2)^2 # flat plate friction coefficient ITTC 1957
        xcmax = 0.3 # chordwise position of the maximum thickness
        # # --- Raymer equation 12.30 ---
        # FF = (1 .+ 0.6 ./ (xcmax) .* DVDict["toc"] + 100 .* DVDict["toc"].^4) * (1.34*Ma^0.18 * cos(DVDict["Λ"])^0.28)
        # --- Torenbeek 1990 ---
        # First term is increase in skin friction due to thickness and quartic is separation drag
        FF = 1 + 2.7 .* DVDict["toc"] + 100 .* DVDict["toc"] .^ 4
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

function get_sol(DVDict::Dict, solverOptions::Dict, evalFuncs::Vector{String}; iComp=1, FEMESH=nothing)
    """
    Wrapper function to do primal solve and return solution struct
    """

    appendageOptions = solverOptions["appendageList"][iComp]
    WING, STRUT, SOLVERPARAMS, FEMESH = setup_problem(DVDict, appendageOptions, solverOptions; verbose=true)

    SOL = solve(SOLVERPARAMS, FEMESH, WING, STRUT, DVDict, appendageOptions, solverOptions)

    return SOL
end

function cost_funcsFromDVs(DVDict::Dict, iComp::Int64, solverOptions::Dict, evalFuncsList::Vector{String}
)
    """
    Do primal solve with function signature compatible with Zygote
    """

    appendageOptions = solverOptions["appendageList"][iComp]
    # Setup
    FOIL, STRUT, SOLVERPARAMS = setup_problem(DVDict, appendageOptions, solverOptions; verbose=false)
    # Solve
    SOL = solve(SOLVERPARAMS, FOIL, STRUT, DVDict, appendageOptions, solverOptions)
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    costFuncs = evalFuncs(evalFuncsList, SOL.structStates, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions)
    return costFuncs
end

function evalFuncsSens(STATSOLLIST::Vector, evalFuncsSensList::Vector{String}, DVDictList::Vector, FEMESHLIST::Vector, solverOptions::Dict; mode="FiDi")
    """
    Wrapper to compute total sensitivities
    """
    println("===================================================================================================")
    println("        STATIC SENSITIVITIES: ", mode)
    println("===================================================================================================")

    # Initialize output
    funcsSensList::Vector = []
    funcsSens = Dict()

    solverOptions["debug"] = false

    for iComp in eachindex(DVDictList)

        DVDict = DVDictList[iComp]
        FEMESH = FEMESHLIST[iComp]

        if uppercase(mode) == "FIDI" # use finite differences the stupid way

            # It's a tuple bc of DVDict
            sensitivities, = FiniteDifferences.jacobian(
                forward_fdm(2, 1),
                (x1) ->
                    SolveStatic.cost_funcsFromDVs(x1, iComp, solverOptions, evalFuncsSensList),
                DVDict,
            )
            funcsSens = sensitivities

        elseif uppercase(mode) == "ADJOINT"

            println("Computing adjoint for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            u = STATSOL.structStates[1:end.∉[solverParams.dofBlank]]

            @time ∂r∂x = compute_∂r∂x(STATSOL.structStates, DVDict;
                mode="RAD", SOLVERPARAMS=solverParams, appendageOptions=appendageOptions, solverOptions=solverOptions)
            println("Computing ∂r∂u...")
            @time ∂r∂u = compute_∂r∂u(
                u,
                "Analytic";
                solverParams=solverParams
                # , fullState=true
            )
            # psiMat = zeros(Float64, length(u), length(evalFuncsSensList)) # the columns of [\Psi] are the adjoint vectors
            funcsSens = Dict()
            _, DVLengths = Utilities.unpack_dvdict(DVDict)
            for (iDV, dvkey) in enumerate(SORTEDDVS)
                funcsSens[dvkey] = zeros(DTYPE, length(evalFuncsSensList), DVLengths[iDV])
            end

            for (ifunc, costFunc) in enumerate(evalFuncsSensList)
                @time ∂f∂u = compute_∂f∂u(costFunc, STATSOL, DVDict; mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions)
                @time ∂f∂x = compute_∂f∂x(costFunc, STATSOL, DVDict; mode="RAD", appendageOptions=appendageOptions, solverOptions=solverOptions)
                println("+---------------------------------+")
                println(@sprintf("| Computing adjoint: %s ", costFunc))
                println("+---------------------------------+")
                ∂f∂uT = transpose(∂f∂u)
                # println(size(∂f∂uT)) # should be (n_u x n_f) a column vector!
                psiVec = compute_adjointVec(∂r∂u, ∂f∂uT; solverParams=solverParams)
                # psiMat[:, ifunc] = psiVec

                # --- Compute total sensitivities ---
                # funcsSens[costFunc] = ∂f∂x - transpose(psiMat) * ∂r∂x
                # Transpose the adjoint vector so it's now a row vector
                dfdx = ∂f∂x - (transpose(psiVec) * ∂r∂x)
                giDV = 1
                for (iiDV, dvkey) in enumerate(SORTEDDVS)
                    ndv = DVLengths[iiDV]
                    funcsSens[dvkey][ifunc, :] = dfdx[giDV:giDV+ndv-1]
                    giDV += ndv # starting value for next set
                end
            end

        elseif uppercase(mode) == "DIRECT"

            println("Computing direct for component ", iComp)
            STATSOL = STATSOLLIST[iComp]
            solverParams = STATSOL.SOLVERPARAMS
            appendageOptions = solverOptions["appendageList"][iComp]
            u = STATSOL.structStates[1:end.∉[solverParams.dofBlank]]

            @time ∂r∂x = compute_∂r∂x(STATSOL.structStates, DVDict;
                mode="FiDi", solverParams=solverParams, appendageOptions=appendageOptions, solverOptions=solverOptions)

            @time ∂r∂u = compute_∂r∂u(
                u,
                "Analytic";
                solverParams=solverParams,
                # fullState=true
            )
            println("+---------------------------------+")
            println(@sprintf("| Computing direct: "))
            println("+---------------------------------+")
            phiMat = compute_directMatrix(∂r∂u, ∂r∂x; solverParams=solverParams)
            funcsSens = Dict()
            for dvkey in keys(DVDict)
                funcsSens[dvkey] = zeros(DTYPE, length(evalFuncsSensList), length(DVDict[dvkey]))
            end

            for (ifunc, costFunc) in enumerate(evalFuncsSensList)
                ∂f∂u = compute_∂f∂u(costFunc, STATSOL, DVDict; mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions)
                ∂f∂x = compute_∂f∂x(costFunc, STATSOL, DVDict; mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions)

                # --- Compute total sensitivities ---
                # funcsSens[costFunc] = ∂f∂x - ∂f∂u * [ϕ]
                dfdx = ∂f∂x - (∂f∂u[:, 1:end.∉[solverParams.dofBlank]] * phiMat)
                iDV = 1
                for dvkey in keys(DVDict)
                    ndv = length(DVDict[dvkey])
                    funcsSens[dvkey][ifunc, :] = dfdx[iDV:iDV+ndv-1]
                    iDV += ndv # starting value for next set
                end
            end
        else
            error("Invalid mode")
        end

        push!(funcsSensList, funcsSens)

    end

    save(@sprintf("funcsSensList-%s.jld2", mode), "derivs", funcsSensList)

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

function compute_∂r∂x(
    allStructStates::Vector, DVDict::Dict;
    mode="FiDi", SOLVERPARAMS=nothing, appendageOptions=nothing, solverOptions=nothing
)
    """
    Partial derivatives of residuals with respect to design variables w/o reconverging the solution
    """

    println("Computing ∂r∂x...")
    u = allStructStates[1:end.∉[SOLVERPARAMS.dofBlank]]
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    if uppercase(mode) == "FIDI" # Finite difference
        dh = 1e-3
        ∂r∂x = zeros(DTYPE, length(u), length(DVVec))
        println("step size:", dh)
        for ii in eachindex(DVVec)
            r_i = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
            )
            DVVec[ii] += dh
            r_f = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
            )
            DVVec[ii] -= dh

            ∂r∂x[:, ii] = (r_f - r_i) ./ dh
        end
    elseif uppercase(mode) == "CS" # TODO: THIS WOULD BE THE BEST APPROACH
        dh = 1e-14
        ∂r∂x = zeros(DTYPE, length(u), length(DVVec))
        println("step size:", dh)
        for ii in eachindex(DVVec)
            DVVec[ii] += 1im * dh
            r_f = SolveStatic.compute_residuals(
                u, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
            )
            DVVec[ii] -= 1im * dh

            ∂r∂x[:, ii] = imag((r_f)) ./ dh
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
            ),
            DVVec, # compute deriv at this DV
        )
    else
        error("Invalid mode")
    end
    return ∂r∂x
end

function compute_∂f∂x(
    costFunc::String, SOL, DVDict::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict())

    println("Computing ∂f∂x...")
    # ∂f∂u = zeros(Float64, length(DVDict))
    # Do analytic
    if uppercase(mode) == "ANALYTIC"
        println("Haven't done this yet")
    elseif uppercase(mode) == "RAD" # WORKS
        backend = AD.ZygoteBackend()
        DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
        ∂f∂x, = AD.gradient(
            backend,
            x -> evalFuncs(costFunc, SOL.structStates, SOL, x, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions),
            DVVec, # compute deriv at this DV
        )
        ∂f∂x = reshape(∂f∂x, 1, length(∂f∂x))
    elseif uppercase(mode) == "FIDI"
        dh = 1e-3
        DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
        ∂f∂x = zeros(DTYPE, length(DVVec))
        println("step size:", dh)
        for ii in eachindex(DVVec)
            f_i = evalFuncs(costFunc, SOL.structStates, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions)
            DVVec[ii] += dh
            f_f = evalFuncs(costFunc, SOL.structStates, SOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions)
            DVVec[ii] -= dh

            ∂f∂x[ii] = (f_f - f_i) / dh
        end
        ∂f∂x = reshape(∂f∂x, 1, length(∂f∂x))
    end
    return ∂f∂x
end

function compute_∂f∂u(
    costFunc::String, SOL, DVDict::Dict;
    mode="RAD", appendageOptions=Dict(), solverOptions=Dict()
)
    """
    Compute the gradient of the cost functions with respect to the structural states
    SOL is the solution struct at the current design point
    """

    println("Computing ∂f∂u...")
    ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
    DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
    # Do analytic
    if uppercase(mode) == "ANALYTIC"
        # NOTE: NOT DEBUGGED OR WORKING!!!
        states = SOL.structStates
        # forces = SOL.fHydro
        foil = SOL.FOIL
        nnodes = foil.nNodes
        constants = SOL.SOLVERPARAMS

        chordVec = DVDict["c"]

        # There should be no reason why the density or flow speed is diff 
        # between 'foil' data structures in the multi-appendage case
        # so this line is ok
        qdyn = 0.5 * foil.ρ_f * foil.U∞^2

        fHydro = constants.AICmat * states

        # theta = states[5:NDOF:end]
        # Moments = forces[5:NDOF:end]
        # W = states[3:NDOF:end]
        # Lift = forces[3:NDOF:end]


        if appendageOptions["config"] == "wing"
            ADIM = constants.planformArea
        elseif appendageOptions["config"] == "t-foil" || appendageOptions["config"] == "full-wing"
            ADIM = 2 * constants.planformArea
        end

        # ************************************************
        #     COMPUTE COST FUNCS
        # ************************************************
        if "wtip" == costFunc
            ∂f∂u[end-6] = 1.0
        end
        if "lift" == costFunc || "cl" == costFunc
            liftIdx = 3
            for ii in liftIdx:NDOF:length(states)
                # Compute the derivative of each traction load wrt the states
                dforce_i_du = constants.AICmat[ii, :]
                ∂f∂u .+= dforce_i_du
            end
            if "cl" == costFunc
                ∂f∂u ./= (qdyn * ADIM)
            end
        end
        # Moment about mid-chord (where the finite element is)
        if "moment" == costFunc || "cmy" == costFunc
            momentIdx = 5
            for ii in momentIdx:NDOF:length(states)
                # Compute the derivative of each traction load wrt the states
                dforce_i_du = constants.AICmat[ii, :]
                ∂f∂u .+= dforce_i_du
            end
            # Coefficient of moment about the mid-chord
            if "cmy" == costFunc
                ∂f∂u ./= (qdyn * ADIM * mean(chordVec))
            end
        end

        # if "cd" in costFunc
        #     CD = 0.0
        #     costFuncs["cd"] = CD
        # end
        # if "cdi" in costFunc || "fxi" in costFunc
        #     twist = theta[1:globappendageOptions["nNodes"]]

        #     clalpha, Fxi, CDi = HydroStrip.compute_glauert_circ(DVDict["s"], chordVec, deg2rad(DVDict["α₀"]), solverOptions["U∞"], appendageOptions["nNodes"];
        #         h=DVDict["depth0"],
        #         useFS=solverOptions["use_freeSurface"],
        #         rho=solverOptions["ρ_f"],
        #         twist=twist,
        #         debug=solverOptions["debug"],
        #         solverOptions=solverOptions,
        #     )

        #     if "cdi" in costFunc
        #         costFuncs["cdi"] = CDi
        #     end
        #     if "fxi" in costFunc
        #         costFuncs["fxi"] = Fxi
        #     end
        # end
        # # From Hörner Chapter 8
        # if "cdj" in costFunc || "fxj" in costFunc
        #     tocbar = 0.5 * (DVDict["toc"][1] + DVDict["toc_strut"][1])
        #     CDt = 17 * (tocbar)^2 - 0.05
        #     dj = CDt * (qdyn * (tocbar * DVDict["c"][1])^2)
        #     CDj = dj / (qdyn * ADIM)
        #     costFuncs["cdj"] = CDj
        #     costFuncs["fxj"] = dj
        # end
        # if "cds" in costFunc || "fxs" in costFunc
        #     t = DVDict["toc_strut"][end] * DVDict["c_strut"][end]
        #     # --- Hörner CHapter 10 ---
        #     # CDts = 0.24
        #     # ds = CDts * (qdyn * (t)^2)
        #     # CDs = ds / (qdyn * ADIM)
        #     # Chapman 1971 assuming x/c = 0.35
        #     CDs = 0.009 + 0.013 * DVDict["toc_strut"][end]
        #     ds = CDs * qdyn * t * DVDict["c_strut"][end]
        #     costFuncs["cds"] = CDs
        #     costFuncs["fxs"] = ds
        # end
        # # # TODO:
        # # if "cdw" in evalFuncs || "fxw" in evalFuncs
        # #     # rws = EQ 6.149 FALTINSEN thickness effect on wave resistance
        # #     # rwgamma = EQ 6.145 FALTINSEN wave resistance due to lift
        # # end
        # if "cdpr" in costFunc || "fxpr" in costFunc
        #     if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
        #         WSA = 2 * ADIM # both sides
        #     elseif appendageOptions["config"] == "t-foil"
        #         WSA = 2 * ADIM + 2 * DVDict["s_strut"] * mean(DVDict["c_strut"])
        #     end
        #     println("I'm not debugged")
        #     NU = 1.1892E-06 # kinematic viscosity of seawater at 15C
        #     Re = globsolverOptions["U∞"] * mean(chordVec) / NU
        #     Ma = globsolverOptions["U∞"] / 1500
        #     cfittc = 0.075 / (log10(Re) - 2)^2 # flat plate friction coefficient ITTC 1957
        #     xcmax = 0.3 # chordwise position of the maximum thickness
        #     # # --- Raymer equation 12.30 ---
        #     # FF = (1 .+ 0.6 ./ (xcmax) .* DVDict["toc"] + 100 .* DVDict["toc"].^4) * (1.34*Ma^0.18 * cos(DVDict["Λ"])^0.28)
        #     # --- Torenbeek 1990 ---
        #     # First term is increase in skin friction due to thickness and quartic is separation drag
        #     FF = 1 + 2.7 .* DVDict["toc"] + 100 .* DVDict["toc"] .^ 4
        #     FF = mean(FF)
        #     Df = qdyn * WSA * cfittc
        #     Dpr = Df * FF
        #     costFuncs["fxpr"] = Dpr
        #     costFuncs["cdpr"] = Dpr / (qdyn * ADIM)
        # end
        # # --- Center of forces ---
        # # These calculations are in local appendage frame
        # if "cofz" in costFunc # center of forces in z direction
        #     xcenter = sum(Lift .* SOL.FEMESH.mesh[:, XDIM]) / sum(Lift)
        #     ycenter = sum(Lift .* SOL.FEMESH.mesh[:, YDIM]) / sum(Lift)
        #     zcenter = sum(Lift .* SOL.FEMESH.mesh[:, ZDIM]) / sum(Lift)
        #     costFuncs["cofz"] = [xcenter, ycenter, zcenter]
        # end
        # if "comy" in costFunc # center of moments about y axis
        #     xcenter = sum(Moments .* SOL.FEMESH.mesh[:, XDIM]) / sum(Moments)
        #     ycenter = sum(Moments .* SOL.FEMESH.mesh[:, YDIM]) / sum(Moments)
        #     zcenter = sum(Moments .* SOL.FEMESH.mesh[:, ZDIM]) / sum(Moments)
        #     costFuncs["comy"] = [xcenter, ycenter, zcenter]
        # end


    elseif uppercase(mode) == "RAD" # works
        backend = AD.ZygoteBackend()
        ∂f∂u, = AD.gradient(
            backend,
            u -> evalFuncs(
                costFunc, u, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions),
            SOL.structStates,
        )
        ∂f∂u = reshape(∂f∂u, 1, length(∂f∂u))
    elseif uppercase(mode) == "FIDI" # Finite difference
        dh = 1e-3
        ∂f∂u = zeros(DTYPE, 1, length(SOL.structStates))
        println("step size:", dh)
        for ii in eachindex(SOL.structStates)
            r_i = SolveStatic.evalFuncs(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
            )
            SOL.structStates[ii] += dh
            r_f = SolveStatic.evalFuncs(
                [costFunc], SOL.structStates, SOL, DVVec, DVLengths;
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
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

function compute_∂r∂u(structuralStates, mode="FiDi"; solverParams=nothing)
    """
    Jacobian of residuals with respect to structural states
    EXCLUDING BC NODES

    u - structural states
    """

    if uppercase(mode) == "FIDI" # Finite difference
        # First derivative using 3 stencil points
        ∂r∂u = FiniteDifferences.jacobian(forward_fdm(2, 1), compute_residuals, structuralStates)

    elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        # This is a tuple
        ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

        # elseif uppercase(mode) == "RAD" # Reverse automatic differentiation
        #     @time ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)

    elseif uppercase(mode) == "ANALYTIC"
        # In the case of a linear elastic beam under static fluid loading, 
        # dr/du = Ks + Kf
        # NOTE Kf = AIC matrix
        # where AIC * states = forces on RHS (external)
        # TODO: longer term, the AIC is influenced by the structural states b/c of the twist distribution
        ∂r∂u = solverParams.Kmat[1:end.∉[solverParams.dofBlank], 1:end.∉[solverParams.dofBlank]]
        +solverParams.AICmat[1:end.∉[solverParams.dofBlank], 1:end.∉[solverParams.dofBlank]]
        # if fullState # need to include the clamped nodes
        #     ∂r∂u = solverParams.Kmat + solverParams.AICmat
        # end
        # The behavior of the analytic derivatives is interesting since it takes about 6 NL iterations to 
        # converge to the same solution as the RAD, which only takes 2 NL iterations.
    else
        error("Invalid mode")
    end

    return ∂r∂u
end

function compute_residuals(structStates, DVs, DVLengths::Vector{Int64};
    appendageOptions=Dict(), solverOptions=Dict())
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

    allStructuralStates, _ = FEMMethods.put_BC_back(structStates, elemType; appendageOptions=appendageOptions)
    foilTotalStates = SolverRoutines.return_totalStates(allStructuralStates, DVDict, elemType; appendageOptions=appendageOptions)

    # There is probably a nicer way to restructure the code so the order of calls is
    # 1. top level solve call to applies the newton raphson
    # 2.    compute_residuals
    solverOptions["debug"] = false
    # I THINK THIS IS SLOW
    _, _, SOLVERPARAMS = setup_problem(DVDict, appendageOptions, solverOptions)


    # --- Outputs ---
    F = -SOLVERPARAMS.AICmat * foilTotalStates
    FOut = F[1:end.∉[SOLVERPARAMS.dofBlank]]


    # --- Stack them ---
    resVec = SOLVERPARAMS.Kmat[1:end.∉[SOLVERPARAMS.dofBlank], 1:end.∉[SOLVERPARAMS.dofBlank]] * structStates - FOut

    return resVec
end

function compute_directMatrix(∂r∂u, ∂r∂x; solverParams=nothing)
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