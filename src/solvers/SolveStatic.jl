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

# --- Libraries ---
using FiniteDifferences, ChainRulesCore
using LinearAlgebra, Statistics
using JSON
using Zygote
using Printf, DelimitedFiles

# --- DCFoil modules ---
include("../InitModel.jl")
using .InitModel

include("../struct/BeamProperties.jl")
using .BeamProperties

include("../struct/EBBeam.jl")
using .EBBeam: NDOF

include("../struct/FEMMethods.jl")
using .FEMMethods

include("../hydro/HydroStrip.jl")
using .HydroStrip

include("../constants/SolutionConstants.jl")
using .SolutionConstants

include("./SolverRoutines.jl")
using .SolverRoutines

include("./DCFoilSolution.jl")
using .DCFoilSolution

include("../io/TecplotIO.jl")
using .TecplotIO

# ==============================================================================
#                         COMMON VARIABLES
# ==============================================================================
elemType = "COMP2"
loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solve(FEMESH, DVDict::Dict, evalFuncs, solverOptions::Dict, appendageOptions::Dict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    Inputs
    ------
    DVDict: Dict
        The dictionary of design variables
    evalFuncs: Dict
        The dictionary of functions to evaluate
    solverOptions: Dict
        The dictionary of solver options
    Returns
    -------
    costFuncs: Dict
        The dictionary of cost functions
    """
    # ************************************************
    #     INITIALIZE
    # ************************************************
    outputDir = solverOptions["outputDir"]
    global FOIL, STRUT = InitModel.init_model_wrapper(DVDict, solverOptions) # seems to only be global in this module
    nNodes = FOIL.nNodes
    global globappendageOptions = appendageOptions
    # global globsolverOptions = solverOptions
    global gDVDict = DVDict
    println("====================================================================================")
    println("          BEGINNING STATIC HYDROELASTIC SOLUTION")
    println("====================================================================================")

    # ************************************************
    #     SOLVE FEM FIRST TIME
    # ************************************************
    abVec = DVDict["ab"]
    x_αbVec = DVDict["x_αb"]
    global chordVec = DVDict["c"] # need for evalFuncs
    ebVec = 0.25 * chordVec .+ abVec
    if appendageOptions["config"] == "t-foil"
        strutchordVec = DVDict["c_strut"]
        strutabVec = DVDict["ab_strut"]
        strutx_αbVec = DVDict["x_αb_strut"]
        strutebVec = 0.25 * strutchordVec .+ strutabVec
    elseif appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
        strutchordVec = nothing
        strutabVec = nothing
        strutx_αbVec = nothing
        strutebVec = nothing
    else
        error("Unsupported config: ", appendageOptions["config"])
    end
    Λ = DVDict["Λ"]
    α₀ = DVDict["α₀"]
    structMesh = FEMESH.mesh
    elemConn = FEMESH.elemConn
    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive; config=appendageOptions["config"], STRUT=STRUT, ab_strut=strutabVec, x_αb_strut=strutx_αbVec)

    # --- Initialize states ---
    u = zeros(length(globalF))

    # if solverOptions["debug"]
    #     if elemType == "COMP2"
    #         # Get transformation matrix for the tip load
    #         angleDefault = deg2rad(-90) # default angle of rotation of the axes to match beam
    #     else
    #         angleDefault = 0.0
    #     end
    #     axisDefault = "z"
    #     T1 = SolverRoutines.get_rotate3dMat(angleDefault, axis=axisDefault)
    #     T = T1
    #     Z = zeros(3, 3)
    #     transMatL2G = [
    #         T Z Z Z Z Z
    #         Z T Z Z Z Z
    #         Z Z T Z Z Z
    #         Z Z Z T Z Z
    #         Z Z Z Z T Z
    #         Z Z Z Z Z T
    #     ]
    #     FEMMethods.apply_tip_load!(globalF, elemType, transMatL2G, loadType; solverOptions=solverOptions)
    #     global globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped"; solverOptions=solverOptions)
    #     K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    #     # # --- Debug printout of matrices in human readable form after BC application ---
    #     # writedlm(outputDir * "K.csv", K,",")
    #     # writedlm(outputDir * "M.csv", M,",")

    #     # ---------------------------
    #     #   Pre-solve system
    #     # ---------------------------
    #     q = FEMMethods.solve_structure(K, M, F)
    #     # --- Populate displacement vector ---
    #     u[globalDOFBlankingList] .= 0.0
    #     idxNotBlanked = [x for x ∈ eachindex(u) if x ∉ globalDOFBlankingList] # list comprehension
    #     u[idxNotBlanked] .= q
    # end
    # ---------------------------
    #   Get initial fluid tracts
    # ---------------------------

    # fTractions, AIC, planformArea = HydroStrip.compute_steady_hydroLoads(u, structMesh, α₀, chordVec, abVec, ebVec, Λ, FOIL, elemType)
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(globalM)[1], structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; appendageOptions=appendageOptions, STRUT=STRUT, strutchordVec=strutchordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fTractions, _, _ = HydroStrip.integrate_hydroLoads(u, AIC, α₀, elemType, appendageOptions["config"]; appendageOptions=appendageOptions, solverOptions=solverOptions)
    globalF = fTractions
    global AICnoBC = AIC

    # # --- Debug printout of matrices in human readable form ---
    # println("Global stiffness matrix:")
    # println("------------------------")
    # show(stdout, "text/plain", globalK)
    # println("")
    # println("Global mass matrix:")
    # println("-------------------")
    # show(stdout, "text/plain", globalM)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    ChainRulesCore.ignore_derivatives() do
        global globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped"; appendageOptions=appendageOptions)
    end
    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    # # --- Debug printout of matrices in human readable form after BC application ---
    # writedlm(outputDir * "K.csv", K,",")
    # writedlm(outputDir * "M.csv", M,",")

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(K, M, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ eachindex(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q

    # hydrolift = fTractions[3:NDOF:end]
    # println("hydro lift on half wing", hydrolift[1:solverOptions["nNodes"]])
    # println("hydro moments", fTractions[5:NDOF:end])
    # println("w deflections", u[3:NDOF:end])

    # ************************************************
    #     CONVERGE r(u) = 0
    # ************************************************
    # --- Assign constants accessible in this module ---
    # This is needed for derivatives!
    derivMode = "RAD"
    global CONSTANTS = SolutionConstants.DCFoilConstants(K, zeros(2, 2), zeros(2, 2), elemType, structMesh, AIC, derivMode, planformArea)

    # Actual solve
    qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q)
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType; appendageOptions=appendageOptions)

    # --- Get hydroLoads again on solution ---
    # fHydro, AIC, _ = HydroStrip.compute_steady_hydroLoads(uSol, structMesh, FOIL, elemType)
    # fHydro, AIC, _ = HydroStrip.compute_steady_hydroLoads(uSol, structMesh, α₀, chordVec, abVec, ebVec, Λ, FOIL, elemType)
    _, _, _, AIC, _, planformArea = HydroStrip.compute_AICs(size(globalM)[1], structMesh, elemConn, Λ, chordVec, abVec, ebVec, FOIL, FOIL.U∞, 0.0, elemType; appendageOptions=appendageOptions, STRUT=STRUT, strutchordVec=strutchordVec, strutabVec=strutabVec, strutebVec=strutebVec)
    fHydro, _, _ = HydroStrip.integrate_hydroLoads(u, AIC, α₀, elemType, appendageOptions["config"]; appendageOptions=appendageOptions, solverOptions=solverOptions)
    # global Kf = AIC

    # ************************************************
    #     WRITE SOLUTION OUT TO FILES
    # ************************************************
    # # Also a quick static divergence check
    # if costFuncs["psitip"] * DVDict["θ"] > 0.0
    #     println("+---------------------------------------------------+")
    #     println("|  WARNING: STATIC DIVERGENCE CONDITION DETECTED!   |")
    #     println("|  PRODUCT OF FIBER ANGLE AND TIP TWIST ARE +VE     |")
    #     println("+---------------------------------------------------+")
    # end
    # println("Post converged")
    # println("hydro lift", fHydro[3:NDOF:end])
    # println("hydro moments", fHydro[5:NDOF:end])
    # println("w deflections", uSol[3:NDOF:end])

    write_sol(uSol, fHydro, elemType, outputDir)

    global STATSOL = DCFoilSolution.StaticSolution(uSol, fHydro)

    # ************************************************
    #     Get obj funcs
    # ************************************************
    # compute_CostFuncs(STATSOL, evalFuncs, solverOptions)

    return STATSOL
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

function postprocess_statics(states, forces)
    """
    TODO: make it so this is differentiable
    """

    if constants.elemType == "BT2"
        nDOF = 4
        theta = states[3:nDOF:end]
        Moments = forces[3:nDOF:end]
        W = states[1:nDOF:end]
        Lift = forces[1:nDOF:end]
    elseif constants.elemType == "COMP2"
        nDOF = 9
        theta = states[5:nDOF:end]
        Moments = forces[5:nDOF:end]
        W = states[3:nDOF:end]
        Lift = forces[3:nDOF:end]
    else
        error("Invalid element type")
    end

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************
    costFuncs = Dict() # initialize empty costFunc dictionary
    if "wtip" in evalFuncs
        w_tip = W[end]
        costFuncs["wtip"] = w_tip
    end
    if "psitip" in evalFuncs
        psi_tip = theta[end]
        costFuncs["psitip"] = psi_tip
    end
    if "lift" in evalFuncs
        TotalLift = sum(Lift)
        costFuncs["lift"] = TotalLift
    end
    if "moment" in evalFuncs
        TotalMoment = sum(Moments)
        costFuncs["moment"] = TotalMoment
    end
    if "cl" in evalFuncs
        CL = TotalLift / (0.5 * foil.ρ_f * foil.U∞^2 * constants.planformArea)
        costFuncs["cl"] = CL
    end
    if "cmy" in evalFuncs
        CM = TotalMoment / (0.5 * foil.ρ_f * foil.U∞^2 * constants.planformArea * mean(chordVec))
        costFuncs["cmy"] = CM
    end
    if "cd" in evalFuncs
        CD = 0.0
        costFuncs["cd"] = CD
    end

    return obj
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
function evalFuncs(states, forces, evalFuncs; constants=CONSTANTS, foil=FOIL, chordVec=chordVec, DVDict=gDVDict)
    """
    Given {u} and the forces, compute the cost functions
    """

    if constants.elemType == "BT2"
        nDOF = 4
        theta = states[3:nDOF:end]
        Moments = forces[3:nDOF:end]
        W = states[1:nDOF:end]
        Lift = forces[1:nDOF:end]
    elseif constants.elemType == "COMP2"
        nDOF = 9
        theta = states[5:nDOF:end]
        Moments = forces[5:nDOF:end]
        W = states[3:nDOF:end]
        Lift = forces[3:nDOF:end]
    else
        error("Invalid element type")
    end

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************
    costFuncs = Dict() # initialize empty costFunc dictionary
    if globappendageOptions["config"] == "wing"
        ADIM = constants.planformArea
    elseif globappendageOptions["config"] == "t-foil" || globappendageOptions["config"] == "full-wing"
        ADIM = 2 * constants.planformArea
    end
    qdyn = 0.5 * foil.ρ_f * foil.U∞^2
    if "wtip" in evalFuncs
        w_tip = W[globappendageOptions["nNodes"]]
        costFuncs["wtip"] = w_tip
    end
    if "psitip" in evalFuncs
        psi_tip = theta[globappendageOptions["nNodes"]]
        costFuncs["psitip"] = psi_tip
    end
    if "lift" in evalFuncs
        TotalLift = sum(Lift)
        costFuncs["lift"] = TotalLift
    end
    if "moment" in evalFuncs
        TotalMoment = sum(Moments)
        costFuncs["moment"] = TotalMoment
    end
    if "cl" in evalFuncs
        CL = TotalLift / (qdyn * ADIM)
        costFuncs["cl"] = CL
    end
    if "cmy" in evalFuncs
        CM = TotalMoment / (qdyn * ADIM * mean(chordVec))
        costFuncs["cmy"] = CM
    end
    if "cd" in evalFuncs
        CD = 0.0
        costFuncs["cd"] = CD
    end
    if "cdi" in evalFuncs || "fxi" in evalFuncs
        twist = theta[1:globappendageOptions["nNodes"]]

        clalpha, Fxi, CDi = HydroStrip.compute_glauert_circ(DVDict["s"], chordVec, deg2rad(DVDict["α₀"]), globsolverOptions["U∞"], globappendageOptions["nNodes"];
            h=DVDict["s_strut"],
            useFS=globsolverOptions["use_freeSurface"],
            rho=globsolverOptions["ρ_f"],
            twist=twist,
            debug=globsolverOptions["debug"],
            solverOptions=globsolverOptions, # TODO: this should probably happen on the solve mode too
        )

        if "cdi" in evalFuncs
            costFuncs["cdi"] = CDi
        end
        if "fxi" in evalFuncs
            costFuncs["fxi"] = Fxi
        end
    end
    # From Hörner Chapter 8
    if "cdj" in evalFuncs || "fxj" in evalFuncs
        tocbar = 0.5 * (DVDict["toc"][1] + DVDict["toc_strut"][1])
        CDt = 17 * (tocbar)^2 - 0.05
        dj = CDt * (qdyn * (tocbar * DVDict["c"][1])^2)
        CDj = dj / (qdyn * ADIM)
        costFuncs["cdj"] = CDj
        costFuncs["fxj"] = dj
    end
    if "cds" in evalFuncs || "fxs" in evalFuncs
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
    # TODO: PICKUP HERE GET COL AS AN OUTPUT!!
    # # TODO:
    # if "cdw" in evalFuncs || "fxw" in evalFuncs
    #     # rws = EQ 6.149 FALTINSEN thickness effect on wave resistance
    #     # rwgamma = EQ 6.145 FALTINSEN wave resistance due to lift
    # end
    if "cdpr" in evalFuncs || "fxpr" in evalFuncs
        if globappendageOptions["config"] == "wing" || globappendageOptions["config"] == "full-wing"
            WSA = 2 * ADIM # both sides
        elseif globappendageOptions["config"] == "t-foil"
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

    return costFuncs
end

function get_sol(DVDict, solverOptions)
    """
    Wrapper function to do primal solve and return solution struct
    """
    # Setup
    structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug =
        setup_solver(α₀, Λ, span, c, toc, ab, x_αb, g, θ, solverOptions)

    # Solve
    obj, SOL = solve(structMesh, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug)

    return SOL
end

function cost_funcs_with_derivs(α₀, Λ, span, c, toc, ab, x_αb, g, θ, solverOptions)
    """
    Do primal solve with function signature compatible with Zygote
    """
    # Setup
    structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug =
        setup_solver(α₀, Λ, span, c, toc, ab, x_αb, g, θ, solverOptions)

    # Solve
    obj = solve(structMesh, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug)

    return obj
end

function evalFuncsSens(DVDict::Dict, solverOptions::Dict; mode="FiDi")
    """
    Wrapper to compute total sensitivities
    """
    println("===================================================================================================")
    println("        STATIC SENSITIVITIES: ", mode)
    println("===================================================================================================")

    # Initialize output dictionary
    funcsSens = Dict()

    if mode == "FiDi" # use finite differences the stupid way

        @time sensitivities, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolveStatic.compute_costFuncs(x, solverOptions),
            DVDict)

        # --- Iterate over DVDict keys ---
        dv_ctr = 1 # counter for total DVs
        for (key_ctr, pair) in enumerate(DVDict)
            key = pair[1]
            val = pair[2]
            x_len = length(val)
            if x_len == 1
                funcsSens[key] = sensitivities[dv_ctr]
                dv_ctr += 1
            elseif x_len > 1
                funcsSens[key] = sensitivities[dv_ctr:(dv_ctr+x_len-1)]
                dv_ctr += x_len
            end
        end

    elseif mode == "RAD" # use automatic differentiation via Zygote

        @time sensitivities = Zygote.gradient((x1, x2, x3, x4, x5, x6, x7, x8, x9) ->
                cost_funcs_with_derivs(x1, x2, x3, x4, x5, x6, x7, x8, x9, solverOptions),
            DVDict["α₀"], DVDict["Λ"], DVDict["s"], DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_αb"], DVDict["zeta"], DVDict["θ"])
        # --- Order the sensitivities properly ---
        funcsSens["α₀"] = sensitivities[1]
        funcsSens["Λ"] = sensitivities[2]
        funcsSens["s"] = sensitivities[3]
        funcsSens["c"] = sensitivities[4]
        funcsSens["toc"] = sensitivities[5]
        funcsSens["ab"] = sensitivities[6]
        funcsSens["x_αb"] = sensitivities[7]
        funcsSens["zeta"] = sensitivities[8]
        funcsSens["θ"] = sensitivities[9]

    elseif mode == "FAD"
        error("FAD not implemented yet")
    end

    # ************************************************
    #     Sort the sensitivities by key (alphabetical)
    # ************************************************
    sorted_keys = sort(collect(keys(funcsSens)))
    sorted_dict = Dict()
    for key in sorted_keys
        sorted_dict[key] = funcsSens[key]
    end
    funcsSens = sorted_dict

    return funcsSens
end

function compute_∂f∂x(foilPDESol)

end

function compute_∂r∂x(foilPDESol)

end

function compute_∂f∂u(foilPDESol)

end

function compute_∂r∂u(structuralStates, mode="FiDi")
    """
    Jacobian of residuals with respect to structural states
    EXCLUDING BC NODES
    """

    if mode == "FiDi" # Finite difference
        # First derivative using 3 stencil points
        ∂r∂u = FiniteDifferences.jacobian(central_fdm(3, 1), compute_residuals, structuralStates)

    elseif mode == "RAD" # Reverse automatic differentiation
        # This is a tuple
        ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

        # elseif mode == "RAD" # Reverse automatic differentiation
        #     @time ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)

    elseif mode == "analytic"
        # In the case of a linear elastic beam under static fluid loading, 
        # dr/du = Ks + Kf
        # NOTE Kf = AIC matrix
        # where AIC * states = forces on RHS (external)
        # TODO: longer term, the AIC is influenced by the structural states b/c of the twist distribution
        ∂r∂u = CONSTANTS.Kmat + CONSTANTS.AICmat[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]

        # The behavior of the analytic derivatives is interesting since it takes about 6 NL iterations to 
        # converge to the same solution as the RAD, which only takes 2 NL iterations.
    else
        error("Invalid mode")
    end

    return ∂r∂u
end

function compute_residuals(structuralStates)
    """
    Compute residual for every node that is not the clamped root node

    r(u) = [K]{u} - {f(u)}

    where f(u) is the force vector from the current solution

    Inputs
    ------
    structuralStates : array
        State vector with nodal DOFs and deformations EXCLUDING BCs
    """

    # NOTE THAT WE ONLY DO THIS CALL HERE
    if CONSTANTS.elemType == "bend-twist" # knock off the root element
        exit()
    elseif CONSTANTS.elemType == "BT2" # knock off root element
        completeStates, _ = FEMMethods.put_BC_back(structuralStates, CONSTANTS.elemType)
        foilTotalStates, nDOF = SolverRoutines.return_totalStates(completeStates, FOIL.α₀, CONSTANTS.elemType)
        F = -CONSTANTS.AICmat * foilTotalStates
        FOut = F[5:end]
    elseif CONSTANTS.elemType == "COMP2"
        completeStates, _ = FEMMethods.put_BC_back(structuralStates, CONSTANTS.elemType; appendageOptions=globappendageOptions)
        foilTotalStates, nDOF = SolverRoutines.return_totalStates(completeStates, FOIL.α₀, CONSTANTS.elemType; STRUT=STRUT, appendageOptions=globappendageOptions)
        F = -CONSTANTS.AICmat * foilTotalStates
        if globappendageOptions["config"] == "t-foil"
            FOut = F[1:end-NDOF]
        elseif globappendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
            FOut = F[NDOF+1:end]
        else
            error("Unsupported config: ", globappendageOptions["config"])
        end
    else
        error("Invalid element type")
        println(CONSTANTS.elemType)
    end


    # --- Stack them ---
    resVec = CONSTANTS.Kmat * structuralStates - FOut

    return resVec
end

function compute_direct()
    """
    Computes direct vector
    """
end

function compute_adjoint()

end

function compute_jacobian(stateVec)
    """
    Compute the jacobian df/dx

    Inputs:
        stateVec:

    returns:


    """
    # ************************************************
    #     Compute cost func gradients
    # ************************************************


end


end # end module