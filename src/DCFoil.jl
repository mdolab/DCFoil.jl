# --- Julia ---
"""
@File    :   DCFoil.jl
@Time    :   2022/07/23
@Author  :   Galen Ng
@Desc    :   This is a required "gluing" file so the package 
             management works properly. This is the module imported
"""
module DCFoil

# ==============================================================================
#                         PACKAGES
# ==============================================================================
using JSON
using Printf
# using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations

# ==============================================================================
#                         HEADER FILES
# ==============================================================================
# @recompile_invalidations begin # This should precompile all the solvers
for headerName in [
    # NOTE: THE ORDER MATTERS
    # --- Not used in this script but needed for submodules ---
    "constants/SolutionConstants", "constants/DesignConstants", "constants/DataTypes",
    "struct/MaterialLibrary", "bodydynamics/HullLibrary",
    "hydro/Unsteady",
    "struct/BeamProperties", "struct/EBBeam",
    "solvers/DCFoilSolution",
    "solvers/NewtonRaphson",
    "solvers/EigenvalueProblem",
    "solvers/SolverRoutines",
    "hydro/HydroStrip",
    "adrules/CustomRules",
    # --- Used in this script ---
    "InitModel", "struct/FEMMethods",
    "io/TecplotIO",
    "solvers/SolveStatic", "solvers/SolveForced", "solvers/SolveFlutter",
    # include("./solvers/SolveBodyDynamics.jl")
    "CostFunctions",
]

    include(headerName * ".jl")

end
# end
using .SolveStatic: SolveStatic
using .SolveForced: SolveForced
using .SolveFlutter: SolveFlutter
# using .SolveBodyDynamics
using .TecplotIO: TecplotIO
using .InitModel: InitModel
using .FEMMethods: FEMMethods
using .CostFunctions: staticCostFuncs, forcedCostFuncs, flutterCostFuncs
allCostFuncs = vcat(staticCostFuncs, forcedCostFuncs, flutterCostFuncs)

# ==============================================================================
#                         API functions
# ==============================================================================
# --- Public functions ---
# Don't need 'DCFoil.' prefix when importing
export init_model, run_model, evalFuncs, evalFuncsSens

function init_model(DVDictList::Vector, evalFuncs::Vector{String}; solverOptions)
    """
    Things that need to be done before running the model.
    Global vars are used in run_model
    """
    # ---------------------------
    #   Default options
    # ---------------------------
    set_defaultOptions!(solverOptions)
    global outputDir = solverOptions["outputDir"]

    # ---------------------------
    #   Write DVs and options
    # ---------------------------
    for iComp in eachindex(DVDictList)
        DVDict = DVDictList[iComp]
        stringData = JSON.json(DVDict)
        open(outputDir * @sprintf("init_DVDict-comp%03d.json", iComp), "w") do io
            write(io, stringData)
        end
    end
    stringData = JSON.json(solverOptions)
    open(outputDir * "solverOptions.json", "w") do io
        write(io, stringData)
    end

    # ---------------------------
    #   Cost functions
    # ---------------------------
    global costFuncsDict = Dict()

    # ---------------------------
    #   Mesh generation
    # ---------------------------
    global FOILList = []
    global STRUTList = []
    for iComp in eachindex(DVDictList)
        DVDict = DVDictList[iComp]
        appendageOptions = solverOptions["appendageList"][iComp]
        FOIL, STRUT, HULL = InitModel.init_model_wrapper(DVDict, solverOptions, appendageOptions)
        push!(FOILList, FOIL)
        push!(STRUTList, STRUT)
    end
    global HULL = HULL

    structMeshList, elemConnList = FEMMethods.make_fullMesh(DVDictList, solverOptions)
    global FEMESHLIST = []
    for iComp in eachindex(structMeshList)
        DVDict = DVDictList[iComp]
        structMesh = structMeshList[iComp]
        elemConn = elemConnList[iComp]
        FEMESH = FEMMethods.FEMESH(structMesh, elemConn, DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_αb"], DVDict["θ"], zeros(10, 2))

        push!(FEMESHLIST, FEMESH)
    end

    # --- Write mesh to tecplot for later visualization ---
    for iComp in eachindex(structMeshList)
        DVDict = DVDictList[iComp]
        TecplotIO.write_mesh(DVDict, FEMESHLIST, solverOptions, outputDir, @sprintf("mesh_comp%03d.dat", iComp))
    end
    if solverOptions["debug"]
        for iComp in eachindex(structMeshList)
            elemConn = elemConnList[iComp]
            open(outputDir * @sprintf("elemConn-comp%03d.txt", iComp), "w") do io
                for iElem in eachindex(elemConn[:, 1])
                    write(io, @sprintf("%03d\t%03d\n", elemConn[iElem, 1], elemConn[iElem, 2]))
                end
            end
        end
    end

    # ************************************************
    #     Finally, print out a warning
    # ************************************************
    println("+-----------------------------------------------------------+")
    println("| WARNING: DCFOIL MAKE SURE UPSTREAM FOILS ARE LISTED FIRST |")
    println("+-----------------------------------------------------------+")
end

function run_model(DVDictList::Vector, evalFuncs::Vector{String}; solverOptions=Dict())
    """
    This call runs DCFoil in three possible solver modes.
    Output is then a dictionary with possibly three keys at most.

    Parameters
    ----------
    DVDictList : array
        list of dictionaries for the appendages
    evalFuncs : array
        list of cost functions to evaluate
    solverOptions : dict
        dictionary of solver options

    Returns
    -------
    SOLDICT : dict
        dictionary of solutions
    """
    # ==============================================================================
    #                         Introduction
    # ==============================================================================
    println("+-------------------------------------+")
    println("|  Running DCFoil with appendages:    |")
    println("+-------------------------------------+")
    for iComp in eachindex(DVDictList)
        appendageOptions = solverOptions["appendageList"][iComp]
        println(@sprintf("| Component %03d: ", iComp), appendageOptions["compName"])
    end
    println("+-------------------------------------+")

    SOLDICT = Dict()
    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"] && !(solverOptions["run_body"])
        STATSOLLIST = []
        for iComp in eachindex(solverOptions["appendageList"])
            appendageOptions = solverOptions["appendageList"][iComp]
            # Maybe look at mounting location here instead of using the 'iComp'
            println("+--------------------------------------------------------+")
            println("|  Running static solve for ", appendageOptions["compName"])
            println("+--------------------------------------------------------+")

            FEMESH = FEMESHLIST[iComp]
            DVDict = DVDictList[iComp]

            # STATSOL = SolveStatic.solve(FEMESH, DVDict, solverOptions, appendageOptions)
            STATSOL = SolveStatic.get_sol(DVDict, solverOptions, evalFuncs; iComp=iComp, FEMESH=FEMESH)
            if solverOptions["writeTecplotSolution"]
                SolveStatic.write_tecplot(DVDict, STATSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
            end
            push!(STATSOLLIST, STATSOL)
        end
        SOLDICT["STATIC"] = STATSOLLIST
    end

    if length(solverOptions["appendageList"]) > 1
        println("Dynamic solver does not work for multiple components")
    elseif length(solverOptions["appendageList"]) == 1
        FEMESH = FEMESHLIST[1]
        appendageOptions = solverOptions["appendageList"][1]
        # ==============================================================================
        #                         Forced vibration solution
        # ==============================================================================
        if solverOptions["run_forced"]
            @time global forcedCostFuncs = SolveForced.solve(FEMESH, DVDict, solverOptions, appendageOptions)
        end

        # ==============================================================================
        #                         Flutter solution
        # ==============================================================================
        if solverOptions["run_modal"]
            structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes = SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)
            if solverOptions["writeTecplotSolution"]
                SolveFlutter.write_tecplot_natural(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, structMesh, outputDir; solverOptions=solverOptions)
            end
        end

        if solverOptions["run_flutter"]
            @time global FLUTTERSOL = SolveFlutter.get_sol(DVDict, solverOptions)
            if solverOptions["writeTecplotSolution"]
                SolveFlutter.write_tecplot(DVDict, FLUTTERSOL, structMesh, outputDir; solverOptions=solverOptions)
            end
        end
        SOLDICT["FLUTTER"] = FLUTTERSOL
    end

    # # ==============================================================================
    # #                         Body solver
    # # ==============================================================================
    # # if true, the above solver modes still matter as to which analyses happen!
    # if solverOptions["run_body"]
    #    BODYSTATSOL, APPENDAGESTATSOLLIST = SolveBodyDynamics.solve_trim(DVDictList, FEMESHLIST, HULL, solverOptions, 2)
    # end

    return SOLDICT
end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function evalFuncs(SOLDICT::Dict, DVDictList::Vector, evalFuncs::Vector{String}, solverOptions=Dict())
    """
    Common interface to compute cost functions

    Inputs
    ------
    evalFuncs : 1d array
        List of what cost functions to evaluate
    """
    x = 0.0 # dummy
    evalFuncsDict = Dict()


    if solverOptions["run_static"]
        # Get evalFuncs that are in the staticCostFuncs list
        staticEvalFuncs = [key for key in evalFuncs if key in staticCostFuncs]        
        STATSOLLIST = SOLDICT["STATIC"]

        for iComp in eachindex(solverOptions["appendageList"])
            appendageOptions = solverOptions["appendageList"][iComp]
            STATSOL = STATSOLLIST[iComp]
            DVDict = DVDictList[iComp]
            compName = appendageOptions["compName"]
            staticFuncs = SolveStatic.evalFuncs(staticEvalFuncs, STATSOL.structStates, STATSOL, DVDict; appendageOptions=appendageOptions, solverOptions=solverOptions)
            for key in keys(staticFuncs)
                newKey = @sprintf("%s-%s", key, compName)
                evalFuncsDict[newKey] = staticFuncs[key]
            end
        end

    elseif key in forcedCostFuncs && solverOptions["run_forced"]
        SolveForced.evalFuncs()

    elseif key in flutterCostFuncs && solverOptions["run_flutter"]
        obj, _ = SolveFlutter.postprocess_damping(FLUTTERSOL.N_MAX_Q_ITER, FLUTTERSOL.flowHistory, FLUTTERSOL.NTotalModesFound, FLUTTERSOL.nFlow, FLUTTERSOL.p_r, FLUTTERSOL.iblank, solverOptions["rhoKS"])
        flutterCostFuncsDict = Dict(
            "ksflutter" => obj,
        )
        evalFuncsDict = merge(evalFuncsDict, flutterCostFuncsDict)
    else
        println("Unsupported cost function: ", key, "or solver mode not on")
    end

    # --- Write cost funcs to file ---
    outputDir = solverOptions["outputDir"]
    stringData = JSON.json(evalFuncsDict)
    open(outputDir * "funcs.json", "w") do io
        write(io, stringData)
    end

    return evalFuncsDict
end # evalFuncs

function evalFuncsSens(SOLDICT::Dict, DVDictList::Vector, evalFuncsSens::Vector{String}, solverOptions=Dict(); mode="FiDi")
    """
    Think of this as the gluing call for getting all derivatives

    """

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSensDict = Dict()

    if solverOptions["run_flutter"]
        @time costFuncsSensDict = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode=mode)
    end

    if solverOptions["run_static"]
        STATSOLLIST = SOLDICT["STATIC"]
        evalFuncsSensStat = [key for key in evalFuncsSens if key in staticCostFuncs]
        @time costFuncsSensList = SolveStatic.evalFuncsSens(STATSOLLIST, evalFuncsSensStat, DVDictList, FEMESHLIST, solverOptions; mode=mode)
        costFuncsSensDict = costFuncsSensList
    end

    return costFuncsSensDict
end

function set_defaultOptions!(solverOptions)
    """
    Set default options
    """

    function check_key!(solverOptions::Dict, key::String, default)
        if !haskey(solverOptions, key)
            println("Setting default option: ", key, " to ", default)
            solverOptions[key] = default
        end
    end
    keys = [
        # ************************************************
        #     I/O
        # ************************************************
        "name",
        "outputDir",
        "debug",
        "writeTecplotSolution",
        # ************************************************
        #     Flow
        # ************************************************
        "U∞",
        "ρ_f",
        "use_freeSurface",
        "use_cavitation",
        "use_ventilation",
        # ************************************************
        #     Hull properties
        # ************************************************
        "hull",
        # ************************************************
        #     Solver modes
        # ************************************************
        "run_static",
        "run_forced",
        "run_modal",
        "run_flutter",
        "rhoKS",
        "maxQIter",
        "fSweep",
        "tipForceMag",
        "nModes",
        "uRange",
    ]
    defaults = [
        # ************************************************
        #     I/O
        # ************************************************
        "default",
        "./OUTPUT/",
        false,
        false,
        # ************************************************
        #     Flow
        # ************************************************
        1.0,
        1000.0,
        false,
        false,
        false,
        # ************************************************
        #     Hull properties
        # ************************************************
        nothing,
        # ************************************************
        #     Solver modes
        # ************************************************
        false,
        false,
        false,
        false,
        80.0,
        100, # maxQIter
        [0.1, 10.0], # fSweep
        0.0, # tipForceMag
        10, # nModes
        [1.0, 2.0] # uRange
    ]
    for ii in eachindex(keys)
        check_key!(solverOptions, keys[ii], defaults[ii])
    end
end

# let
#     DVDictList = Vector
#     evalFuncs = Vector{String}
#     # precompile(init_model, (Dict, Vector{String},))
#     precompile(run_model, (DVDictList, evalFuncs))
#     # precompile(evalFuncs, (Vector{String}, Dict,))
#     # precompile(evalFuncsSens, (Dict, Vector{String}, Dict,))
# end

end # module