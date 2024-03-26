# --- Julia ---
"""
@File    :   DCFoil.jl
@Time    :   2022/07/23
@Author  :   Galen Ng
@Desc    :   This is a required "gluing" file so the package 
             management works properly. This is the module imported
"""
module DCFoil

# --- Libraries ---
using JSON
using Printf

include("./solvers/SolveStatic.jl")
using .SolveStatic

include("./solvers/SolveForced.jl")
using .SolveForced

include("./solvers/SolveFlutter.jl")
using .SolveFlutter

include("./io/TecplotIO.jl")
using .TecplotIO

include("./InitModel.jl")
using .InitModel

include("./struct/FEMMethods.jl")
using .FEMMethods

# --- Public functions ---
# Don't need 'DCFoil.' prefix when importing
export init_model, run_model, evalFuncs, evalFuncsSens

# ==============================================================================
#                         API functions
# ==============================================================================
function init_model(DVDictList::Vector, evalFuncs; solverOptions)
    """
    Things that need to be done before running the model.
    Global vars are used in run_model

    TODOS: make it so it returns the init model struct (better memory handling)
    """
    # ---------------------------
    #   Default options
    # ---------------------------
    set_defaultOptions!(solverOptions)
    global outputDir = solverOptions["outputDir"]

    # ---------------------------
    #   Write DVs and options
    # ---------------------------
    for iComp in 1:length(DVDictList)
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
    for iComp in 1:length(DVDictList)
        DVDict = DVDictList[iComp]
        FOIL, STRUT = InitModel.init_model_wrapper(DVDict, solverOptions)
        push!(FOILList, FOIL)
        push!(STRUTList, STRUT)
    end

    # nElemWing = FOIL.nNodes - 1
    # if solverOptions["config"] == "wing" || solverOptions["config"] == "full-wing"
    #     nElStrut = 0
    # elseif solverOptions["config"] == "t-foil"
    #     nElStrut = STRUT.nNodes - 1
    # else
    #     error("Unsupported config: ", solverOptions["config"])
    # end
    structMeshList, elemConnList = FEMMethods.make_fullMesh(DVDictList, solverOptions)
    # global structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"];
    #     config=solverOptions["config"],
    #     nElStrut=nElStrut,
    #     spanStrut=DVDict["s_strut"],
    #     rotation=solverOptions["rotation"]
    # )
    global FEMESHLIST = []
    for iComp in 1:length(structMeshList)
        # global structMesh = structMeshList[icomp]
        # global elemConn = elemConnList[icomp]
        DVDict = DVDictList[iComp]
        FEMESH = FEMMethods.FEMESH(structMeshList[iComp], elemConnList[iComp], DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_αb"], DVDict["θ"], zeros(10, 2))
        push!(FEMESHLIST, FEMESH)
    end

    # --- Write mesh to tecplot for later visualization ---
    for iComp in 1:length(structMeshList)
        DVDict = DVDictList[iComp]
        TecplotIO.write_mesh(DVDict, FEMESHLIST, solverOptions, outputDir, @sprintf("mesh_comp%03d.dat", iComp))
    end
    if solverOptions["debug"]
        for iComp in 1:length(structMeshList)
            elemConn = elemConnList[iComp]
            open(outputDir * @sprintf("elemConn-comp%03d.txt", iComp), "w") do io
                for iElem in 1:length(elemConn[:, 1])
                    write(io, @sprintf("%03d\t%03d\n", elemConn[iElem, 1], elemConn[iElem, 2]))
                end
            end
        end
    end
end

function run_model(DVDictList::Vector, evalFuncs::Vector{String}; solverOptions=Dict())
    """
    Runs the model but does not return anything.
    The solution structures hang around as global variables.

    TODOS: make it so it returns the solution struct
    """
    # ==============================================================================
    #                         Introduction
    # ==============================================================================
    println("+--------------------------------+")
    println("|  Running DCFoil with foils:    |")
    println("+--------------------------------+")
    for iComp in 1:length(DVDictList)
        appendageOptions = solverOptions["appendageList"][iComp]
        println(@sprintf("Component %03d: ", iComp), appendageOptions["compName"])
    end
    println("+--------------------------------+")

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        for iComp in 1:length(solverOptions["appendageList"])
            FEMESH = FEMESHLIST[iComp]
            appendageOptions = solverOptions["appendageList"][iComp]
            DVDict = DVDictList[iComp]
            println("Running static solve for component: ", appendageOptions["compName"])
            global STATSOL = SolveStatic.solve(FEMESH, DVDict, evalFuncs, solverOptions, appendageOptions)
            if solverOptions["writeTecplotSolution"]
                SolveStatic.write_tecplot(DVDict, STATSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
            end
        end
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
        global FLUTTERSOL = nothing
        if solverOptions["run_flutter"]
            @time global FLUTTERSOL = SolveFlutter.get_sol(DVDict, solverOptions)
            if solverOptions["writeTecplotSolution"]
                SolveFlutter.write_tecplot(DVDict, FLUTTERSOL, structMesh, outputDir; solverOptions=solverOptions)
            end
        end

        return FLUTTERSOL
    end
end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
# function evalFuncs(evalFuncs::Vector{String}, solverOptions=Dict())
function evalFuncs(FLUTTERSOL, evalFuncs::Vector{String}, solverOptions=Dict())
    """
    Common interface to compute cost functions

    Inputs
    ------
    evalFuncs : 1d array
        List of what cost functions to evaluate
    """
    x = 0.0 # dummy
    evalFuncsDict = Dict()

    # --- Solver cost funcs ---
    staticCostFuncs = [
        "psitip"
        "wtip"
        "lift"
        "moment"
        "cl"
        "cmy"
        # Lift-induced drag
        "cdi"
        "fxi"
        # Junction drag (empirical relation interference)
        "cdj"
        "fxj"
        # Spray drag
        "cds"
        "fxs"
        # Profile drag
        "cdpr"
        "fxpr"
    ]
    forcedCostFuncs = [
        "peakpsitip" # maximum deformation amplitude (abs val) across forced frequency sweep
        "peakwtip"
        "vibareapsi" # integrated deformations under the spectral curve (see Ng et al. 2022)
        "vibareaw"
    ]
    flutterCostFuncs = [
        "ksflutter" # flutter value (damping)
        "lockin" # lock-in value
        "gap" # mode gap width
    ]

    # Assemble all possible
    allCostFuncs = vcat(staticCostFuncs, forcedCostFuncs, flutterCostFuncs)

    # ************************************************
    #     Loop over all evalFuncs
    # ************************************************
    for k in evalFuncs

        if k in staticCostFuncs
            staticEvalFuncs = SolveStatic.evalFuncs(STATSOL.structStates, STATSOL.fHydro, evalFuncs)
            evalFuncsDict[k] = staticEvalFuncs[k]

        elseif k in forcedCostFuncs
            SolveForced.evalFuncs()

        elseif k in flutterCostFuncs
            obj, _ = SolveFlutter.postprocess_damping(FLUTTERSOL.N_MAX_Q_ITER, FLUTTERSOL.flowHistory, FLUTTERSOL.NTotalModesFound, FLUTTERSOL.nFlow, FLUTTERSOL.p_r, FLUTTERSOL.iblank, solverOptions["rhoKS"])
            flutterCostFuncsDict = Dict(
                "ksflutter" => obj,
            )
            evalFuncsDict = merge(evalFuncsDict, flutterCostFuncsDict)
        else
            println("Unsupported cost function: ", k)
        end
    end

    # --- Write cost funcs to file ---
    outputDir = solverOptions["outputDir"]
    stringData = JSON.json(evalFuncsDict)
    open(outputDir * "funcs.json", "w") do io
        write(io, stringData)
    end

    return evalFuncsDict
end # evalFuncs

function evalFuncsSens(DVDict::Dict, evalFuncs::Vector{String}, solverOptions=Dict(); mode="FiDi")

    # # ---------------------------
    # #   Mesh generation
    # # ---------------------------
    # FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)
    # nElem = FOIL.nNodes - 1
    # structMesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"]; config=solverOptions["config"])

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSensDict = Dict()

    # ==============================================================================
    #                         Cost functions
    # ==============================================================================
    if solverOptions["run_flutter"]
        @time costFuncsSensDict = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode=mode)
    end

    return costFuncsSensDict
end

function set_defaultOptions!(solverOptions)
    """
    Set default options
    """
    # ************************************************
    #     I/O
    # ************************************************
    if !haskey(solverOptions, "name")
        solverOptions["name"] = "default"
    end
    if !haskey(solverOptions, "outputDir")
        solverOptions["outputDir"] = "./OUTPUT/"
    end
    if !haskey(solverOptions, "debug")
        solverOptions["debug"] = false
    end
    if !haskey(solverOptions, "writeTecplotSolution")
        solverOptions["writeTecplotSolution"] = false
    end
    # ************************************************
    #     General appendage options
    # ************************************************
    if !haskey(solverOptions, "config")
        solverOptions["config"] = "wing"
    end
    if !haskey(solverOptions, "material")
        solverOptions["material"] = "cfrp"
    end
    if !haskey(solverOptions, "strut_material")
        solverOptions["strut_material"] = "cfrp"
    end
    # ************************************************
    #     Flow
    # ************************************************
    if !haskey(solverOptions, "U∞")
        solverOptions["U∞"] = 1.0
    end
    if !haskey(solverOptions, "ρ_f")
        solverOptions["ρ_f"] = 1000.0
    end
    if !haskey(solverOptions, "use_freeSurface")
        solverOptions["use_freeSurface"] = false
    end
    if !haskey(solverOptions, "use_cavitation")
        solverOptions["use_cavitation"] = false
    end
    if !haskey(solverOptions, "use_ventilation")
        solverOptions["use_ventilation"] = false
    end
    # ************************************************
    #     Solver modes
    # ************************************************
    if !haskey(solverOptions, "run_static")
        solverOptions["run_static"] = false
    end
    if !haskey(solverOptions, "run_forced")
        solverOptions["run_forced"] = false
    end
    if !haskey(solverOptions, "run_modal")
        solverOptions["run_modal"] = false
    end
    if !haskey(solverOptions, "run_flutter")
        solverOptions["run_flutter"] = false
    end
    if !haskey(solverOptions, "rhoKS")
        solverOptions["rhoKS"] = 80.0
    end
    if !haskey(solverOptions, "maxQIter")
        solverOptions["maxQIter"] = 100
    end
    if !haskey(solverOptions, "fSweep")
        solverOptions["fSweep"] = [0.1, 10.0]
    end
    if !haskey(solverOptions, "tipForceMag")
        solverOptions["tipForceMag"] = 0.0
    end
    if !haskey(solverOptions, "nModes")
        solverOptions["nModes"] = 10
    end
    if !haskey(solverOptions, "uRange")
        solverOptions["uRange"] = [1.0, 2.0]
    end
    if !haskey(solverOptions, "name")
        solverOptions["name"] = "default"
    end
    if !haskey(solverOptions, "material")
        solverOptions["material"] = "cfrp"
    end
    if !haskey(solverOptions, "strut_material")
        solverOptions["strut_material"] = "cfrp"
    end
end

# precompile(init_model, (Dict, Vector{String},))
# precompile(run_model, (Dict, Vector{String},))
# precompile(evalFuncs, (Vector{String}, Dict,))
# precompile(evalFuncsSens, (Dict, Vector{String}, Dict,))

end # module