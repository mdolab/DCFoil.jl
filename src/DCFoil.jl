module DCFoil
"""
@File    :   DCFoil.jl
@Time    :   2022/07/23
@Author  :   Galen Ng
@Desc    :   This is a required "gluing" file so the package 
             management works properly. This is the module imported
"""

# ==============================================================================
#                         PACKAGES
# ==============================================================================
using JSON
using Printf
# using Debugger
# using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations

# Set the default data type
include("constants/DataTypes.jl")
const DTYPE = DblPrec # Cmplx

# ==============================================================================
#                         HEADER FILES
# ==============================================================================
for headerName in [
    # NOTE: THE ORDER MATTERS
    # --- MACH framework ---
    # "../dcfoil/mach",
    # --- Not used in this script but needed for submodules ---
    "constants/SolutionConstants", "constants/DesignConstants",
    "utils/Utilities", "utils/Interpolation",
    "struct/EBBeam",
    "CostFunctions", "DesignVariables",
    "struct/MaterialLibrary", "bodydynamics/HullLibrary",
    "hydro/Unsteady",
    "hydro/OceanWaves",
    "struct/BeamProperties",
    "solvers/NewtonRaphson",
    "solvers/EigenvalueProblem",
    "solvers/SolverRoutines",
    "utils/Preprocessing",
    "hydro/VPM", "hydro/LiftingLine", # General LL code
    "hydro/GlauertLL", # Glauert LL code
    "adrules/CustomRules",
    # --- Used in this script ---
    "struct/FEMMethods",
    "hydro/HydroStrip",
    "ComputeFunctions",
    "InitModel",
    "solvers/DCFoilSolution",
    "io/TecplotIO", "io/MeshIO",
    "solvers/SolveStatic", "solvers/SolveForced", "solvers/SolveFlutter",
    # include("./solvers/SolveBodyDynamics.jl")
]
    include(headerName * ".jl")
end

using .SolveStatic: SolveStatic
using .SolveForced: SolveForced
using .SolveFlutter: SolveFlutter
# using .SolveBodyDynamics
using .TecplotIO: TecplotIO
using .InitModel: InitModel
using .FEMMethods: FEMMethods
using .CostFunctions: staticCostFuncs, forcedCostFuncs, flutterCostFuncs, allCostFuncs


# ==============================================================================
#                         API functions
# ==============================================================================
# --- Public functions ---
export init_model, run_model, evalFuncs, evalFuncsSens


function setup_model(DVDictList, evalFuncsList; solverOptions)
    """
    Code to make mesh and other things if you're not starting from coordinates
    Global vars are used in run_model
    """
    # ---------------------------
    #   Default options
    # ---------------------------
    set_defaultOptions!(solverOptions)
    outputDir = solverOptions["outputDir"]

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
    #   Component structs
    # ---------------------------
    global FOILList = []
    global STRUTList = []
    for iComp in eachindex(DVDictList)
        DVDict = DVDictList[iComp]
        appendageOptions = solverOptions["appendageList"][iComp]
        FOIL, STRUT, HULL = InitModel.init_modelFromDVDict(DVDict, solverOptions, appendageOptions)
        push!(FOILList, FOIL)
        push!(STRUTList, STRUT)
    end
    global HULL = HULL

    # ---------------------------
    #   Mesh generation/loading
    # ---------------------------
    if isnothing(solverOptions["gridFile"])
        structMeshList, elemConnList = FEMMethods.make_fullMesh(DVDictList, solverOptions)
        global FEMESHList = []
        for iComp in eachindex(structMeshList)
            DVDict = DVDictList[iComp]
            structMesh = structMeshList[iComp]
            elemConn = elemConnList[iComp]
            FEMESH = FEMMethods.StructMesh(structMesh, elemConn, DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_ab"], DVDict["theta_f"], zeros(10, 2))

            push!(FEMESHList, FEMESH)
        end

    else
        GridStruct = MeshIO.add_mesh(solverOptions["gridFile"])
        LECoords, nodeConn, TECoords = GridStruct.LECoords, GridStruct.nodeConn, GridStruct.TECoords
        return LECoords, nodeConn, TECoords
    end

end

function init_model(LECoords, nodeConn, TECoords; solverOptions, appendageParamsList)
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
    #   Write options
    # ---------------------------
    stringData = JSON.json(solverOptions)
    open(outputDir * "solverOptions.json", "w") do io
        write(io, stringData)
    end

    # ---------------------------
    #   Write DVs and options
    # ---------------------------
    for iComp in eachindex(appendageParamsList)
        appendageParams = appendageParamsList[iComp]
        stringData = JSON.json(appendageParams)
        open(outputDir * @sprintf("init_DVDict-comp%03d.json", iComp), "w") do io
            write(io, stringData)
        end
    end

    # ---------------------------
    #   Component structs
    # ---------------------------
    global FOILList = []
    global STRUTList = []
    if !isnothing(solverOptions["gridFile"])
        global FEMESHList = []
    end

    for iComp in eachindex(appendageParamsList)
        DVDict = appendageParamsList[iComp]
        appendageOptions = solverOptions["appendageList"][iComp]
        if isnothing(solverOptions["gridFile"])
            FOIL, STRUT, HULL = InitModel.init_modelFromDVDict(DVDict, solverOptions, appendageOptions)
        else
            FOIL, STRUT, HULL, FEMESH, _, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, DVDict, solverOptions, appendageOptions)
            push!(FEMESHList, FEMESH)

            # println("mesh here\n", FEMESH.mesh)

            TecplotIO.write_hydromesh(LLSystem, FlowCond.uvec, outputDir)
        end

        push!(FOILList, FOIL)
        push!(STRUTList, STRUT)

    end

    # --- Write mesh to tecplot for later visualization ---
    TecplotIO.write_structmesh(FEMESHList, solverOptions, outputDir)

    global HULL = HULL

    # ************************************************
    #     Finally, print out a warning
    # ************************************************
    println("+-----------------------------------------------------------+")
    println("| WARNING: DCFOIL MAKE SURE UPSTREAM FOILS ARE LISTED FIRST |")
    println("+-----------------------------------------------------------+")
end

function run_model(LECoords, nodeConn, TECoords, evalFuncsList; solverOptions=Dict(), appendageParamsList)
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
    for iComp in eachindex(appendageParamsList)
        appendageOptions = solverOptions["appendageList"][iComp]
        println(@sprintf("| Component %03d: ", iComp), appendageOptions["compName"])
    end
    println("+-------------------------------------+")

    SOLDICT = Dict()
    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        STATSOLLIST = []
        CLMain::DTYPE = 0.0
        for iComp in eachindex(solverOptions["appendageList"])
            appendageOptions = solverOptions["appendageList"][iComp]
            # Maybe look at mounting location here instead of using the 'iComp'
            println("+--------------------------------------------------------+")
            println("|  Running static solve for ", appendageOptions["compName"])
            println("+--------------------------------------------------------+")

            FEMESH = FEMESHList[iComp]
            DVDict = appendageParamsList[iComp]

            # STATSOL = SolveStatic.solve(FEMESH, DVDict, solverOptions, appendageOptions)
            if isnothing(solverOptions["gridFile"])
                STATSOL = SolveStatic.compute_solFromDVDict(appendageParamsList, solverOptions, evalFuncsList; iComp=iComp, CLMain=CLMain)
            else
                tStatic = @elapsed begin
                    STATSOL = SolveStatic.compute_solFromCoords(LECoords, nodeConn, TECoords, appendageParamsList, solverOptions)
                end
            end
            if solverOptions["writeTecplotSolution"]
                SolveStatic.write_tecplot(DVDict, STATSOL, FEMESH, outputDir; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp)
            end
            push!(STATSOLLIST, STATSOL)

            # --- Need to compute main hydrofoil CL ---
            if iComp == 1 && solverOptions["use_dwCorrection"] && !solverOptions["use_nlll"]
                DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
                CLMain = SolveStatic.evalFuncs("cl", STATSOL.structStates, STATSOL, DVVec, DVLengths;
                    appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=appendageParamsList, CLMain=0.0)
            end
        end
        SOLDICT["STATIC"] = STATSOLLIST
    end

    if length(solverOptions["appendageList"]) > 1
        println("Dynamic solver does not work for multiple components")
    elseif length(solverOptions["appendageList"]) == 1
        FEMESH = FEMESHList[1]
        appendageOptions = solverOptions["appendageList"][1]
        # ==============================================================================
        #                         Forced vibration solution
        # ==============================================================================
        if solverOptions["run_forced"]
            # @time global forcedCostFuncs = SolveForced.solve(FEMESH, DVDict, solverOptions, appendageOptions)
            @time VIBSOL = SolveForced.solveFromCoords(LECoords, TECoords, nodeConn, DVDict, solverOptions, appendageOptions)
            SOLDICT["FORCED"] = VIBSOL
        end

        # ==============================================================================
        #                         Flutter solution
        # ==============================================================================
        if solverOptions["run_modal"]
            appendageParams = appendageParamsList[1]
            # structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes = SolveFlutter.solve_frequencies(FEMESH, DVDict, solverOptions, appendageOptions)
            structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes = SolveFlutter.solve_frequencies(LECoords, TECoords, nodeConn, FEMESH, appendageParams, solverOptions, appendageOptions)
            if solverOptions["writeTecplotSolution"]
                SolveFlutter.write_tecplot_natural(appendageParams, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, FEMESH.mesh, FEMESH.chord, outputDir; solverOptions=solverOptions)
            end
        end

        if solverOptions["run_flutter"]
            appendageParams = appendageParamsList[1]
            if isnothing(solverOptions["gridFile"])
                @time global FLUTTERSOL = SolveFlutter.get_sol(appendageParams, solverOptions)
            else
                @time FLUTTERSOL = SolveFlutter.compute_solFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions)
            end
            if solverOptions["writeTecplotSolution"]
                SolveFlutter.write_tecplot(appendageParams, FLUTTERSOL, FEMESH.chord, FEMESH.mesh, outputDir; solverOptions=solverOptions)
            end
            SOLDICT["FLUTTER"] = FLUTTERSOL
        end
    end

    # # ==============================================================================
    # #                         Body solver
    # # ==============================================================================
    # # if true, the above solver modes still matter as to which analyses happen!
    # if solverOptions["run_body"]
    #    BODYSTATSOL, APPENDAGESTATSOLLIST = SolveBodyDynamics.solve_trim(DVDictList, FEMESHList, HULL, solverOptions, 2)
    # end

    return SOLDICT
end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function evalFuncs(SOLDICT, LECoords, nodeConn, TECoords, DVDictList, evalFuncsList, solverOptions=Dict())
    """
    Common interface to compute cost functions

    Inputs
    ------
    evalFuncs : 1d array
        List of what cost functions to evaluate
    """
    x = 0.0 # dummy
    evalFuncsDict = Dict()
    # LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
    ptVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)

    if solverOptions["run_static"]
        # Get evalFuncs that are in the staticCostFuncs list
        staticEvalFuncs = [key for key in evalFuncsList if key in staticCostFuncs]
        STATSOLLIST = SOLDICT["STATIC"]
        CLMain::DTYPE = 0.0
        for iComp in eachindex(solverOptions["appendageList"])
            appendageOptions = solverOptions["appendageList"][iComp]
            STATSOL = STATSOLLIST[iComp]
            DVDict = DVDictList[iComp]
            compName = appendageOptions["compName"]

            # DVVec, DVLengths = Utilities.unpack_dvdict(DVDict)
            # staticFuncs = SolveStatic.evalFuncs(staticEvalFuncs, STATSOL.structStates, STATSOL, DVVec, DVLengths; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain)
            for evalFunc in staticEvalFuncs

                staticFunc = SolveStatic.get_evalFunc(evalFunc, STATSOL.structStates, STATSOL, ptVec, nodeConn, DVDict; appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, DVDictList=DVDictList, CLMain=CLMain)

                newKey = @sprintf("%s-%s", evalFunc, compName)
                evalFuncsDict[newKey] = staticFunc
            end


            # --- Need to compute main hydrofoil CL ---
            if iComp == 1 && solverOptions["use_dwCorrection"] && !solverOptions["use_nlll"]
                try
                    CLMain = staticFuncs["cl"]
                catch
                    # warning
                    println("+" * "-"^80 * "+")
                    println("CL not found in staticFuncs! Cannot compute dw correction")
                    println("+" * "-"^80 * "+")
                end
            end
        end

    elseif solverOptions["run_forced"]
        forcedEvalFuncs = [key for key in evalFuncsList if key in forcedCostFuncs]
        SolveForced.evalFuncs()

    elseif solverOptions["run_flutter"]
        flutterEvalFuncs = [key for key in evalFuncsList if key in flutterCostFuncs]
        obj, _ = SolveFlutter.postprocess_damping(FLUTTERSOL.N_MAX_Q_ITER, FLUTTERSOL.flowHistory, FLUTTERSOL.NTotalModesFound, FLUTTERSOL.nFlow, FLUTTERSOL.p_r, FLUTTERSOL.iblank, solverOptions["rhoKS"])
        for evalFunc in flutterEvalFuncs
            evalFuncsDict[evalFunc] = obj
        end
    else
        println("Unsupported cost function: $(key) or solver mode not on")
    end

    # --- Write cost funcs to file ---
    outputDir = solverOptions["outputDir"]
    stringData = JSON.json(evalFuncsDict)
    open(outputDir * "funcs.json", "w") do io
        write(io, stringData)
    end

    return evalFuncsDict
end # evalFuncs


function evalFuncsSens(
    SOLDICT::Dict, appendageParamsList::Vector, LECoords, nodeConn, TECoords, evalFuncSensList, solverOptions=Dict();
    mode="FiDi", CLMain=0.0
)
    """
    Think of this as the gluing call for getting derivatives for a single cost function at a time

    """

    GridStruct = MeshIO.Grid(LECoords, nodeConn, TECoords)
    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSens = 0.0 # scope
    if solverOptions["run_flutter"]
        flutterFuncList = []
        for evalFuncSensKey in evalFuncSensList
            if evalFuncSensKey in flutterCostFuncs
                push!(flutterFuncList, evalFuncSensKey)
            end
        end
        if mode == "ADJOINT"
            flutterDerivMode = "RAD"
        else
            flutterDerivMode = "FiDi"
        end
        appendageParams = appendageParamsList[1]
        tFlutter = @elapsed begin
            costFuncsSens = SolveFlutter.evalFuncsSens(flutterFuncList, appendageParams, GridStruct, solverOptions; mode=flutterDerivMode)
        end
    end

    if solverOptions["run_static"]
        staticFuncList = []
        for evalFuncSens in evalFuncSensList
            if evalFuncSens in staticCostFuncs
                push!(staticFuncList, evalFuncSens)
            end
        end
        STATSOLLIST = SOLDICT["STATIC"]

        tStatic = @elapsed begin
            costFuncsSens = SolveStatic.evalFuncsSens(STATSOLLIST, staticFuncList, appendageParamsList, GridStruct, FEMESHList, solverOptions;
                mode=mode, CLMain=CLMain)
        end
        println("Static sensitivity time:\t$(tStatic) sec")
    end

    return costFuncsSens
end

function set_defaultOptions!(solverOptions)
    """
    Set default options
    """

    function check_key!(solverOptions::Dict, key::String, default)
        if !haskey(solverOptions, key)
            println("Setting default option: $(key) to ", default)
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
        "gridFile",
        # ************************************************
        #     Flow
        # ************************************************
        "Uinf",
        "rhof",
        "nu",
        "use_freeSurface",
        "use_cavitation",
        "use_ventilation",
        "use_dwCorrection",
        "use_nlll",
        # ************************************************
        #     Hull properties
        # ************************************************
        "hull",
        # ************************************************
        #     Solver modes
        # ************************************************
        "run_static",
        "res_jacobian",
        "onlyStructDerivs",
        "run_forced",
        "run_modal",
        "run_flutter",
        "run_body",
        "rhoKS",
        "maxQIter",
        "fRange",
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
        nothing,
        # ************************************************
        #     Flow
        # ************************************************
        1.0,
        1000.0,
        1.1892E-06, # kinematic viscosity of seawater at 15C
        false,
        false,
        false,
        false,
        false, # use_nlll
        # ************************************************
        #     Hull properties
        # ************************************************
        nothing,
        # ************************************************
        #     Solver modes
        # ************************************************
        false,
        "analytic", # residual jacobian
        false,
        false,
        false,
        false,
        false, # run_body
        80.0,
        100, # maxQIter
        [0.1, 10.0], # fRange
        0.0, # tipForceMag
        10, # nModes
        [1.0, 2.0] # uRange
    ]
    for ii in eachindex(keys)
        check_key!(solverOptions, keys[ii], defaults[ii])
    end
end

# ==============================================================================
#                         Precompile helpers
# ==============================================================================
# This workflow just allows the functions in the module to be precompiled for this type of argument
let # local binding scope
    SOLDICT = Dict
    DVDictList = Vector
    evalFuncs = Vector{String}
    solverOptions = Dict
    precompile(init_model, (DVDictList, evalFuncs))
    precompile(run_model, (DVDictList, evalFuncs))
    precompile(evalFuncs, (SOLDICT, DVDictList, evalFuncs, solverOptions))
    precompile(evalFuncsSens, (SOLDICT, DVDictList, evalFuncs, solverOptions))
end

end # module
