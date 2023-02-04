# --- Julia ---

# @File    :   DCFoil.jl
# @Time    :   2022/07/23
# @Author  :   Galen Ng
# @Desc    :   This is a required "gluing" file so the package management works properly

module DCFoil


# --- Public functions ---
export run_model

# --- Libraries ---
include("./solvers/SolveStatic.jl")
include("./solvers/SolveForced.jl")
include("./solvers/SolveFlutter.jl")
using JSON
using .SolveStatic
using .SolveForced
using .SolveFlutter

function run_model(DVDict, evalFuncs; run_static=false, run_forced=false, run_modal=false, run_flutter=false, fSweep=nothing, tipForceMag=nothing, nModes=1, uSweep=nothing, fSearch=nothing, outputDir="./OUTPUT/", debug=false)
    """
    The interface into the src code
    """

    # --- Write the init dict to output folder ---
    stringData = JSON.json(DVDict)
    open(outputDir * "init_DVDict.json", "w") do io
        write(io, stringData)
    end

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if run_static
        SolveStatic.solve(DVDict, evalFuncs, outputDir)
    end

    # ==============================================================================
    #                         Forced vibration solution
    # ==============================================================================
    if run_forced
        SolveForced.solve(DVDict, outputDir, fSweep, tipForceMag)
    end

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if run_modal
        SolveFlutter.solve_frequencies(DVDict, nModes, outputDir)
    end
    if run_flutter
        SolveFlutter.solve(DVDict, outputDir, uSweep, fSearch, nModes; debug=debug)
    end

end

end