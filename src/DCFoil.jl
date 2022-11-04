# --- Julia ---

# @File    :   DCFoil.jl
# @Time    :   2022/07/23
# @Author  :   Galen Ng
# @Desc    :   This is a required "gluing" file so the package management works properly

module DCFoil


# --- Public functions ---
export run_model

# --- Libraries ---
include("./solvers/SolveSteady.jl")
include("./solvers/SolveDynamic.jl")
include("./solvers/SolveFlutter.jl")
using JSON
using .SolveSteady
using .SolveDynamic

function run_model(DVDict, evalFuncs, static=true, dynamic=true, fSweep=nothing, tipForceMag=nothing, outputDir="./OUTPUT/")
    """
    Here is the interface into the src code
    """
    # --- Write the init dict to output folder ---
    stringData = JSON.json(DVDict)
    open(outputDir * "init_DVDict.json", "w") do io
        write(io, stringData)
    end

    # ==============================================================================
    # Steady solution
    # ==============================================================================
    if static
        SolveSteady.solve(DVDict, evalFuncs, outputDir)
    end

    # ==============================================================================
    # Dynamic solution
    # ==============================================================================
    if dynamic
        SolveDynamic.solve(DVDict, outputDir, fSweep, tipForceMag)

        # TODO:
        # SolveFlutter.solve(DVDict, outputDir, fSearch)
    end
end

end