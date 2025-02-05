# --- Julia 1.11---
"""
@File          :   CoupledSolver.jl
@Date created  :   2025/01/27
@Last modified :   2025/01/27
@Author        :   Galen Ng
@Desc          :   This file contains the coupled solver algorithm for the static hydrodynamic and the static structural solvers
                   It's based on pyAerostructure
"""


module CoupledSolver
using ..LDTransfer

function do_GSIter(solve_hydro, solve_structure, structStates, hydroStates)

    # --- Transfer displacements ---
    deflections = LDTransfer.transfer_LD(structStates)

    # --- Solve hydrodynamics ---
    fLL, hydroStates_GS = solve_hydro(deflections)

    # --- Set forces ---
    fHydro = LDTransfer.transfer_LD(fLL)

    # --- Solve Structure ---
    structStates_GS = solve_structure(K, M, fHydro)

    return structStates_GS, hydroStates_GS
end

function convergenceCheck()

end

function converge_coupledRes()
    """

    """

    for ii in 1:maxIters
        do_GSIter()

        convergenceCheck()

    end

    return convergedStructStates, convergedHydroStates
end

end