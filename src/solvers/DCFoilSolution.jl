# --- Julia 1.7---
"""
@File    :   DCFoilSolution.jl
@Time    :   2023/03/11
@Author  :   Galen Ng
@Desc    :   After you've run a solution, you want to save the data in memory
because you'll need it for the costfunc and sensitivity calls
"""

module DCFoilSolution

struct StaticSolution
    structStates
    fHydro
end
struct FlutterSolution
    eigs_r # dimensional eigvals
    eigs_i # dimensional eigvals
    R_eigs_r
    R_eigs_i
    NTotalModesFound
    N_MAX_Q_ITER
    flowHistory
    nFlow::Int64
    iblank
end
end