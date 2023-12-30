# --- Julia 1.7---
"""
@File    :   DCFoilSolution.jl
@Time    :   2023/03/11
@Author  :   Galen Ng
@Desc    :   After you've run a solution, you want to save the data in memory
because you'll need it for the costfunc and sensitivity calls
"""

module DCFoilSolution

struct StaticSolution{T<:Float64}
    structStates::Vector{T}
    fHydro::Vector{T}
end
struct FlutterSolution
    eigs_r # dimensional eigvals
    eigs_i # dimensional eigvals
    R_eigs_r # stacked eigenvectors
    R_eigs_i # stacked eigenvectors
    NTotalModesFound::Int64
    N_MAX_Q_ITER
    flowHistory
    nFlow::Int64
    iblank
    p_r # nondim eigvals (for obj)
end
end