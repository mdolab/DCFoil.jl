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

struct FlutterSolution{T<:Float64}
    eigs_r::Matrix{T} # dimensional eigvals
    eigs_i::Matrix{T} # dimensional eigvals
    R_eigs_r::Array{T, 3} # stacked eigenvectors
    R_eigs_i::Array{T, 3} # stacked eigenvectors
    NTotalModesFound::Int64
    N_MAX_Q_ITER::Int64
    flowHistory::Matrix{T}
    nFlow::Int64
    iblank::Matrix{Int64}
    p_r::Matrix{T} # nondim eigvals (for obj)
end

end