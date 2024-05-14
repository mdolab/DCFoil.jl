"""
@File    :   DCFoilSolution.jl
@Time    :   2023/03/11
@Author  :   Galen Ng
@Desc    :   After you've run a solution, you want to save the data in memory
because you'll need it for the costfunc and sensitivity calls
"""

module DCFoilSolution

using ..FEMMethods: StructMesh
using ..SolutionConstants: DCFoilSolverParams
using ..DesignConstants: Foil, DynamicFoil

struct StaticSolution{TF}
    structStates::Vector{TF} # state variables (u)
    fHydro::Vector{TF} # hydrodynamic forces
    FEMESH::StructMesh #struct type
    SOLVERPARAMS::DCFoilSolverParams
    FOIL::DynamicFoil#{TF,TI,TA}
    STRUT::DynamicFoil#{TF,TI,TA}
end

struct BodyStaticSolution{TF,TA<:AbstractVector{TF}}
    deltaC::TA # control inputs
end

struct FlutterSolution{TF,TI}
    eigs_r::Matrix{TF} # dimensional eigvals
    eigs_i::Matrix{TF} # dimensional eigvals
    R_eigs_r::Array{TF,3} # stacked eigenvectors
    R_eigs_i::Array{TF,3} # stacked eigenvectors
    NTotalModesFound::TI
    N_MAX_Q_ITER::TI
    flowHistory::Matrix{TF} # size(N_MAX_Q_ITER, 3) [velocity, density, dynamic pressure]
    nFlow::TI
    iblank::Matrix{TI}
    p_r::Matrix{TF} # nondim eigvals (for obj)(3*nModes, N_MAX_Q_ITER)
end

end