
# --- Julia ---

# @File    :   SolutionConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store used in the solution


module SolutionConstants

include("../hydro/Hydro.jl")
include("../struct/BeamProperties.jl")
using .Hydro, .StructProp


mutable struct DCFoilConstants{T<:Float64}
    """
    This is a catch all mutable struct to store variables that we do not 
    want in function calls like r(u) or f(u)

    """
    Kmat::Matrix{T}
    elemType::String
    mesh::LinRange{T,Int64}
    AICmat::Matrix{T} # Aero influence coeff matrix
    mode::String # type of derivative for drdu
    planformArea::T
end

mutable struct DCFoilDynamicConstants{T<:Float64}
    """
    For the dynamic hydroelastic solve, there are more constants to store
    """
    elemType::String
    mesh::LinRange{T,Int64}
    Dmat::Matrix{ComplexF64} # dynamic matrix # TODO: these might change
    AICmat::Matrix{ComplexF64} # just the aero part of Dmat # TODO: these might change
    extForceVec::Vector{T} # external force vector excluding BC nodes
end

end # end module