
# --- Julia ---

# @File    :   SolutionConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store used in the solution


module SolutionConstants

include("../hydro/Hydro.jl")
include("../struct/BeamProperties.jl")
using .Hydro, .StructProp


mutable struct DCFoilConstants
    """
    This is a catch all mutable struct to store variables that we do not 
    want in function calls like r(u) or f(u)

    """
    Kmat
    elemType::String
    mesh
    AICmat # Aero influence coeff matrix
    mode::String # type of derivative for drdu
    planformArea
end

mutable struct DCFoilDynamicConstants
    """
    """
    elemType::String
    mesh
    Dmat # dynamic matrix
    AICmat # just the aero part of Dmat
    extForceVec # external force vector excluding BC nodes
end

end # end module