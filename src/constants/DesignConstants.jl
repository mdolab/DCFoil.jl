
# --- Julia ---

# @File    :   DesignConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store data specific to the hydrofoil model


module DesignConstants

include("../hydro/Hydro.jl")
include("../struct/BeamProperties.jl")
using .Hydro, .StructProp

mutable struct foil
    """
    Foil object with key properties for the system solution
    This is a mutable struct, so it can be modified during the solution process
    TODO: More design vars
    """
    c # chord length vector
    t # thickness vector
    s # semispan [m]
    ab # dist from midchord to EA vector (+ve for EA aft) [m]
    eb # dist from CP to EA (+ve for EA aft) [m]
    x_αb # static imbalance (+ve for CG aft) [m]
    mₛ # structural mass vector [kg/m]
    Iₛ # structural moment of inertia vector [kg-m]
    EIₛ # bending stiffness vector [N-m²]
    GJₛ # torsion stiffness vector [N-m²]
    Kₛ # bend-twist coupling vector [N-m²]
    Sₛ # warping resistance vector [N-m⁴]
    α₀ # rigid initial angle of attack [deg]
    U∞ # flow speed [m/s]
    Λ # sweep angle [rad]
    g # structural damping percentage
    clα # lift slopes [1/rad]
    ρ_f::Float64 # fluid density [kg/m³]
    neval::Int64 # number of evaluation points on span
    constitutive::String # constitutive model
end

mutable struct dynamicFoil
    """
    Dynamic foil object that inherits initially form the steady foil mutable struct
    """
    c # chord length vector
    t # thickness vector
    s # semispan [m]
    ab # dist from midchord to EA vector (+ve for EA aft) [m]
    eb # dist from CP to EA (+ve for EA aft) [m]
    x_αb # static imbalance (+ve for CG aft) [m]
    mₛ # structural mass vector [kg/m]
    Iₛ # structural moment of inertia vector [kg-m]
    EIₛ # bending stiffness vector [N-m²]
    GJₛ # torsion stiffness vector [N-m²]
    Kₛ # bend-twist coupling vector [N-m²]
    Sₛ # warping resistance vector [N-m⁴]
    α₀ # rigid initial angle of attack [deg]
    U∞ # flow speed [m/s]
    Λ # sweep angle [rad]
    g # structural damping percentage
    fSweep # forcing frequency sweep [Hz] (for harmonically forced solution)
    uSweep # forward speed sweep [m/s] (for flutter solution)
    clα # lift slopes [1/rad]
    ρ_f::Float64 # fluid density [kg/m³]
    neval::Int64 # number of evaluation points on span
    constitutive::String # constitutive model

end

end # end module