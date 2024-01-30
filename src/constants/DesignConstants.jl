# --- Julia ---

# @File    :   DesignConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store data specific to the hydrofoil model


module DesignConstants

include("../hydro/HydroStrip.jl")
include("../struct/BeamProperties.jl")
using .HydroStrip, .BeamProperties

struct foil{T<:Float64}
    """
    Foil object with key properties for the system solution
    This is a mutable struct, so it can be modified during the solution process
    DO NOT STORE DVS HERE
    Half-wing data
    """
    mₛ::Vector{T} # structural mass vector [kg/m]
    Iₛ::Vector{T} # structural moment of inertia vector [kg-m]
    EIₛ::Vector{T} # OOP bending stiffness vector [N-m²]
    EIIPₛ::Vector{T} # IP bending stiffness vector [N-m²]
    GJₛ::Vector{T} # torsion stiffness vector [N-m²]
    Kₛ::Vector{T} # bend-twist coupling vector [N-m²]
    Sₛ::Vector{T} # warping resistance vector [N-m⁴]
    EAₛ::Vector{T} # axial stiffness vector [N-m²]
    α₀::T # rigid initial angle of attack wrt flow (or yaw/sideslip angle) [deg] THE ONLY TIME THIS IS USED IS WHEN A DERIVATIVE WRT ALPHA IS NOT NEEDED
    U∞::T # flow speed [m/s]
    g::T # modal damping ratio at first 2 modes
    clα::Vector{T} # lift slopes [1/rad]
    ρ_f::T # fluid density [kg/m³]
    nNodes::Int64 # number of evaluation points on span
    constitutive::String # constitutive model
end

struct dynamicFoil{T<:Float64}
    """
    Dynamic foil object that inherits initially form the static foil mutable struct
    """
    mₛ::Vector{T} # structural mass vector [kg/m]
    Iₛ::Vector{T} # structural moment of inertia vector [kg-m]
    EIₛ::Vector{T} # OOP bending stiffness vector [N-m²]
    EIIPₛ::Vector{T} # IP bending stiffness vector [N-m²]
    GJₛ::Vector{T} # torsion stiffness vector [N-m²]
    Kₛ::Vector{T} # bend-twist coupling vector [N-m²]
    Sₛ::Vector{T} # warping resistance vector [N-m⁴]
    EAₛ::Vector{T} # axial stiffness vector [N-m²]
    α₀::T # rigid initial angle of attack [deg]
    U∞::T # flow speed [m/s]
    g::T # modal damping ratio at first 2 modes
    clα::Vector{T} # lift slopes [1/rad]
    ρ_f::T # fluid density [kg/m³]
    nNodes::Int64 # number of evaluation points on span
    constitutive::String # constitutive model
    # --- Only things different for the dynamic foil ---
    fSweep # forcing frequency sweep [Hz] for harmonically forced solution AND search frequency for flutter
    uRange::Vector{T} # forward speed sweep [m/s] (for flutter solution)
end

end # end module