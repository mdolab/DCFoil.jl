# --- Julia ---

# @File    :   DesignConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store data specific to the hydrofoil model


module DesignConstants
using ..DCFoil: RealOrComplex

struct Foil{TF,TC,TI,TS,TA<:AbstractVector{TF},TB<:AbstractVector{TC}}
    """
    DO NOT STORE DVS HERE
    Foil object with key properties for the system solution
    Half-wing data
    """
    mₛ::TA # structural mass vector [kg/m]
    Iₛ::TA # structural moment of inertia vector [kg-m]
    EIₛ::TA # OOP bending stiffness vector [N-m²]
    EIIPₛ::TA # IP bending stiffness vector [N-m²]
    GJₛ::TA # torsion stiffness vector [N-m²]
    Kₛ::TA # bend-twist coupling vector [N-m²]
    Sₛ::TA # warping resistance vector [N-m⁴]
    EAₛ::TA # axial stiffness vector [N-m²]
    # U∞::TF # flow speed [m/s]
    # ζ::TC # modal damping ratio at first 2 modes
    # clα::Vector # lift slopes [1/rad]
    eb::TB # distance from center of pressure ahead of elastic axis [m]
    ab::TB # distance from midchord to EA, +ve for EA aft [m]
    chord::TB # chord vector [m]
    # ρ_f::TF # fluid density [kg/m³]
    nNodes::TI # number of evaluation points on span
    constitutive::TS # constitutive model
end

struct DynamicFoil{TF,TC,TI,TS,TA<:AbstractVector{TF},TB<:AbstractVector{TC}}
    """
    Dynamic foil object that inherits initially form the static foil mutable struct
    """
    mₛ::TA # structural mass vector [kg/m]
    Iₛ::TA # structural moment of inertia vector [kg-m]
    EIₛ::TA # OOP bending stiffness vector [N-m²]
    EIIPₛ::TA # IP bending stiffness vector [N-m²]
    GJₛ::TA # torsion stiffness vector [N-m²]
    Kₛ::TA # bend-twist coupling vector [N-m²]
    Sₛ::TA # warping resistance vector [N-m⁴]
    EAₛ::TA # axial stiffness vector [N-m²]
    # U∞::TF # flow speed [m/s]
    # ζ::TC # modal damping ratio at first 2 modes
    # clα::Vector # lift slopes [1/rad]
    eb::TB
    ab::TB
    chord::TB
    # ρ_f::TF # fluid density [kg/m³]
    nNodes::TI # number of evaluation points on span
    constitutive::TS # constitutive model
    # --- Only things different for the dynamic foil ---
    fRange::Vector # forcing frequency sweep [Hz] for harmonically forced solution AND search frequency for flutter
    uRange::Vector # forward speed sweep [m/s] (for flutter solution)
end

struct Hull{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    """
    Vessel object with key properties for the system solution
    This is a mutable struct, so it can be modified during the solution process
    DO NOT STORE DVS HERE
    """
    # --- Vessel properties ---
    mass::TF # mass of hull [kg]
    Ib::TM # BFS inertia matrix [kg-m²]
    xcg::TF # x-coordinate of center of gravity from the bow [m]
    loa::TF # length over all [m]
    beam::TF # beam [m]
end

# Store all possible DVs in a vector of strings (needed for wrestling into AD package formats)
const SORTEDDVS::Vector{String} = [
    "ab"
    "ab_strut"
    "beta"
    "c"
    "c_strut"
    "depth0"
    "rake"
    "s"
    "s_strut"
    "toc"
    "toc_strut"
    "x_ab"
    "x_ab_strut"
    "zeta"
    "sweep"
    "alfa0"
    "theta_f"
    "theta_f_strut"
]

const SORTEDSTRUCTDVS::Vector{String} = [
    "ab"
    "ab_strut"
    "beta"
    "c"
    "c_strut"
    "toc"
    "toc_strut"
    "x_ab"
    "x_ab_strut"
    "theta_f"
    "theta_f_strut"
]


# All possible configurations for a hydrofoil
const CONFIGS::Vector{String} =
    [
        "wing",
        "full-wing",
        "t-foil",
    ]

end # end module