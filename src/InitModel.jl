
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- Public functions ---
export init_steady

include("Hydro.jl")
include("Struct.jl")
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

mutable struct DCFoilConstants
  """
  This is a catch all mutable struct to store variables that we do not 
  want in function calls like r(u) or f(u)

    TODO: there's probably a better place to put this call
  """
  Kmat
  elemType::String
  mesh
  AICmat # Aero influence coeff matrix
  mode::String # type of derivative for drdu
  planformArea
end

function init_steady(neval::Int64, DVDict::Dict)
  """
  Initialize a steady hydrofoil model

  Inputs:
      neval: Int64, number of evaluation points on span
      DVDict: Dict, dictionary of model parameters, the design variables

  returns:
    foil: struct
  """

  # --- First print to screen in a box ---
  println("+", "-"^80, "+")
  println("This is your design variable setup...")
  for kv in DVDict
    println(kv)
  end
  println("+", "-"^80, "+")

  # ---------------------------
  #   Geometry
  # ---------------------------
  c = DVDict["c"]
  t = DVDict["toc"] * c
  ab = DVDict["ab"]
  eb = 0.25 * c + ab
  x_αb = DVDict["x_αb"]

  # ---------------------------
  #   Structure
  # ---------------------------
  # --- composite ---
  # TODO: Make a material property library
  if (DVDict["material"] == "cfrp")
    ρₛ = 1590.0
    E₁ = 117.8e9
    E₂ = 13.4e9
    G₁₂ = 3.9e9
    ν₁₂ = 0.25
    constitutive = "orthotropic"
  elseif (DVDict["material"] == "ss") # stainless-steel
    ρₛ = 7900
    E₁ = 193e9
    E₂ = 193e9
    G₁₂ = 77.2
    ν₁₂ = 0.3
    constitutive = "isotropic"
  elseif (DVDict["material"] == "test")
    ρₛ = 1590.0
    E₁ = 1
    E₂ = 1
    G₁₂ = 1
    ν₁₂ = 0.25
    constitutive = "isotropic"
  end
  g = DVDict["g"]
  θ = DVDict["θ"]

  # --- Compute the structural properties for the foil ---
  EIₛ = zeros(Float64, neval)
  Kₛ = zeros(Float64, neval)
  GJₛ = zeros(Float64, neval)
  Sₛ = zeros(Float64, neval)
  Iₛ = zeros(Float64, neval)
  mₛ = zeros(Float64, neval)
  # --- Loop over the span ---
  for ii in 1:neval
    section = StructProp.section_property(c[ii], t[ii], ab[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, θ)

    EIₛ[ii], Kₛ[ii], GJₛ[ii], Sₛ[ii], Iₛ[ii], mₛ[ii] = StructProp.compute_section_property(section)
  end

  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα = Hydro.compute_glauert_circ(semispan=DVDict["s"], chordVec=c, α₀=DVDict["α₀"] * π / 180, U∞=DVDict["U∞"], neval=neval)

  # ---------------------------
  #   Build final model
  # ---------------------------
  model = foil(c, t, DVDict["s"], ab, eb, x_αb, mₛ, Iₛ, EIₛ, GJₛ, Kₛ, Sₛ, DVDict["α₀"], DVDict["U∞"], DVDict["Λ"], g, clα, DVDict["ρ_f"], DVDict["neval"], constitutive)

  return model

end

end # end module