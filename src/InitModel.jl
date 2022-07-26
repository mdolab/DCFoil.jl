
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model


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
  ab # dist from midchord to EA vector (+ve for EA aft)
  eb # dist from CP to EA (+ve for EA aft)
  x_α # static imbalance (+ve for CG aft)
  mₛ # structural mass vector [kg/m]
  Iₛ # structural moment of inertia vector [kg-m]
  EIₛ # bending stiffness vector 
  GJₛ # torsion stiffness vector 
  Kₛ # bend-twist coupling vector
  Sₛ # warping resistance vector
  α₀ # rigid initial angle of attack [deg]
  U∞ # flow speed [m/s]
  Λ # sweep angle [rad]
  g # structural damping percentage
  clα # lift slopes
  ρ_f::Float64 # fluid density [kg/m³]
  neval::Int64 # number of evaluation points on span
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
  x_α = DVDict["x_α"]

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
  model = foil(c, t, DVDict["s"], ab, eb, x_α, mₛ, Iₛ, EIₛ, GJₛ, Kₛ, Sₛ, DVDict["α₀"], DVDict["U∞"], DVDict["Λ"], g, clα, DVDict["ρ_f"], DVDict["neval"])

  return model

end

end # end module