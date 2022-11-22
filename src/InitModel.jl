
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- Public functions ---
export init_steady, init_dynamic

include("./hydro/Hydro.jl")
include("./struct/BeamProperties.jl")
include("./constants/DesignConstants.jl")
using .Hydro, .StructProp
using .DesignConstants

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
  println("+","-"^50, "+")
  println("|            Design variable dictionary:           |")
  println("+","-"^50, "+")
  for kv in DVDict
    println(kv)
  end

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
  model = DesignConstants.foil(c, t, DVDict["s"], ab, eb, x_αb, mₛ, Iₛ, EIₛ, GJₛ, Kₛ, Sₛ, DVDict["α₀"], DVDict["U∞"], DVDict["Λ"], g, clα, DVDict["ρ_f"], DVDict["neval"], constitutive)

  return model

end

function init_dynamic(fSweep, DVDict::Dict)
  """
  Perform much of the same initializations as init_steady() except with other features
  """
  steadyModel = init_steady(DVDict["neval"], DVDict)

  model = DesignConstants.dynamicFoil(steadyModel.c, steadyModel.t, steadyModel.s, steadyModel.ab, steadyModel.eb, steadyModel.x_αb, steadyModel.mₛ, steadyModel.Iₛ, steadyModel.EIₛ, steadyModel.GJₛ, steadyModel.Kₛ, steadyModel.Sₛ, steadyModel.α₀, steadyModel.U∞, steadyModel.Λ, steadyModel.g, fSweep, steadyModel.clα, steadyModel.ρ_f, steadyModel.neval, steadyModel.constitutive)

  return model
end

end # end module