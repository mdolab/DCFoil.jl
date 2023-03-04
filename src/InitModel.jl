
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- Public functions ---
export init_static, init_dynamic

include("./hydro/Hydro.jl")
include("./struct/BeamProperties.jl")
include("./constants/DesignConstants.jl")
using .Hydro, .StructProp
using .DesignConstants

function init_static(neval::Int64, DVDict::Dict)
  """
  Initialize a static hydrofoil model

  Inputs:
      neval: Int64, number of evaluation points on span
      DVDict: Dict, dictionary of model parameters, the design variables

  returns:
    foil: struct
  """

  # --- First print to screen in a box ---
  println("+", "-"^50, "+")
  println("|            Design variable dictionary:           |")
  println("+", "-"^50, "+")
  for kv in DVDict
    println(kv)
  end

  # ---------------------------
  #   Geometry
  # ---------------------------
  c::Vector{Float64} = DVDict["c"]
  t::Vector{Float64} = DVDict["toc"] * c
  ab::Vector{Float64} = DVDict["ab"]
  eb::Vector{Float64} = 0.25 * c + ab
  x_αb::Vector{Float64} = DVDict["x_αb"]

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
    G₁₂ = 77.2e9
    ν₁₂ = 0.3
    constitutive = "isotropic"
  elseif (DVDict["material"] == "rigid") # unrealistic rigid material
    ρₛ = 7900
    E₁ = 193e12
    E₂ = 193e12
    G₁₂ = 77.2e12
    ν₁₂ = 0.3
    constitutive = "isotropic"
  elseif (DVDict["material"] == "eirikurPl") # unrealistic rigid material
    ρₛ = 2800
    E₁ = 70e9
    E₂ = 70e9
    ν₁₂ = 0.3
    G₁₂ = E₁ / 2 / (1 + ν₁₂)
    constitutive = "isotropic"
  elseif (DVDict["material"] == "test-iso")
    ρₛ = 1590.0
    E₁ = 1
    E₂ = 1
    G₁₂ = 1
    ν₁₂ = 0.25
    # constitutive = "isotropic"
    constitutive = "orthotropic" # NOTE: Need to use this because the isotropic case uses an ellipse for GJ
  elseif (DVDict["material"] == "test-comp")
    ρₛ = 1590.0
    E₁ = 1
    E₂ = 1
    G₁₂ = 1
    ν₁₂ = 0.25
    constitutive = "orthotropic"

  end
  g::Float64 = DVDict["g"]
  θ::Float64 = DVDict["θ"]

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

    EIₛ[ii], Kₛ[ii], GJₛ[ii], Sₛ[ii], Iₛ[ii], mₛ[ii] = StructProp.compute_section_property(section, constitutive)
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

function init_dynamic(fSweep, DVDict::Dict; uSweep=0:0.1:1)
  """
  Perform much of the same initializations as init_static() except with other features

  the default uSweep is a dummy array so type declaration works
  """
  staticModel = init_static(DVDict["neval"], DVDict)

  model = DesignConstants.dynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_αb, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.neval, staticModel.constitutive, fSweep, uSweep)

  return model
end

end # end module