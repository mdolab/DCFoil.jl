
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
include("./struct/MaterialLibrary.jl")
include("./constants/DesignConstants.jl")
using .Hydro, .StructProp, .MaterialLibrary
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
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(DVDict["material"])
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
  clα = Hydro.compute_glauert_circ(semispan=DVDict["s"], chordVec=c, α₀=deg2rad(DVDict["α₀"]), U∞=DVDict["U∞"], neval=neval)

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