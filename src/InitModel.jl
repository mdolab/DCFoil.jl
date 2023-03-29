
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

function init_static(DVDict::Dict, solverOptions)
  """
  Initialize a static hydrofoil model

  Inputs:
      DVDict: Dict, dictionary of model parameters, the design variables
      --> you can let these accept Complex64 dtype to complex step the code
      solverOptions: Dict, dictionary of solver options

  returns:
    foil: struct
  """

  # # --- First print to screen in a box ---
  # println("+", "-"^50, "+")
  # println("|            Design variable dictionary:           |")
  # println("+", "-"^50, "+")
  # for kv in DVDict
  #   println(kv)
  # end

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
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(solverOptions["material"])
  g::Float64 = DVDict["g"]
  θ::Float64 = DVDict["θ"]

  # --- Compute the structural properties for the foil ---
  nNodes = solverOptions["nNodes"]
  EIₛ = zeros(Float64, nNodes)
  Kₛ = zeros(Float64, nNodes)
  GJₛ = zeros(Float64, nNodes)
  Sₛ = zeros(Float64, nNodes)
  Iₛ = zeros(Float64, nNodes)
  mₛ = zeros(Float64, nNodes)
  # --- Loop over the span ---
  for ii in 1:nNodes
    section = StructProp.section_property(c[ii], t[ii], ab[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, θ)

    EIₛ[ii], Kₛ[ii], GJₛ[ii], Sₛ[ii], Iₛ[ii], mₛ[ii] = StructProp.compute_section_property(section, constitutive)
  end

  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα = Hydro.compute_glauert_circ(semispan=DVDict["s"], chordVec=c, α₀=deg2rad(DVDict["α₀"]), U∞=solverOptions["U∞"], nNodes=nNodes)

  # ---------------------------
  #   Build final model
  # ---------------------------
  model = DesignConstants.foil(c, t, DVDict["s"], ab, eb, x_αb, mₛ, Iₛ, EIₛ, GJₛ, Kₛ, Sₛ, DVDict["α₀"], solverOptions["U∞"], DVDict["Λ"], g, clα, solverOptions["ρ_f"], solverOptions["nNodes"], constitutive)

  return model

end

function init_dynamic(DVDict::Dict, solverOptions::Dict; fSweep=0.1:0.1:1, uRange=[0.0, 1.0])
  """
  Perform much of the same initializations as init_static() except with other features
  """
  staticModel = init_static(DVDict, solverOptions)

  model = DesignConstants.dynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_αb, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.nNodes, staticModel.constitutive, fSweep, uRange)

  return model
end

end # end module