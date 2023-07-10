
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- Public functions ---
export init_static, init_dynamic
# --- Libraries ---
using Zygote
include("./hydro/Hydro.jl")
include("./struct/BeamProperties.jl")
include("./struct/MaterialLibrary.jl")
include("./constants/DesignConstants.jl")
using .Hydro, .StructProp, .MaterialLibrary
using .DesignConstants

# function init_static(DVDict::Dict, solverOptions)
function init_static(α₀, span, c, toc, ab, x_αb, g, θ, solverOptions::Dict)
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
  # c::Vector{Float64} = DVDict["c"]
  # t::Vector{Float64} = DVDict["toc"] * c
  # ab::Vector{Float64} = DVDict["ab"]
  # x_αb::Vector{Float64} = DVDict["x_αb"]
  eb::Vector{Float64} = 0.25 * c + ab
  t::Vector{Float64} = toc * c

  # ---------------------------
  #   Structure
  # ---------------------------
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(solverOptions["material"])
  # g::Float64 = DVDict["g"]
  # θ::Float64 = DVDict["θ"]

  # --- Compute the structural properties for the foil ---
  nNodes = solverOptions["nNodes"]
  EIₛ = zeros(Float64, nNodes)
  Kₛ = zeros(Float64, nNodes)
  GJₛ = zeros(Float64, nNodes)
  Sₛ = zeros(Float64, nNodes)
  Iₛ = zeros(Float64, nNodes)
  mₛ = zeros(Float64, nNodes)
  # --- Loop over the span ---
  EI_z = Zygote.Buffer(EIₛ)
  EIIP_z = Zygote.Buffer(EIₛ)
  EA_z = Zygote.Buffer(EIₛ)
  K_z = Zygote.Buffer(Kₛ)
  GJ_z = Zygote.Buffer(GJₛ)
  S_z = Zygote.Buffer(Sₛ)
  I_z = Zygote.Buffer(Iₛ)
  m_z = Zygote.Buffer(mₛ)
  for ii in 1:nNodes
    section = StructProp.section_property(c[ii], t[ii], ab[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, θ)

    EI_z[ii], EIIP_z[ii], K_z[ii], GJ_z[ii], S_z[ii], EA_z[ii], I_z[ii], m_z[ii] = StructProp.compute_section_property(section, constitutive)
  end

  EIₛ = copy(EI_z)
  EIIPₛ = copy(EIIP_z)
  EAₛ = copy(EA_z)
  Kₛ = copy(K_z)
  GJₛ = copy(GJ_z)
  Sₛ = copy(S_z)
  Iₛ = copy(I_z)
  mₛ = copy(m_z)


  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα = Hydro.compute_glauert_circ(span, c, deg2rad(α₀), solverOptions["U∞"], nNodes)

  # ---------------------------
  #   Build final model
  # ---------------------------
  model = DesignConstants.foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, α₀, solverOptions["U∞"], g, clα, solverOptions["ρ_f"], solverOptions["nNodes"], constitutive)

  return model

end

# function init_dynamic(DVDict::Dict, solverOptions::Dict; fSweep=0.1:0.1:1, uRange=[0.0, 1.0])
function init_dynamic(α₀, span, c, toc, ab, x_αb, g, θ, solverOptions::Dict; fSweep=0.1:0.1:1, uRange=[0.0, 1.0])
  """
  Perform much of the same initializations as init_static() except with other features
  """
  # statModel = init_static(DVDict, solverOptions)
  statModel = init_static(α₀, span, c, toc, ab, x_αb, g, θ, solverOptions)

  # model = DesignConstants.dynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_αb, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.nNodes, staticModel.constitutive, fSweep, uRange)
  model = DesignConstants.dynamicFoil(
    statModel.mₛ, statModel.Iₛ, statModel.EIₛ, statModel.EIIPₛ, statModel.GJₛ, statModel.Kₛ, statModel.Sₛ, statModel.EAₛ, statModel.α₀, statModel.U∞, statModel.g, statModel.clα, statModel.ρ_f, statModel.nNodes, statModel.constitutive, fSweep, uRange)

  return model
end

function init_model_wrapper(DVDict::Dict, solverOptions; fSweep=0.1:0.1:1, uRange=[0.0, 1.0])
  """
  This is a wrapper for init_dynamic() that unpacks a DV dictionary
  """

  # ************************************************
  #     DVs that need to be unpacked
  # ************************************************
  # NOTE: this is not all DVs!
  α₀ = DVDict["α₀"]
  span = DVDict["s"]
  c = DVDict["c"]
  toc = DVDict["toc"]
  ab = DVDict["ab"]
  x_αb = DVDict["x_αb"]
  g = DVDict["g"]
  θ = DVDict["θ"]

  model = init_dynamic(α₀, span, c, toc, ab, x_αb, g, θ, solverOptions; fSweep, uRange)

  return model
end

end # end module