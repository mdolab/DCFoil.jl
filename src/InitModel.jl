
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- PACKAGES ---
using Zygote

# --- DCFoil modules ---
using ..HydroStrip
using ..BeamProperties
using ..DesignConstants
using ..MaterialLibrary
using ..HullLibrary

function init_static(α₀, rake, span, c, toc, ab, x_αb, g, θ, beta, span_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut, foilOptions::Dict, solverOptions::Dict)
  """
  Initialize a static hydrofoil model

  Inputs:
      DVs for derivative computation

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
  eb::Vector{Float64} = 0.25 * c .+ ab
  t::Vector{Float64} = toc .* c

  # ---------------------------
  #   Structure
  # ---------------------------
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(foilOptions["material"])
  # g::Float64 = DVDict["zeta"]
  # θ::Float64 = DVDict["θ"]

  # --- Compute the structural properties for the foil ---
  nNodes = foilOptions["nNodes"]
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
    section = BeamProperties.SectionProperty(c[ii], t[ii], ab[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, θ, zeros(20, 2))

    # TODO: should probably redo this to be element-based, not node-based
    EI_z[ii], EIIP_z[ii], K_z[ii], GJ_z[ii], S_z[ii], EA_z[ii], I_z[ii], m_z[ii] = BeamProperties.compute_section_property(section, constitutive)
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
  clα = Vector{Float64}(undef, nNodes)
  clα, _, _ = HydroStrip.compute_glauert_circ(span, c, deg2rad(α₀ + rake), solverOptions["U∞"], nNodes;
    h=span_strut, # TODO: DEPTH
    useFS=solverOptions["use_freeSurface"],
    rho=solverOptions["ρ_f"],
    config=foilOptions["config"],
  )

  # ---------------------------
  #   Build final model
  # ---------------------------
  wingModel = DesignConstants.foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["U∞"], g, clα, solverOptions["ρ_f"], foilOptions["nNodes"], constitutive)

  # ************************************************
  #     Strut properties
  # ************************************************
  if foilOptions["config"] == "t-foil"
    # Do it again using the strut properties
    nNodesStrut = foilOptions["nNodeStrut"]
    ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(foilOptions["strut_material"])
    t_strut::Vector{Float64} = toc_strut .* c_strut
    eb_strut::Vector{Float64} = 0.25 * c_strut .+ ab_strut

    EIₛ = zeros(Float64, nNodesStrut)
    Kₛ = zeros(Float64, nNodesStrut)
    GJₛ = zeros(Float64, nNodesStrut)
    Sₛ = zeros(Float64, nNodesStrut)
    Iₛ = zeros(Float64, nNodesStrut)
    mₛ = zeros(Float64, nNodesStrut)
    # --- Loop over the span ---
    EI_z = Zygote.Buffer(EIₛ)
    EIIP_z = Zygote.Buffer(EIₛ)
    EA_z = Zygote.Buffer(EIₛ)
    K_z = Zygote.Buffer(Kₛ)
    GJ_z = Zygote.Buffer(GJₛ)
    S_z = Zygote.Buffer(Sₛ)
    I_z = Zygote.Buffer(Iₛ)
    m_z = Zygote.Buffer(mₛ)

    for ii in 1:nNodesStrut
      section = BeamProperties.SectionProperty(c_strut[ii], t_strut[ii], ab_strut[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, θ_strut, zeros(20, 2))
      # TODO: should probably redo this to be element-based, not node-based
      EI_z[ii], EIIP_z[ii], K_z[ii], GJ_z[ii], S_z[ii], EA_z[ii], I_z[ii], m_z[ii] = BeamProperties.compute_section_property(section, constitutive)
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
    clα, _, _ = HydroStrip.compute_glauert_circ(span_strut, c_strut, deg2rad(0.001), solverOptions["U∞"], nNodesStrut)

    # ---------------------------
    #   Build final model
    # ---------------------------
    strutModel = DesignConstants.foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["U∞"], g, clα, solverOptions["ρ_f"], foilOptions["nNodeStrut"], constitutive)

  elseif foilOptions["config"] == "wing" || foilOptions["config"] == "full-wing"
    strutModel = nothing
  else
    error("Unsupported config: ", foilOptions["config"])
  end

  return wingModel, strutModel

end

function init_dynamic(α₀, rake, span, c, toc, ab, x_αb, g, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut,
  foilOptions::Dict, solverOptions::Dict; fSweep=0.1:0.1:1, uRange=[0.0, 1.0]
)
  """
  Perform much of the same initializations as init_static() except with other features
  """
  # statModel = init_static(DVDict, solverOptions)
  statWingModel, statStrutModel = init_static(α₀, rake, span, c, toc, ab, x_αb, g, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut, foilOptions, solverOptions)

  # model = DesignConstants.dynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_αb, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.nNodes, staticModel.constitutive, fSweep, uRange)
  WingModel = DesignConstants.dynamicFoil(
    statWingModel.mₛ, statWingModel.Iₛ, statWingModel.EIₛ, statWingModel.EIIPₛ, statWingModel.GJₛ, statWingModel.Kₛ, statWingModel.Sₛ, statWingModel.EAₛ, statWingModel.U∞, statWingModel.g, statWingModel.clα, statWingModel.ρ_f, statWingModel.nNodes, statWingModel.constitutive,
    fSweep, uRange
  )
  if statStrutModel == nothing
    StrutModel = nothing
  else
    StrutModel = DesignConstants.dynamicFoil(
      statStrutModel.mₛ, statStrutModel.Iₛ, statStrutModel.EIₛ, statStrutModel.EIIPₛ, statStrutModel.GJₛ, statStrutModel.Kₛ, statStrutModel.Sₛ, statStrutModel.EAₛ, statStrutModel.U∞, statStrutModel.g, statStrutModel.clα, statStrutModel.ρ_f, statStrutModel.nNodes, statStrutModel.constitutive, fSweep, uRange
    )
  end

  return WingModel, StrutModel
end

function init_hull(solverOptions::Dict)
  """
  Initialize the hull model
  """
  mass, length, beam, xcg, Ib = HullLibrary.return_hullprop(solverOptions["hull"])
  HullModel = DesignConstants.hull(mass, Ib, xcg, length, beam)
  return HullModel
end

function init_model_wrapper(DVDict::Dict, solverOptions::Dict, appendageOptions::Dict; fSweep=0.1:0.1:1, uRange=[0.0, 1.0])
  """
  This is a wrapper for init_dynamic() that unpacks a DV dictionary
  """

  # ************************************************
  #     DVs that need to be unpacked
  # ************************************************
  # NOTE: this is not all DVs!
  α₀ = DVDict["α₀"]
  rake = DVDict["rake"]
  span = DVDict["s"]
  c = DVDict["c"]
  toc = DVDict["toc"]
  ab = DVDict["ab"]
  x_αb = DVDict["x_αb"]
  g = DVDict["zeta"]
  θ = DVDict["θ"]
  beta = DVDict["beta"]
  s_strut = DVDict["s_strut"]
  c_strut = DVDict["c_strut"]
  toc_strut = DVDict["toc_strut"]
  ab_strut = DVDict["ab_strut"]
  x_αb_strut = DVDict["x_αb_strut"]
  θ_strut = DVDict["θ_strut"]

  # if length(solverOptions["appendageList"]) == 1
  WingModel, StrutModel = init_dynamic(α₀, rake, span, c, toc, ab, x_αb, g, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut, appendageOptions, solverOptions; fSweep, uRange)
  # else
  # error("Only one appendage is supported at the moment")
  # end


  if solverOptions["run_body"]
    HullModel = init_hull(solverOptions)
  else
    HullModel = nothing
  end

  return WingModel, StrutModel, HullModel
end

end # end module