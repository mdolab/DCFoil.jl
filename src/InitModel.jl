
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- PACKAGES ---
using Zygote

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex
using ..HydroStrip: HydroStrip
using ..BeamProperties: BeamProperties
using ..DesignConstants: DesignConstants, SORTEDDVS
using ..MaterialLibrary: MaterialLibrary
using ..HullLibrary: HullLibrary

function init_static(
  α₀, rake, span, chord, toc, ab, x_αb, zeta, θ,
  beta, span_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut,
  foilOptions::Dict, solverOptions::Dict
)
  """
  Initialize a static hydrofoil model

  Inputs:
      DVs for derivative computation

  returns:
    foil: struct
  """

  # ---------------------------
  #   Geometry
  # ---------------------------
  eb = 0.25 * chord .+ ab
  t = toc .* chord

  # ---------------------------
  #   Structure
  # ---------------------------
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(foilOptions["material"])

  # --- Compute the structural properties for the foil ---
  nNodes = foilOptions["nNodes"]
  EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodes, chord, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, θ, constitutive)
  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα, _, _ = HydroStrip.compute_glauert_circ(span, chord, deg2rad(α₀ + rake), solverOptions["U∞"], nNodes;
    h=span_strut, # TODO: DEPTH
    useFS=solverOptions["use_freeSurface"],
    rho=solverOptions["ρ_f"],
    config=foilOptions["config"],
  )

  # ---------------------------
  #   Build final model
  # ---------------------------
  wingModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["U∞"], zeta,
    clα, eb, ab, chord, solverOptions["ρ_f"], foilOptions["nNodes"], constitutive)

  # ************************************************
  #     Strut properties
  # ************************************************
  if foilOptions["config"] == "t-foil"
    # Do it again using the strut properties
    nNodesStrut = foilOptions["nNodeStrut"]
    ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(foilOptions["strut_material"])
    t_strut = toc_strut .* c_strut
    eb_strut = 0.25 * c_strut .+ ab_strut

    EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = compute_beam(nNodesStrut, c_strut, t_strut, ab_strut, ρₛ, E₁, E₂, G₁₂, ν₁₂, θ_strut, constitutive)

    # ---------------------------
    #   Hydrodynamics
    # ---------------------------
    clα, _, _ = HydroStrip.compute_glauert_circ(span_strut, c_strut, deg2rad(0.001), solverOptions["U∞"], nNodesStrut)

    # ---------------------------
    #   Build final model
    # ---------------------------
    strutModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["U∞"], zeta,
      clα, eb_strut, ab_strut, c_strut, solverOptions["ρ_f"], foilOptions["nNodeStrut"], constitutive)

  elseif foilOptions["config"] == "wing" || foilOptions["config"] == "full-wing"
    strutModel = nothing
  else
    error("Unsupported config: ", foilOptions["config"])
  end

  return wingModel, strutModel

end

function init_dynamic(α₀, rake, span, c, toc, ab, x_αb, ζ, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut,
  foilOptions::Dict, solverOptions::Dict; fRange=[0.1, 1], uRange=[0.0, 1.0]
)
  """
  Perform much of the same initializations as init_static() except with other features
  """
  # statModel = init_static(DVDict, solverOptions)
  statWingModel, statStrutModel = init_static(α₀, rake, span, c, toc, ab, x_αb, ζ, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut, foilOptions, solverOptions)

  # model = DesignConstants.DynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_αb, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.nNodes, staticModel.constitutive, fRange, uRange)
  WingModel = DesignConstants.DynamicFoil(
    statWingModel.mₛ, statWingModel.Iₛ, statWingModel.EIₛ, statWingModel.EIIPₛ, statWingModel.GJₛ, statWingModel.Kₛ, statWingModel.Sₛ, statWingModel.EAₛ, statWingModel.U∞, statWingModel.ζ,
    statWingModel.clα, statWingModel.eb, statWingModel.ab, statWingModel.chord, statWingModel.ρ_f, statWingModel.nNodes, statWingModel.constitutive,
    fRange, uRange
  )
  if statStrutModel == nothing
    StrutModel = nothing
  else
    StrutModel = DesignConstants.DynamicFoil(
      statStrutModel.mₛ, statStrutModel.Iₛ, statStrutModel.EIₛ, statStrutModel.EIIPₛ, statStrutModel.GJₛ, statStrutModel.Kₛ, statStrutModel.Sₛ, statStrutModel.EAₛ, statStrutModel.U∞, statStrutModel.g,
      statStrutModel.clα, statStrutModel.eb, statStrutModel.ab, statStrutModel.chord, statStrutModel.ρ_f, statStrutModel.nNodes, statStrutModel.constitutive, fRange, uRange
    )
  end

  return WingModel, StrutModel
end

function init_hull(solverOptions::Dict)
  """
  Initialize the hull model
  """
  mass, length, beam, xcg, Ib = HullLibrary.return_hullprop(solverOptions["hull"])
  HullModel = DesignConstants.Hull(mass, Ib, xcg, length, beam)
  return HullModel
end

function init_model_wrapper(DVDict::Dict, solverOptions::Dict, appendageOptions::Dict; fRange=[0.1, 1.0], uRange=[0.0, 1.0])
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
  zeta = DVDict["zeta"]
  θ = DVDict["θ"]
  beta = DVDict["beta"]
  s_strut = DVDict["s_strut"]
  c_strut = DVDict["c_strut"]
  toc_strut = DVDict["toc_strut"]
  ab_strut = DVDict["ab_strut"]
  x_αb_strut = DVDict["x_αb_strut"]
  θ_strut = DVDict["θ_strut"]

  # if length(solverOptions["appendageList"]) == 1
  WingModel, StrutModel = init_dynamic(α₀, rake, span, c, toc, ab, x_αb, zeta, θ, beta, s_strut, c_strut, toc_strut, ab_strut, x_αb_strut, θ_strut, appendageOptions, solverOptions; fRange=fRange, uRange=uRange)
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