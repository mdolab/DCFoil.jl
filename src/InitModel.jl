
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
using ..Utilities: get_1DPropertiesFromFile

function init_static(
  α₀, sweepAng, rake, span, chord, toc, ab, x_ab, zeta, theta_f,
  beta, span_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut,
  depth0,
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

  EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodes, chord, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f, constitutive; solverOptions=solverOptions)
  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα, _, _ = HydroStrip.compute_hydroLLProperties(span, chord, α₀, rake, sweepAng, depth0; solverOptions=solverOptions)
  # clα, _, _ = HydroStrip.compute_glauert_circ(span, chord, deg2rad(α₀ + rake), solverOptions["Uinf"];
  #   h=depth0,
  #   useFS=solverOptions["use_freeSurface"],
  #   rho=solverOptions["rhof"],
  #   config=foilOptions["config"],
  #   debug=solverOptions["debug"]
  # )

  # ---------------------------
  #   Build final model
  # ---------------------------
  wingModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["Uinf"], zeta,
    clα, eb, ab, chord, solverOptions["rhof"], foilOptions["nNodes"], constitutive)

  # ************************************************
  #     Strut properties
  # ************************************************
  if foilOptions["config"] == "t-foil" && !solverOptions["use_nlll"]
    # Do it again using the strut properties
    nNodesStrut = foilOptions["nNodeStrut"]
    ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(foilOptions["strut_material"])
    t_strut = toc_strut .* c_strut
    eb_strut = 0.25 * c_strut .+ ab_strut

    EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodesStrut, c_strut, t_strut, ab_strut, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f_strut, constitutive; solverOptions=solverOptions)

    # ---------------------------
    #   Hydrodynamics
    # ---------------------------
    clα, _, _ = HydroStrip.compute_hydroLLProperties(span_strut, c_strut, deg2rad(0.001), 0.0, 0.0, 0.0; solverOptions=solverOptions)
    # clα, _, _ = GlauertLL.compute_glauert_circ(span_strut, c_strut, , solverOptions["Uinf"])

    # ---------------------------
    #   Build final model
    # ---------------------------
    strutModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["Uinf"], zeta,
      clα, eb_strut, ab_strut, c_strut, solverOptions["rhof"], foilOptions["nNodeStrut"], constitutive)

  elseif foilOptions["config"] == "wing" || foilOptions["config"] == "full-wing"
    strutModel = nothing
  else
    error("Unsupported config: ", foilOptions["config"])
  end

  # appendageSystem = wingModel

  return wingModel, strutModel

end

function init_dynamic(α₀, sweepAng, rake, span, c, toc, ab, x_ab, ζ, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut, depth0,
  foilOptions::Dict, solverOptions::Dict; fRange=[0.1, 1], uRange=[0.0, 1.0]
)
  """
  Perform much of the same initializations as init_static() except with other features
  """
  # statModel = init_static(DVDict, solverOptions)
  statWingModel, statStrutModel = init_static(α₀, sweepAng, rake, span, c, toc, ab, x_ab, ζ, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut, depth0, foilOptions, solverOptions)

  # model = DesignConstants.DynamicFoil(staticModel.c, staticModel.t, staticModel.s, staticModel.ab, staticModel.eb, staticModel.x_ab, staticModel.mₛ, staticModel.Iₛ, staticModel.EIₛ, staticModel.GJₛ, staticModel.Kₛ, staticModel.Sₛ, staticModel.α₀, staticModel.U∞, staticModel.Λ, staticModel.g, staticModel.clα, staticModel.ρ_f, staticModel.nNodes, staticModel.constitutive, fRange, uRange)
  WingModel = DesignConstants.DynamicFoil(
    statWingModel.mₛ, statWingModel.Iₛ, statWingModel.EIₛ, statWingModel.EIIPₛ, statWingModel.GJₛ, statWingModel.Kₛ, statWingModel.Sₛ, statWingModel.EAₛ, statWingModel.U∞, statWingModel.ζ,
    statWingModel.clα, statWingModel.eb, statWingModel.ab, statWingModel.chord, statWingModel.ρ_f, statWingModel.nNodes, statWingModel.constitutive,
    fRange, uRange
  )
  if statStrutModel == nothing
    StrutModel = nothing
  else
    StrutModel = DesignConstants.DynamicFoil(
      statStrutModel.mₛ, statStrutModel.Iₛ, statStrutModel.EIₛ, statStrutModel.EIIPₛ, statStrutModel.GJₛ, statStrutModel.Kₛ, statStrutModel.Sₛ, statStrutModel.EAₛ, statStrutModel.U∞, statStrutModel.ζ,
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

  # --- Check if beam properties come from file or params ---
  if !isnothing(appendageOptions["path_to_props"])
    # TODO: PICKUP HERE
    Utilities.get_1DPropertiesFromFile(appendageOptions["path_to_props"])
  else
    α₀ = DVDict["alfa0"]
    sweepAng = DVDict["sweep"]
    rake = DVDict["rake"]
    span = DVDict["s"] * 2
    c = DVDict["c"]
    toc = DVDict["toc"]
    ab = DVDict["ab"]
    x_ab = DVDict["x_ab"]
    zeta = DVDict["zeta"]
    theta_f = DVDict["theta_f"]
    beta = DVDict["beta"]
    s_strut = DVDict["s_strut"]
    c_strut = DVDict["c_strut"]
    toc_strut = DVDict["toc_strut"]
    ab_strut = DVDict["ab_strut"]
    x_ab_strut = DVDict["x_ab_strut"]
    theta_f_strut = DVDict["theta_f_strut"]
    depth0 = DVDict["depth0"]
  end


  WingModel, StrutModel = init_dynamic(α₀, sweepAng, rake, span, c, toc, ab, x_ab, zeta, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut, depth0, appendageOptions, solverOptions; fRange=fRange, uRange=uRange)


  if solverOptions["run_body"]
    HullModel = init_hull(solverOptions)
  else
    HullModel = nothing
  end

  return WingModel, StrutModel, HullModel
end

function init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)

  fRange = solverOptions["fRange"]
  uRange = solverOptions["uRange"]


  midchords, chordLengths, spanwiseVectors = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

  if !isnothing(appendageOptions["path_to_geom_props"])
    print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

    α₀ = appendageParams["alfa0"]
    sweepAng = appendageParams["sweep"]
    rake = appendageParams["rake"]
    span = appendageParams["s"] * 2
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    beta = appendageParams["beta"]
    s_strut = appendageParams["s_strut"]
    c_strut = appendageParams["c_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]
    depth0 = appendageParams["depth0"]

    toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
  else
    α₀ = appendageParams["alfa0"]
    sweepAng = appendageParams["sweep"]
    rake = appendageParams["rake"]
    span = appendageParams["s"] * 2
    toc = appendageParams["toc"]
    ab = appendageParams["ab"]
    x_ab = appendageParams["x_ab"]
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    beta = appendageParams["beta"]
    s_strut = appendageParams["s_strut"]
    c_strut = appendageParams["c_strut"]
    toc_strut = appendageParams["toc_strut"]
    ab_strut = appendageParams["ab_strut"]
    x_ab_strut = appendageParams["x_ab_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]
    depth0 = appendageParams["depth0"]
  end


  WingModel, StrutModel = init_dynamic(α₀, sweepAng, rake, span, chordLengths, toc, ab, x_ab, zeta, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut, depth0, appendageOptions, solverOptions; fRange=fRange, uRange=uRange)

  structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, appendageParams, appendageOptions)
  FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, zeros(10, 2))


  if solverOptions["run_body"]
    HullModel = init_hull(solverOptions)
  else
    HullModel = nothing
  end

  return WingModel, StrutModel, HullModel, FEMESH
end

end # end module