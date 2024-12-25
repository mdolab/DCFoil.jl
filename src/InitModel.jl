
# --- Julia ---

# @File    :   InitModel.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to initialize the hydrofoil model and store data


module InitModel

# --- PACKAGES ---
using Zygote
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
# using Debugger

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex
using ..HydroStrip: HydroStrip
using ..BeamProperties: BeamProperties
using ..DesignConstants: DesignConstants, SORTEDDVS, CONFIGS
using ..MaterialLibrary: MaterialLibrary
using ..HullLibrary: HullLibrary
using ..Preprocessing: Preprocessing
using ..FEMMethods: FEMMethods
using ..Utilities: Utilities

function init_static(
  α₀, sweepAng, rake, span, chordLengths, toc, ab, x_ab, zeta, theta_f,
  beta, span_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut,
  depth0,
  appendageOptions::Dict, solverOptions::Dict
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
  eb::Vector{RealOrComplex} = 0.25 * chordLengths .+ ab
  t::Vector{RealOrComplex} = toc .* chordLengths

  # ---------------------------
  #   Structure
  # ---------------------------
  ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(appendageOptions["material"])

  # --- Compute the structural properties for the foil ---
  nNodes = appendageOptions["nNodes"]

  if haskey(appendageOptions, "path_to_struct_props") && !isnothing(appendageOptions["path_to_struct_props"])
    println("Reading structural properties from file: ", appendageOptions["path_to_struct_props"])
    EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = Preprocessing.get_1DBeamPropertiesFromFile(appendageOptions["path_to_struct_props"])
  else
    EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodes, chordLengths, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f, constitutive; solverOptions=solverOptions)
  end

  # ---------------------------
  #   Hydrodynamics
  # ---------------------------
  clα, _, _, LLSystem, FlowCond = HydroStrip.compute_hydroLLProperties(span, chordLengths, α₀, rake, sweepAng, depth0; solverOptions=solverOptions, appendageOptions=appendageOptions)

  # ---------------------------
  #   Build final model
  # ---------------------------
  wingModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, solverOptions["Uinf"], zeta,
    clα, eb, ab, chordLengths, solverOptions["rhof"], appendageOptions["nNodes"], constitutive)

  # ************************************************
  #     Strut properties
  # ************************************************
  if appendageOptions["config"] == "t-foil" && !solverOptions["use_nlll"]
    # Do it again using the strut properties
    nNodesStrut = appendageOptions["nNodeStrut"]
    ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(appendageOptions["strut_material"])
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
      clα, eb_strut, ab_strut, c_strut, solverOptions["rhof"], appendageOptions["nNodeStrut"], constitutive)

  elseif appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
    strutModel = nothing
  elseif !(appendageOptions["config"] in CONFIGS)
    error("Unsupported config: ", appendageOptions["config"])
  end

  # appendageSystem = wingModel

  return wingModel, strutModel, LLSystem, FlowCond

end

function init_staticHydro(LECoords, TECoords, nodeConn, appendageParams,
  appendageOptions::Dict, solverOptions::Dict
)
  """
  Initialize a static hydrofoil model

  Inputs:
      DVs for derivative computation

  returns:
    foil: struct
  """

  ptVec, mm, nn = Utilities.unpack_coords(LECoords, TECoords)
  LLOutputs, LLSystem, FlowCond = HydroStrip.compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

  return LLOutputs, LLSystem, FlowCond

end

function init_dynamic(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams::Dict,
  appendageOptions::Dict, solverOptions::Dict; fRange=[0.1, 1], uRange=[0.0, 1.0]
)
  """
  Perform much of the same initializations as init_static() except with other features
  """


  LLOutputs, LLSystem, FlowCond = init_staticHydro(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)


  statWingStructModel, statStrutStructModel = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

  WingStructModel = DesignConstants.DynamicFoil(
    statWingStructModel.mₛ, statWingStructModel.Iₛ, statWingStructModel.EIₛ, statWingStructModel.EIIPₛ, statWingStructModel.GJₛ, statWingStructModel.Kₛ, statWingStructModel.Sₛ, statWingStructModel.EAₛ,
    statWingStructModel.eb, statWingStructModel.ab, statWingStructModel.chord, statWingStructModel.nNodes, statWingStructModel.constitutive,
    fRange, uRange
  )

  if isnothing(statStrutStructModel)
    StrutStructModel = nothing
  else
    StrutStructModel = DesignConstants.DynamicFoil(
      statStrutStructModel.mₛ, statStrutStructModel.Iₛ, statStrutStructModel.EIₛ, statStrutStructModel.EIIPₛ, statStrutStructModel.GJₛ, statStrutStructModel.Kₛ, statStrutStructModel.Sₛ, statStrutStructModel.EAₛ, statStrutStructModel.U∞, statStrutStructModel.ζ,
      statStrutStructModel.clα, statStrutStructModel.eb, statStrutStructModel.ab, statStrutStructModel.chord, statStrutStructModel.nNodes, statStrutStructModel.constitutive, fRange, uRange
    )
  end

  return WingStructModel, StrutStructModel, LLOutputs, LLSystem, FlowCond
end

function init_hull(solverOptions::Dict)
  """
  Initialize the hull model
  """
  mass, length, beam, xcg, Ib = HullLibrary.return_hullprop(solverOptions["hull"])
  HullModel = DesignConstants.Hull(mass, Ib, xcg, length, beam)
  return HullModel
end

# function init_modelFromDVDict(DVDict::Dict, solverOptions::Dict, appendageOptions::Dict; fRange=[0.1, 1.0], uRange=[0.0, 1.0])
#   """
#   This is a wrapper for init_dynamic() that unpacks a DV dictionary
#   """

#   # ************************************************
#   #     DVs that need to be unpacked
#   # ************************************************
#   # NOTE: this is not all DVs!

#   if haskey(appendageOptions, "path_to_geom_props")
#     α₀ = DVDict["alfa0"]
#     sweepAng = DVDict["sweep"]
#     rake = DVDict["rake"]
#     span = DVDict["s"] * 2
#     c = DVDict["c"]
#     zeta = DVDict["zeta"]
#     theta_f = DVDict["theta_f"]
#     beta = DVDict["beta"]
#     s_strut = DVDict["s_strut"]
#     c_strut = DVDict["c_strut"]
#     theta_f_strut = DVDict["theta_f_strut"]
#     depth0 = DVDict["depth0"]

#     toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
#   else
#     α₀ = DVDict["alfa0"]
#     sweepAng = DVDict["sweep"]
#     rake = DVDict["rake"]
#     span = DVDict["s"] * 2
#     c::Vector{RealOrComplex} = DVDict["c"]
#     toc::Vector{RealOrComplex} = DVDict["toc"]
#     ab::Vector{RealOrComplex} = DVDict["ab"]
#     x_ab::Vector{RealOrComplex} = DVDict["x_ab"]
#     # toc = DVDict["toc"]
#     # ab = DVDict["ab"]
#     # x_ab = DVDict["x_ab"]
#     zeta = DVDict["zeta"]
#     theta_f = DVDict["theta_f"]
#     beta = DVDict["beta"]
#     s_strut = DVDict["s_strut"]
#     c_strut = DVDict["c_strut"]
#     toc_strut = DVDict["toc_strut"]
#     ab_strut = DVDict["ab_strut"]
#     x_ab_strut = DVDict["x_ab_strut"]
#     theta_f_strut = DVDict["theta_f_strut"]
#     depth0 = DVDict["depth0"]
#   end

#   WingModel, StrutModel = init_dynamic(α₀, sweepAng, rake, span, c, toc, ab, x_ab, zeta, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut, depth0, appendageOptions, solverOptions; fRange=fRange, uRange=uRange)


#   if solverOptions["run_body"]
#     HullModel = init_hull(solverOptions)
#   else
#     HullModel = nothing
#   end

#   return WingModel, StrutModel, HullModel
# end

function init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)

  fRange = solverOptions["fRange"]
  uRange = solverOptions["uRange"]

  idxTip = Preprocessing.get_tipnode(LECoords)
  midchords, chordLengths, spanwiseVectors, Λ = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)


  if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
    print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

    α₀ = appendageParams["alfa0"]
    rake = appendageParams["rake"]
    # span = appendageParams["s"] * 2
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    beta = appendageParams["beta"]
    s_strut = appendageParams["s_strut"]
    c_strut = appendageParams["c_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]
    depth0 = appendageParams["depth0"]

    toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
  else
    rake = appendageParams["rake"]
    # span = appendageParams["s"] * 2
    toc::Vector{RealOrComplex} = appendageParams["toc"]
    ab::Vector{RealOrComplex} = appendageParams["ab"]
    x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
    zeta = appendageParams["zeta"]
    theta_f = appendageParams["theta_f"]
    beta = appendageParams["beta"]
    s_strut = appendageParams["s_strut"]
    c_strut = appendageParams["c_strut"]
    toc_strut = appendageParams["toc_strut"]
    ab_strut = appendageParams["ab_strut"]
    x_ab_strut = appendageParams["x_ab_strut"]
    theta_f_strut = appendageParams["theta_f_strut"]
  end

  WingModel, StrutModel, LLOutputs, LLSystem, FlowCond = init_dynamic(LECoords, TECoords, nodeConn, toc, ab, zeta, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions; fRange=fRange, uRange=uRange)

  idxTip = Preprocessing.get_tipnode(LECoords)
  structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, @ignore_derivatives(nodeConn),idxTip, appendageParams, appendageOptions)
  FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))


  if haskey(solverOptions, "run_body") && solverOptions["run_body"]
    HullModel = init_hull(solverOptions)
  else
    HullModel = nothing
  end

  return WingModel, StrutModel, HullModel, FEMESH, LLOutputs, LLSystem, FlowCond
end

end # end module