# --- Julia 1.11---
"""
@File          :   beam_om.jl
@Date created  :   2025/02/10
@Last modified :   2025/02/10
@Author        :   Galen Ng
@Desc          :   openmdao wrapper for the finite element beam
"""

# --- Used in this script ---
using LinearAlgebra
using ChainRulesCore

# --- Module to om wrap ---
for headerName = [
    "../struct/FEMMethods",
    "../InitModel",
    "../ComputeStructFunctions",
]
    include(headerName * ".jl")
end

using .FEMMethods
# ==============================================================================
#                         OpenMDAO operations
# ==============================================================================
using OpenMDAOCore: OpenMDAOCore

struct OMFEBeam <: OpenMDAOCore.AbstractImplicitComp
    """
    Options for the finite element beam solver
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMFEBeam)
    """
    Setup OpenMDAO data
    """

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1

    nNodeTot, nNodeWing, nElemTot, nElemWing = FEMMethods.get_numnodes(self.appendageOptions["config"], self.appendageOptions["nNodes"], self.appendageOptions["nNodeStrut"])


    inputs = [
        OpenMDAOCore.VarData("ptVec", val=zeros(npt * 3 * 2)),
        OpenMDAOCore.VarData("traction_forces", val=zeros(nNodeTot * FEMMethods.NDOF)),
        # --- Structural variables ---
        OpenMDAOCore.VarData("theta_f", val=0.0),
        OpenMDAOCore.VarData("toc", val=ones(nNodeWing)),
    ]
    outputs = [
        OpenMDAOCore.VarData("deflections", val=zeros(nNodeTot * FEMMethods.NDOF)),
    ]
    partials = [
        # --- Residuals ---
        OpenMDAOCore.PartialsData("deflections", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("deflections", "traction_forces", method="exact"),
        OpenMDAOCore.PartialsData("deflections", "theta_f", method="exact"),
        OpenMDAOCore.PartialsData("deflections", "toc", method="exact"),
        OpenMDAOCore.PartialsData("deflections", "deflections", method="exact"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.solve_nonlinear!(self::OMFEBeam, inputs, outputs)
    """
    Solve the FEM model
    """
    println("Solving nonlinear beam")

    ptVec = inputs["ptVec"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]
    traction_forces = inputs["traction_forces"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    # ************************************************
    #     Core solver
    # ************************************************
    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    globalK, globalM, globalF, DOFBlankingList, FEMESH, _, _ = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    qSol = FEMMethods.solve_structure(
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        traction_forces[1:end.∉[DOFBlankingList]],
    )
    uSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE; appendageOptions=appendageOptions)

    # --- Set outputs ---
    outputs["deflections"][:] = uSol

    return nothing
end

function OpenMDAOCore.linearize!(self::OMFEBeam, inputs, outputs, partials)
    """
    This defines the derivatives of outputs
    """

    ptVec = inputs["ptVec"]
    traction_forces = inputs["traction_forces"]
    allStructStates = outputs["deflections"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    globalK, globalM, globalF, DOFBlankingList, FEMESH = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    # Params derivatives are computed using complex step
    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="analytic", # better
        # mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    ∂rs∂us = zeros(size(globalK))
    ∂rs∂us[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] =
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] # - ∂F∂u looking for direct dependence

    # Set ones to diagonal to the deflections of 1st node (where the BC is applied)
    for i in 1:9
        ∂rs∂us[i, i] = 1.0
    end

    # partials["deflections", "ptVec"][:, :] .= 0.0
    # partials["deflections", "ptVec"][1:end.∉[DOFBlankingList], :] = ∂rs∂xPt
    partials["deflections", "ptVec"][:, :] = ∂rs∂xPt
    partials["deflections", "deflections"][:, :] = ∂rs∂us
    partials["deflections", "traction_forces"][1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] =
        -I(length(traction_forces) - length(DOFBlankingList))

    # println("partials toc", size(∂rs∂xParams["toc"]))

    partials["deflections", "theta_f"][:, :] = ∂rs∂xParams["theta_f"]
    partials["deflections", "toc"][:, :] = ∂rs∂xParams["toc"]

    return nothing
end

# Not needed if solve_nonlinear! is defined but needed for partials checking
function OpenMDAOCore.apply_nonlinear!(self::OMFEBeam, inputs, outputs, residuals)
    """
    Apply the nonlinear model
    """

    ptVec = inputs["ptVec"]
    traction_forces = inputs["traction_forces"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]
    allStructStates = outputs["deflections"]
    residuals["deflections"] .= 0.0

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    resVec = FEMMethods.compute_residualsFromCoords(allStructStates, ptVec, nodeConn, traction_forces, [appendageParams];
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    # Residuals are of the output state variable
    for (ii, res) in enumerate(resVec)
        residuals["deflections"][ii] = res
    end

    return nothing
end

# ==============================================================================
#                         Beam cost functions
# ==============================================================================
# Use OM Explicit component to define cost functions like tip deflections, etc.
struct OMFEBeamFuncs <: OpenMDAOCore.AbstractExplicitComp
    """
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMFEBeamFuncs)

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1
    nNodeTot, nNodeWing, nElemTot, nElemWing = FEMMethods.get_numnodes(self.appendageOptions["config"], self.appendageOptions["nNodes"], self.appendageOptions["nNodeStrut"])

    inputs = [
        OpenMDAOCore.VarData("ptVec", val=zeros(3 * 2 * npt)),
        OpenMDAOCore.VarData("deflections", val=zeros(nNodeTot * FEMMethods.NDOF)),
        # --- Structural variables ---
        OpenMDAOCore.VarData("theta_f", val=0.0),
        OpenMDAOCore.VarData("toc", val=ones(nNodeWing)),
    ]

    outputs = [
        OpenMDAOCore.VarData("wtip", val=0.0),
        OpenMDAOCore.VarData("thetatip", val=0.0),
        OpenMDAOCore.VarData("nodes", val=zeros(nNodeTot, 3)),
        OpenMDAOCore.VarData("elemConn", val=zeros(nElemTot, 2)),
        # OpenMDAOCore.VarData("Mmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
        # OpenMDAOCore.VarData("Cmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
        # OpenMDAOCore.VarData("Kmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
    ]

    partials = [
        # WRT ptVec
        # OpenMDAOCore.PartialsData("Kmat", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("Cmat", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("Mmat", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("wtip", "ptVec", method="exact"), # this is zero
        # OpenMDAOCore.PartialsData("thetatip", "ptVec", method="exact"), # this is zero
        OpenMDAOCore.PartialsData("nodes", "ptVec", method="exact"),
        # WRT deflections
        OpenMDAOCore.PartialsData("wtip", "deflections", method="exact"),
        OpenMDAOCore.PartialsData("thetatip", "deflections", method="exact"),
        # WRT struct variables
        # OpenMDAOCore.PartialsData("Kmat", "theta_f", method="exact"),
        # OpenMDAOCore.PartialsData("Kmat", "toc", method="exact"),
        # OpenMDAOCore.PartialsData("Cmat", "theta_f", method="exact"),
        # OpenMDAOCore.PartialsData("Cmat", "toc", method="exact"),
        # OpenMDAOCore.PartialsData("Mmat", "theta_f", method="exact"),
        # OpenMDAOCore.PartialsData("Mmat", "toc", method="exact"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="fd")] # define the partials

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMFEBeamFuncs, inputs, outputs)

    states = inputs["deflections"]
    ptVec = inputs["ptVec"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    wtip = compute_maxtipbend(states)
    thetatip = compute_maxtiptwist(states)

    outputs["wtip"][1] = wtip
    outputs["thetatip"][1] = thetatip


    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    Kmat, Mmat, _, _, FEMESH = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    alphaConst = solverOptions["alphaConst"]
    betaConst = solverOptions["betaConst"]
    Cmat = alphaConst * Mmat .+ betaConst * Kmat

    size(FEMESH.mesh) == size(outputs["nodes"]) || error("Size mismatch")
    size(FEMESH.elemConn) == size(outputs["elemConn"]) || error("Size mismatch")
    outputs["nodes"][:] = FEMESH.mesh
    outputs["elemConn"][:] = FEMESH.elemConn

    # outputs["Kmat"][:] = Kmat
    # outputs["Cmat"][:] = Cmat
    # outputs["Mmat"][:] = Mmat

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMFEBeamFuncs, inputs, partials)
    # states = inputs["deflections"]
    # ptVec = inputs["ptVec"]
    # theta_f = inputs["theta_f"][1]
    # toc = inputs["toc"]

    # # --- Deal with options here ---
    # nodeConn = self.nodeConn
    # appendageParams = self.appendageParams
    # appendageOptions = self.appendageOptions
    # solverOptions = self.solverOptions

    # --- Set struct vars ---
    # appendageParams["theta_f"] = theta_f
    # appendageParams["toc"] = toc

    zv1 = zeros(length(inputs["deflections"]))
    zv1[end-NDOF+WIND] = 1.0

    zv2 = zeros(length(inputs["deflections"]))
    zv2[end-NDOF+ΘIND] = 1.0

    partials["wtip", "deflections"][1, :] = zv1
    partials["thetatip", "deflections"][1, :] = zv2


    # FEMMethods.compute_∂matrices∂x(ptVec, nodeConn,)
    # partials["Kmat", "theta_f"][:] = 0.0
    # partials["Kmat", "toc"][:] = 0.0
    # partials["Kmat", "ptVec"][:] = 0.0
    # partials["Cmat", "theta_f"][:] = 0.0
    # partials["Cmat", "toc"][:] = 0.0
    # partials["Cmat", "ptVec"][:] = 0.0
    # partials["Mmat", "theta_f"][:] = 0.0
    # partials["Mmat", "toc"][:] = 0.0
    # partials["Mmat", "ptVec"][:] = 0.0

    return nothing
end