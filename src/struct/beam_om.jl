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
    ]
    outputs = [
        OpenMDAOCore.VarData("deflections", val=zeros(nNodeTot * FEMMethods.NDOF)),
    ]
    partials = [
        # --- Residuals ---
        OpenMDAOCore.PartialsData("deflections", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("deflections", "traction_forces", method="exact"),
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
    traction_forces = inputs["traction_forces"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # ************************************************
    #     Core solver
    # ************************************************
    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    globalK, globalM, globalC, DOFBlankingList, FEMESH = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

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

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    # ************************************************
    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    globalK, globalM, globalC, DOFBlankingList, FEMESH = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="analytic", # better
        # mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    ∂rs∂us = zeros(size(globalK))
    ∂rs∂us[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] =
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] # - ∂F∂u looking for direct dependence

    # partials["deflections", "ptVec"][:, :] .= 0.0
    # partials["deflections", "ptVec"][1:end.∉[DOFBlankingList], :] = ∂rs∂xPt
    partials["deflections", "ptVec"][:, :] = ∂rs∂xPt
    partials["deflections", "deflections"][:, :] = ∂rs∂us
    partials["deflections", "traction_forces"][1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] =
        -I(length(traction_forces) - length(DOFBlankingList))

    return nothing
end

# Not needed if solve_nonlinear! is defined but needed for partials checking
function OpenMDAOCore.apply_nonlinear!(self::OMFEBeam, inputs, outputs, residuals)
    """
    Apply the nonlinear model
    """

    ptVec = inputs["ptVec"]
    traction_forces = inputs["traction_forces"]
    allStructStates = outputs["deflections"]
    residuals["deflections"] .= 0.0

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions


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
    ]

    outputs = [
        OpenMDAOCore.VarData("wtip", val=0.0),
        OpenMDAOCore.VarData("thetatip", val=0.0),
        OpenMDAOCore.VarData("nodes", val=zeros(3, nNodeTot)),
        OpenMDAOCore.VarData("elemConn", val=zeros(2, nElemTot)),
    ]

    partials = [
        # # WRT ptVec
        # OpenMDAOCore.PartialsData("wtip", "ptVec", method="exact"), # this is zero
        # OpenMDAOCore.PartialsData("thetatip", "ptVec", method="exact"), # this is zero
        OpenMDAOCore.PartialsData("nodes", "ptVec", method="fd"),
        # WRT deflections
        OpenMDAOCore.PartialsData("wtip", "deflections", method="exact"),
        OpenMDAOCore.PartialsData("thetatip", "deflections", method="exact"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="fd")] # define the partials

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMFEBeamFuncs, inputs, outputs)

    states = inputs["deflections"]
    ptVec = inputs["ptVec"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    wtip = compute_maxtipbend(states)
    thetatip = compute_maxtiptwist(states)

    outputs["wtip"][1] = wtip
    outputs["thetatip"][1] = thetatip


    LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    _, _, _, _, FEMESH = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    # println(size(outputs["nodes"]))
    # println(size(outputs["elemConn"]))
    # for (ii, node) in enumerate(eachcol(FEMESH.mesh))
    #     outputs["nodes"][:, ii] = node
    # end

    outputs["nodes"][:] = FEMESH.mesh
    outputs["elemConn"][:] = FEMESH.elemConn

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMFEBeamFuncs, inputs, partials)

    zv1 = zeros(length(inputs["deflections"]))
    zv1[end-NDOF+WIND] = 1.0

    zv2 = zeros(length(inputs["deflections"]))
    zv2[end-NDOF+ΘIND] = 1.0

    partials["wtip", "deflections"][1, :] = zv1
    partials["thetatip", "deflections"][1, :] = zv2

    return nothing
end