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
    "../InitModel"
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
    println("Solving nonlinear")

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
        mode="analytic",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    ∂rs∂us = zeros(size(globalK))
    ∂rs∂us[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] =
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]] # - ∂F∂u looking for direct dependence

    partials["deflections", "ptVec"][:, :] = ∂rs∂xPt
    partials["deflections", "deflections"][:, :] = ∂rs∂us
    partials["deflections", "traction_forces"][:, :] = -I(size(partials["deflections", "traction_forces"]))

    return nothing
end

# Not needed if solve_nonlinear! is defined but needed for partials checking
function OpenMDAOCore.apply_nonlinear!(self::OMFEBeam, inputs, outputs, residuals)
    """
    Apply the nonlinear model
    """
    println("Applying nonlinear febeam")

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

function OpenMDAOCore.setup()

end

function OpenMDAOCore.compute!()

end