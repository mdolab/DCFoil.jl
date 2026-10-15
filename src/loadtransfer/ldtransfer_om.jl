# --- Julia 1.11---
"""
@File          :   ldtransfer_om.jl
@Date created  :   2025/02/18
@Last modified :   2026/10/13
@Author        :   Galen Ng
@Desc          :   openmdao wrappers for the load and displacement transfer
                   * Python stores arrays row-major and julia column-major, so the jacobians wrt 2D variables
                   are built from C-order flattened vectors (see cvec and from_cvec)
"""

for headerName in [
    "../loadtransfer/LDTransfer",
]
    include("$(headerName).jl")
end

using .LDTransfer

using ForwardDiff: ForwardDiff
using OpenMDAOCore: OpenMDAOCore

# --- Helpers to flatten/unflatten 2D arrays in python (C) order ---
cvec(A::AbstractMatrix) = vec(permutedims(A))
cvec(A::AbstractVector) = A
from_cvec(v, dims::Tuple{Int,Int}) = permutedims(reshape(v, reverse(dims)))

# ==============================================================================
#                         Displacement transfer
# ==============================================================================
struct OMDisplacementTransfer <: OpenMDAOCore.AbstractExplicitComp
    """
    Transfer displacements from the structural model to the hydrodynamic model
    """
    n_node::Int # number of FEM nodes for half or full wing
    n_strips::Int # number of lifting line strips for half or full wing
    xMount::Float64 # subtract xMount from collocationPts x coordinates
    config::String # 'full-wing' or 'wing'
    hack_rot_X::Bool # flip sign of X rotation (see LDTransfer.transfer_displacements)
end

function n_strips_full(self::OMDisplacementTransfer)
    # disp_colloc is always for the full span, even if the config is 'wing'
    return self.config == "wing" ? 2 * self.n_strips : self.n_strips
end

function OpenMDAOCore.setup(self::OMDisplacementTransfer)
    inputs = [
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, self.n_strips)),
        OpenMDAOCore.VarData("nodes", val=zeros(self.n_node, 3)),
        OpenMDAOCore.VarData("deflections", val=zeros(9 * self.n_node)),
    ]
    outputs = [
        OpenMDAOCore.VarData("disp_colloc", val=zeros(6, n_strips_full(self))),
    ]
    partials = [
        OpenMDAOCore.PartialsData("disp_colloc", "collocationPts", method="exact"),
        OpenMDAOCore.PartialsData("disp_colloc", "nodes", method="exact"),
        OpenMDAOCore.PartialsData("disp_colloc", "deflections", method="exact"),
    ]
    return inputs, outputs, partials
end

function compute_disp(self::OMDisplacementTransfer, collocationPts, nodes, deflections)
    return LDTransfer.transfer_displacements(collocationPts, nodes, deflections;
        xMount=self.xMount, config=self.config, hack_rot_X=self.hack_rot_X)
end

function OpenMDAOCore.compute!(self::OMDisplacementTransfer, inputs, outputs)
    outputs["disp_colloc"][:, :] = compute_disp(self, inputs["collocationPts"], inputs["nodes"], inputs["deflections"])
    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMDisplacementTransfer, inputs, partials)
    collocationPts = copy(inputs["collocationPts"])
    nodes = copy(inputs["nodes"])
    deflections = copy(inputs["deflections"])

    partials["disp_colloc", "collocationPts"][:, :] = ForwardDiff.jacobian(
        x -> cvec(compute_disp(self, from_cvec(x, size(collocationPts)), nodes, deflections)), cvec(collocationPts))
    partials["disp_colloc", "nodes"][:, :] = ForwardDiff.jacobian(
        x -> cvec(compute_disp(self, collocationPts, from_cvec(x, size(nodes)), deflections)), cvec(nodes))
    partials["disp_colloc", "deflections"][:, :] = ForwardDiff.jacobian(
        x -> cvec(compute_disp(self, collocationPts, nodes, x)), deflections)

    return nothing
end

# ==============================================================================
#                         Load transfer
# ==============================================================================
struct OMLoadTransfer <: OpenMDAOCore.AbstractExplicitComp
    """
    Transfer loads from the hydrodynamic model to the structural model
    """
    n_node::Int # number of FEM nodes for half or full wing
    n_strips::Int # number of lifting line strips for half or full wing
    xMount::Float64 # subtract xMount from collocationPts x coordinates
    config::String # 'full-wing' or 'wing'
end

function OpenMDAOCore.setup(self::OMLoadTransfer)
    inputs = [
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, self.n_strips)),
        OpenMDAOCore.VarData("forces_hydro", val=zeros(3, self.n_strips)),
        OpenMDAOCore.VarData("nodes", val=zeros(self.n_node, 3)),
    ]
    outputs = [
        OpenMDAOCore.VarData("loads_str", val=zeros(9 * self.n_node)),
    ]
    partials = [
        OpenMDAOCore.PartialsData("loads_str", "collocationPts", method="exact"),
        OpenMDAOCore.PartialsData("loads_str", "forces_hydro", method="exact"),
        OpenMDAOCore.PartialsData("loads_str", "nodes", method="exact"),
    ]
    return inputs, outputs, partials
end

function compute_loads(self::OMLoadTransfer, collocationPts, forces_hydro, nodes)
    return LDTransfer.transfer_loads(collocationPts, forces_hydro, nodes; xMount=self.xMount, config=self.config)
end

function OpenMDAOCore.compute!(self::OMLoadTransfer, inputs, outputs)
    outputs["loads_str"][:] = compute_loads(self, inputs["collocationPts"], inputs["forces_hydro"], inputs["nodes"])
    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMLoadTransfer, inputs, partials)
    collocationPts = copy(inputs["collocationPts"])
    forces_hydro = copy(inputs["forces_hydro"])
    nodes = copy(inputs["nodes"])

    partials["loads_str", "collocationPts"][:, :] = ForwardDiff.jacobian(
        x -> compute_loads(self, from_cvec(x, size(collocationPts)), forces_hydro, nodes), cvec(collocationPts))
    partials["loads_str", "forces_hydro"][:, :] = ForwardDiff.jacobian(
        x -> compute_loads(self, collocationPts, from_cvec(x, size(forces_hydro)), nodes), cvec(forces_hydro))
    partials["loads_str", "nodes"][:, :] = ForwardDiff.jacobian(
        x -> compute_loads(self, collocationPts, forces_hydro, from_cvec(x, size(nodes))), cvec(nodes))

    return nothing
end

# ==============================================================================
#                         CL_alpha interpolation
# ==============================================================================
struct OMCLaInterpolation <: OpenMDAOCore.AbstractExplicitComp
    """
    Interpolate CL_alpha from flow collocation points to FEM nodes
    """
    n_node::Int # number of FEM nodes
    n_strips::Int # number of lifting line strips
end

function OpenMDAOCore.setup(self::OMCLaInterpolation)
    inputs = [
        OpenMDAOCore.VarData("CL_alpha", val=zeros(self.n_strips)),
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, self.n_strips)),
        OpenMDAOCore.VarData("nodes", val=zeros(self.n_node, 3)),
    ]
    outputs = [
        OpenMDAOCore.VarData("CL_alpha_node", val=zeros(self.n_node)),
    ]
    partials = [
        OpenMDAOCore.PartialsData("CL_alpha_node", "CL_alpha", method="exact"),
        OpenMDAOCore.PartialsData("CL_alpha_node", "collocationPts", method="exact"),
        OpenMDAOCore.PartialsData("CL_alpha_node", "nodes", method="exact"),
    ]
    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMCLaInterpolation, inputs, outputs)
    outputs["CL_alpha_node"][:] = LDTransfer.interp_cla(inputs["CL_alpha"], inputs["collocationPts"], inputs["nodes"])
    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMCLaInterpolation, inputs, partials)
    CL_alpha = copy(inputs["CL_alpha"])
    collocationPts = copy(inputs["collocationPts"])
    nodes = copy(inputs["nodes"])

    partials["CL_alpha_node", "CL_alpha"][:, :] = ForwardDiff.jacobian(
        x -> LDTransfer.interp_cla(x, collocationPts, nodes), CL_alpha)
    partials["CL_alpha_node", "collocationPts"][:, :] = ForwardDiff.jacobian(
        x -> LDTransfer.interp_cla(CL_alpha, from_cvec(x, size(collocationPts)), nodes), cvec(collocationPts))
    partials["CL_alpha_node", "nodes"][:, :] = ForwardDiff.jacobian(
        x -> LDTransfer.interp_cla(CL_alpha, collocationPts, from_cvec(x, size(nodes))), cvec(nodes))

    return nothing
end
