# --- Julia 1.11---
"""
@File          :   ldtransfer_om.jl
@Date created  :   2025/02/18
@Last modified :   2025/02/19
@Author        :   Galen Ng
@Desc          :   
"""


for headerName in [
    "../loadtransfer/LDTransfer",
]
    include("$(headerName).jl")
end

using .LDTransfer


using OpenMDAOCore: OpenMDAOCore
# ==============================================================================
#                         Load transfer
# ==============================================================================
struct OMLoadTransfer <: OpenMDAOCore.AbstractExplicitComp
    """
    Transfer loads from the hydrodynamic model to the structural model
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMLoadTransfer)
    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMLoadTransfer, inputs, outputs)
    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMLoadTransfer, inputs, partials)
    return nothing
end
# ==============================================================================
#                         Displacement transfer
# ==============================================================================
struct OMDisplacementTransfer <: OpenMDAOCore.AbstractExplicitComp
    """
    Transfer displacements from the structural model to the hydrodynamic model
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMDisplacementTransfer)
    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMDisplacementTransfer, inputs, outputs)
    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMDisplacementTransfer, inputs, partials)
    return nothing
end