# --- Julia 1.11---
"""
@File          :   solveforced_om.jl
@Date created  :   2025/02/24
@Last modified :   2025/02/24
@Author        :   Galen Ng
@Desc          :   openmdao wrapper for forced vibration solution
"""

for headerName = [
    "../solvers/SolveForced",
]
    include(headerName * ".jl")
end

using .SolveForced

# ==============================================================================
#                         OpenMDAO operations
# ==============================================================================
using OpenMDAOCore: OpenMDAOCore

struct OMForced <: OpenMDAOCore.AbstractExplicitComp
    """
    Options for the flutter solver
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMForced)


    inputs = [
        OpenMDAOCore.VarData("deflections", val=zeros(nNodeTot * FEMMethods.NDOF)),
    ]

    outputs = [
        OpenMDAOCore.VarData("elemConn", val=zeros(2, nElemTot)),
    ]

    partials = [
        OpenMDAOCore.PartialsData("thetatip", "deflections", method="exact"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMForced, inputs, outputs)


    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMForced, inputs, partials)


    return nothing
end