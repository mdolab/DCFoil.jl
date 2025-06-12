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
    # "../hydro/LiftingLine",
    # "../struct/FEMMethods",
]
    include(headerName * ".jl")
end

using .SolveForced
# using .LiftingLine
# using .FEMMethods

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

    nNodeTot, nNodeWing, nElemTot, nElemWing = FEMMethods.get_numnodes(self.appendageOptions["config"], self.appendageOptions["nNodes"], self.appendageOptions["nNodeStrut"])

    inputs = [
        # --- Mesh type ---
        # OpenMDAOCore.VarData("collocationPts", val=zeros(3, LiftingLine.NPT_WING)), # collocation points 
        OpenMDAOCore.VarData("nodes", val=zeros(nNodeTot, 3)),
        OpenMDAOCore.VarData("elemConn", val=zeros(nElemTot, 2)),
        # --- linearized quantities ---
        OpenMDAOCore.VarData("cla", val=zeros(nNodeTot)),
        # OpenMDAOCore.VarData("deflections", val=zeros(nNodeTot * FEMMethods.NDOF)),
    ]

    outputs = [
        OpenMDAOCore.VarData("vibareapsi", val=0.0),
        OpenMDAOCore.VarData("vibareaw", val=0.0),
    ]

    partials = [
        # OpenMDAOCore.PartialsData("thetatip", "deflections", method="exact"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMForced, inputs, outputs)


    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    obj, VIBSOL = SolveForced.compute_funcsFromDVsOM(ptVec, nodeConn, displacementsCol, claVec, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=true)
    SolveForced.write_sol(VIBSOL, solverOptions["outputDir"])

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMForced, inputs, partials)


    return nothing
end