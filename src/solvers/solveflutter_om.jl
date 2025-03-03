# --- Julia 1.11---
"""
@File          :   solveflutter_om.jl
@Date created  :   2025/02/24
@Last modified :   2025/02/24
@Author        :   Galen Ng
@Desc          :   openmdao wrapper for flutter solution
"""

for headerName = [
    "../solvers/SolveFlutter",
    "../hydro/LiftingLine",
    "../struct/FEMMethods",
]
    include(headerName * ".jl")
end

using .SolveFlutter
using .LiftingLine
using .FEMMethods

# ==============================================================================
#                         OpenMDAO operations
# ==============================================================================
using OpenMDAOCore: OpenMDAOCore

struct OMFlutter <: OpenMDAOCore.AbstractExplicitComp
    """
    Options for the flutter solver
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMFlutter)

    nNodeTot, nNodeWing, nElemTot, nElemWing = FEMMethods.get_numnodes(self.appendageOptions["config"], self.appendageOptions["nNodes"], self.appendageOptions["nNodeStrut"])

    inputs = [
        # --- Mesh type ---
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, LiftingLine.NPT_WING)), # collocation points 
        OpenMDAOCore.VarData("nodes", val=zeros(3, nNodeTot)),
        OpenMDAOCore.VarData("elemConn", val=zeros(2, nElemTot)),
        # --- linearized quantities ---
        OpenMDAOCore.VarData("cla", val=zeros(nNodeTot)),
        OpenMDAOCore.VarData("Mmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
        OpenMDAOCore.VarData("Cmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
        OpenMDAOCore.VarData("Kmat", val=zeros(nNodeTot * FEMMethods.NDOF, nNodeTot * FEMMethods.NDOF)),
    ]

    outputs = [
        OpenMDAOCore.VarData("ksflutter", val=0.0),
    ]

    partials = [
        OpenMDAOCore.PartialsData("ksflutter", "cla", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "Mmat", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "Cmat", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "Kmat", method="exact"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMFlutter, inputs, outputs)

    cla = inputs["cla"]


    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    obj, SOL = SolveFlutter.compute_solFromCoords(appendageParams, solverOptions)

    outputs["ksflutter"][1] = obj

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMFlutter, inputs, partials)

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    funcsSens = SolveFlutter.evalFuncsSens()

    partials["ksflutter", "cla"][1, :] = funcsSens["cla"]

    return nothing
end