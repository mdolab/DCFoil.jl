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
    # These scripts are already imported in earlier scripts so we don't need to import them again
    # "../hydro/LiftingLine",
    # "../struct/FEMMethods",
]
    include(headerName * ".jl")
end

using .SolveFlutter
# using .LiftingLine
# using .FEMMethods

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

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1

    inputs = [
        # --- Mesh type ---
        OpenMDAOCore.VarData("ptVec", val=zeros(3 * 2 * npt)),
        OpenMDAOCore.VarData("displacements_col", val=zeros(6, LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("nodes", val=zeros(nNodeTot, 3)),
        OpenMDAOCore.VarData("elemConn", val=zeros(nElemTot, 2)),
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

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    cla = inputs["cla"]
    KKmat = inputs["Kmat"]
    CCmat = inputs["Cmat"]
    MMmat = inputs["Mmat"]
    nodes = inputs["nodes"]
    elemConn = inputs["elemConn"]
    ptVec = inputs["ptVec"]
    displacements_col = inputs["displacements_col"]


    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    obj = SolveFlutter.cost_funcsFromDVsOM(ptVec, nodeConn, displacements_col, nodes, elemConn, cla, KKmat, CCmat, MMmat, appendageParams, solverOptions)

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
    # partials["ksflutter", "Kmat"][1, :] = funcsSens["cla"]
    # partials["ksflutter", "Cmat"][1, :] = funcsSens["cla"]
    # partials["ksflutter", "Mmat"][1, :] = funcsSens["cla"]

    return nothing
end