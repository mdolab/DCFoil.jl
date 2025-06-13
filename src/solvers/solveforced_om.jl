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
        OpenMDAOCore.VarData("ksbend", val=0.0),
        OpenMDAOCore.VarData("kstwist", val=0.0),
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

    cla = inputs["cla"]
    ptVec = inputs["ptVec"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]
    displacements_col = inputs["displacements_col"]
    alfa0 = appendageParams["alfa0"]
    println("=============================")
    println("Forced alfa0 = $(alfa0) deg")
    println("=============================")

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    obj, VIBSOL = SolveForced.compute_funcsFromDVsOM(ptVec, nodeConn, displacements_col, claVec, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=true)

    outputs["vibareaw"][1] = obj[1]
    outputs["ksbend"][1] = obj[2]
    outputs["kstwist"][1] = obj[3]

    # --- Write solution file ---
    SolveForced.write_sol(VIBSOL, solverOptions["outputDir"])

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMForced, inputs, partials)

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    appendageOptions = self.appendageOptions
    solverOptions = self.solverOptions

    cla = inputs["cla"]
    ptVec = inputs["ptVec"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]
    displacements_col = inputs["displacements_col"]

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc

    LEMesh, TEMesh = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    GridStruct = Grid(LEMesh, nodeConn, TEMesh)

    funcsSens = SolveForced.evalFuncsSens(appendageParams, GridStruct, displacements_col, cla, solverOptions; mode="RAD")

    partials["vibareaw", "ptVec"][:] = vec(funcsSens["vibareaw"]["mesh"])
    partials["vibareaw", "cla"][:] = funcsSens["vibareaw"]["params"]["cla"]
    partials["vibareaw", "theta_f"][:] = [funcsSens["vibareaw"]["params"]["theta_f"]] # make it a vector
    partials["vibareaw", "toc"][:] = funcsSens["vibareaw"]["params"]["toc"]

    partials["ksbend", "ptVec"][:] = vec(funcsSens["ksbend"]["mesh"])
    partials["ksbend", "cla"][:] = funcsSens["ksbend"]["params"]["cla"]
    partials["ksbend", "theta_f"][:] = [funcsSens["ksbend"]["params"]["theta_f"]] # make it a vector
    partials["ksbend", "toc"][:] = funcsSens["ksbend"]["params"]["toc"]

    partials["kstwist", "ptVec"][:] = vec(funcsSens["kstwist"]["mesh"])
    partials["kstwist", "cla"][:] = funcsSens["kstwist"]["params"]["cla"]
    partials["kstwist", "theta_f"][:] = [funcsSens["kstwist"]["params"]["theta_f"]] # make it a vector
    partials["kstwist", "toc"][:] = funcsSens["kstwist"]["params"]["toc"]

    return nothing
end