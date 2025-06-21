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

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1

    inputs = [
        # --- Mesh type ---
        OpenMDAOCore.VarData("ptVec", val=zeros(3 * 2 * npt)),
        OpenMDAOCore.VarData("displacements_col", val=zeros(6, LiftingLine.NPT_WING)),
        # --- Structural variables ---
        OpenMDAOCore.VarData("theta_f", val=0.0),
        OpenMDAOCore.VarData("toc", val=self.appendageParams["toc"]),
        OpenMDAOCore.VarData("alfa0", val=0.0),
        # --- linearized quantities ---
        OpenMDAOCore.VarData("cla", val=zeros(nNodeTot)),
    ]

    outputs = [
        OpenMDAOCore.VarData("vibareapsi", val=0.0),
        OpenMDAOCore.VarData("vibareaw", val=0.0),
        OpenMDAOCore.VarData("ksbend", val=0.0),
        OpenMDAOCore.VarData("kstwist", val=0.0),
    ]

    partials = [
        OpenMDAOCore.PartialsData("vibareaw", "cla", method="exact"),
        OpenMDAOCore.PartialsData("vibareaw", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("vibareaw", "theta_f", method="exact"),
        OpenMDAOCore.PartialsData("vibareaw", "toc", method="exact"),
        OpenMDAOCore.PartialsData("ksbend", "cla", method="exact"),
        OpenMDAOCore.PartialsData("ksbend", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("ksbend", "theta_f", method="exact"),
        OpenMDAOCore.PartialsData("ksbend", "toc", method="exact"),
        OpenMDAOCore.PartialsData("kstwist", "cla", method="exact"),
        OpenMDAOCore.PartialsData("kstwist", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("kstwist", "theta_f", method="exact"),
        OpenMDAOCore.PartialsData("kstwist", "toc", method="exact"),
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
    alfa0 = inputs["alfa0"][1]

    # --- Set appendageparam vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc
    appendageParams["alfa0"] = alfa0
    println("=============================")
    println("Forced vibration alfa0 = $(alfa0) deg")
    println("=============================")

    obj, VIBSOL = SolveForced.compute_funcsFromDVsOM(ptVec, nodeConn, displacements_col, cla, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=true)

    outputs["vibareaw"][1] = obj[1]
    outputs["ksbend"][1] = obj[2]
    outputs["kstwist"][1] = obj[3]

    println("objectives = ", obj)

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
    alfa0 = inputs["alfa0"][1]

    # --- Set appendageparam vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc
    appendageParams["alfa0"] = alfa0

    LEMesh, TEMesh = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    GridStruct = Grid(LEMesh, nodeConn, TEMesh)
    println("cla = ", cla)
    funcsSens = SolveForced.evalFuncsSens(appendageParams, GridStruct, displacements_col, cla, solverOptions; mode="RAD")
    println("funcsSens rad 'mesh'= ", funcsSens["vibareaw"]["mesh"])
    println("funcsSens rad 'appendage params'= ", funcsSens["vibareaw"]["params"])
    println("funcsSens rad 'mesh'= ", funcsSens["ksbend"]["mesh"])
    println("funcsSens rad 'appendage params'= ", funcsSens["ksbend"]["params"])
    println("funcsSens rad 'mesh'= ", funcsSens["kstwist"]["mesh"])
    println("funcsSens rad 'appendage params'= ", funcsSens["kstwist"]["params"])
    funcsSens_fidi = SolveForced.evalFuncsSens(appendageParams, GridStruct, displacements_col, cla, solverOptions; mode="FiDi")
    println("funcsSens_fidi 'mesh'= ", funcsSens_fidi["vibareaw"]["mesh"])
    println("funcsSens_fidi 'appendage params'= ", funcsSens_fidi["vibareaw"]["params"])

    partials["vibareaw", "ptVec"][:] = vec(funcsSens["vibareaw"]["mesh"])
    partials["vibareaw", "cla"][:] = funcsSens["vibareaw"]["params"]["cla"]
    partials["vibareaw", "theta_f"][:] = funcsSens["vibareaw"]["params"]["theta_f"]
    partials["vibareaw", "toc"][:] = funcsSens["vibareaw"]["params"]["toc"]

    partials["ksbend", "ptVec"][:] = vec(funcsSens["ksbend"]["mesh"])
    partials["ksbend", "cla"][:] = funcsSens["ksbend"]["params"]["cla"]
    partials["ksbend", "theta_f"][:] = funcsSens["ksbend"]["params"]["theta_f"]
    partials["ksbend", "toc"][:] = funcsSens["ksbend"]["params"]["toc"]

    partials["kstwist", "ptVec"][:] = vec(funcsSens["kstwist"]["mesh"])
    partials["kstwist", "cla"][:] = funcsSens["kstwist"]["params"]["cla"]
    partials["kstwist", "theta_f"][:] = funcsSens["kstwist"]["params"]["theta_f"]
    partials["kstwist", "toc"][:] = funcsSens["kstwist"]["params"]["toc"]

    return nothing
end