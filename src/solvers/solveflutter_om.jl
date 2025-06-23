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
        # --- Structural variables ---
        OpenMDAOCore.VarData("theta_f", val=0.0),
        OpenMDAOCore.VarData("toc", val=self.appendageParams["toc"]),
        # OpenMDAOCore.VarData("alfa0", val=0.0),
        # --- linearized quantities ---
        OpenMDAOCore.VarData("cla", val=zeros(nNodeTot)),
    ]

    outputs = [
        OpenMDAOCore.VarData("ksflutter", val=0.0),
    ]

    partials = [
        OpenMDAOCore.PartialsData("ksflutter", "cla", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("ksflutter", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "theta_f", method="exact"),
        OpenMDAOCore.PartialsData("ksflutter", "toc", method="exact"),
        # OpenMDAOCore.PartialsData("ksflutter", "alfa0", method="exact"),
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
    ptVec = inputs["ptVec"]
    theta_f = inputs["theta_f"][1]
    toc = inputs["toc"]
    displacements_col = inputs["displacements_col"]
    alfa0 = appendageParams["alfa0"]
    depth0 = appendageParams["depth0"]
    println("=============================")
    println("Flutter alfa0 = $(alfa0) deg")
    println("Flutter depth0 = $(depth0) m")
    println("=============================")

    # --- Set struct vars ---
    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc



    obj, FLUTTERSOL = SolveFlutter.cost_funcsFromDVsOM(ptVec, nodeConn, displacements_col, cla, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=true)

    outputs["ksflutter"][1] = obj

    # --- Write solution file ---
    SolveFlutter.write_sol(FLUTTERSOL, solverOptions["outputDir"])

    # --- Repack mesh for LiftingLine ---
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    SolveFlutter.solve_frequencies(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMFlutter, inputs, partials)

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

    evalFuncsSensList = ["ksflutter"]

    funcsSens = SolveFlutter.evalFuncsSens(evalFuncsSensList, appendageParams, GridStruct, displacements_col, cla, solverOptions; mode="RAD")

    partials["ksflutter", "ptVec"][:] = vec(funcsSens["ksflutter"]["mesh"])
    partials["ksflutter", "cla"][:] = funcsSens["ksflutter"]["params"]["cla"]
    partials["ksflutter", "theta_f"][:] = [funcsSens["ksflutter"]["params"]["theta_f"]] # make it a vector
    partials["ksflutter", "toc"][:] = funcsSens["ksflutter"]["params"]["toc"]

    return nothing
end