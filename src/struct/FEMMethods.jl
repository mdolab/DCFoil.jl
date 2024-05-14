# --- Julia 1.7---
"""
@File    :   FEMMethods.jl
@Time    :   2022/08/04
@Author  :   Galen Ng
@Desc    :   Finite element library
Module with generic FEM methods

# KNOWN BUGS:
    span derivative may result in array size being wrong
    NaN shows up in structural matrices that breaks the solve
    This NaN originates from the mesh generation routine most likely
    Zygote Buffer data types suck and you should find a more stable way.
    I initialized the mesh properly in the Buffer but it still broke
"""


module FEMMethods

# --- PACKAGES ---
using Zygote
using ChainRulesCore
using DelimitedFiles
using LinearAlgebra
using StaticArrays

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex, DTYPE
using ..EBBeam: EBBeam as BeamElement, NDOF
using ..SolverRoutines
using ..BeamProperties
using ..DesignConstants: DynamicFoil
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE

struct StructMesh{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    """
    Struct to hold the mesh, element connectivity, and node properties
    """
    mesh::TM # node xyz coords (2D array of coordinates of nodes)
    elemConn::Matrix{TI} # element-node connectivity [elemIdx] => [globalNode1Idx, globalNode2Idx]
    # The stuff below is only stored for output file writing. DO NOT USE IN CALCULATIONS
    chord::TA
    toc::TA
    ab::TA
    x_αb::TA
    θ::TF # global fiber frame orientation
    airfoilCoords::TM # airfoil coordinates
end

function make_fullMesh(DVDictList, solverOptions)
    """
    Makes a full body mesh with element connectivity
    """

    # ************************************************
    #     Loop over appendages in the problem
    # ************************************************
    StructMeshList = []
    ElemConnList = []
    for iComp in eachindex(solverOptions["appendageList"])
        appendageDict = solverOptions["appendageList"][iComp]
        DVDict = DVDictList[iComp]
        println("Meshing ", appendageDict["compName"])
        span = DVDict["s"]
        spanStrut = DVDict["s_strut"]
        nElem = appendageDict["nNodes"] - 1
        nElStrut = appendageDict["nNodeStrut"] - 1
        config = appendageDict["config"]
        rake = DVDict["rake"]
        structMesh, elemConn = make_componentMesh(nElem, span; config=config, nElStrut=nElStrut, spanStrut=spanStrut, rake=rake)

        # # check magnitudes of the mesh if greater than 1e5
        # if any(abs.(structMesh) .> 1e5)
        #     println("Mesh is too large")
        # end
        # This appends the structMesh and elemConn to the list that we can unpack later
        push!(StructMeshList, structMesh)
        push!(ElemConnList, elemConn)
    end

    return StructMeshList, ElemConnList

end

function make_componentMesh(
    nElem::Int64, span::DTYPE;
    config="wing", rake=0.000, nElStrut=0, spanStrut=0.0
)
    """
    Makes a mesh and element connectivity
    First element is usually origin (x,y,z) = (0,0,0) except the t-foil where the first element is shifted down by spanStrut
    You do not necessarily have to make this mesh yourself every run
    The default is to mesh y as the span

    Inputs
    ------
    nElem:
        number of elements
    config:
        "wing" or "t-foil"
    rake:
        rotation of the foil in degrees where 0.0 is lifting up in 'z' and y is the spanwise direction
        Positive rake is nose up
    Outputs
    -------
    mesh
        (nNodes, nDim) array with wing first, then strut if "t-foil"
    elemConn
        (nElem, nNodesPerElem) array saying which elements hold which nodes
    """
    rot = deg2rad(rake)
    transMat = SolverRoutines.get_rotate3dMat(rot; axis="y")
    if config == "wing"
        nNodeTot = nElem + 1
        nElemTot = nElem
    elseif config == "full-wing"
        nNodeTot = 2 * nElem + 1
        nElemTot = 2 * nElem
    elseif config == "t-foil"
        nNodeTot = 2 * nElem + nElStrut + 1
        nElemTot = 2 * nElem + nElStrut
    end
    mesh = zeros(DTYPE, nNodeTot, 3)
    elemConn = zeros(Int64, nElemTot, 2)

    mesh_z = Zygote.Buffer(mesh)
    # mesh_z[:, :] .= mesh
    # elemConn_z[:, :] .= elemConn
    # The transformation matrix is good
    # println("transMat initial")
    # show(stdout, "text/plain", transMat)
    # println("")
    mesh, elemConn = fill_mesh(mesh_z, elemConn, transMat, span, nElem; config=config, nElStrut=nElStrut, spanStrut=spanStrut)

    mesh_z = Zygote.Buffer(mesh)
    if config == "t-foil"
        mesh_z[:, ZDIM] = mesh_z[:, ZDIM] .- spanStrut # translate
        for inode in 1:nNodeTot
            mesh_z[inode, :] = transMat * mesh_z[inode, :]
        end
    end
    # mesh = copy(mesh_z)
    # elemConn = copy(elemConn_z)
    # println("mesh")
    # show(stdout, "text/plain", mesh)
    # println("")

    return mesh, elemConn
end

function fill_mesh(
    mesh, elemConn, transMat, span, nElem::Int64;
    config="wing", nElStrut=0, spanStrut=0.0
)

    if config == "wing"
        # Set up a line mesh
        dl = span / (nElem) # dist btwn nodes
        mesh[:, :] = hcat(
            zeros(nElem + 1), #X
            LinRange(0.0, span, nElem + 1), #Y
            zeros(nElem + 1) #Z
        )
        ChainRulesCore.ignore_derivatives() do
            for ee in 1:nElem
                elemConn[ee, 1] = ee
                elemConn[ee, 2] = ee + 1
            end
        end
    elseif config == "full-wing"
        # Simple meshes starting from junction at zero
        # Mesh foil wing
        dl = span / (nElem) # dist btwn nodes
        foilwingMesh = LinRange(0.0, span, nElem + 1)#collect(0:dl:span)
        if abs(rot) < MEPSLARGE # no rotation, just a straight wing
            elemCtr = 1 # elem counter
            nodeCtr = 1 # node counter traversing nodes

            # ************************************************
            #     Wing mesh
            # ************************************************
            # Add foil wing first
            for nodeIdx in 1:nElem
                mesh[nodeCtr, :] = [0.0, foilwingMesh[nodeIdx], 0.0]
                ChainRulesCore.ignore_derivatives() do
                    elemConn[nodeCtr, 1] = nodeIdx
                    elemConn[nodeCtr, 2] = nodeIdx + 1
                end
                elemCtr += 1
                nodeCtr += 1
            end

            # Grab end of wing
            mesh[nodeCtr, :] = [0.0, foilwingMesh[end], 0.0]
            nodeCtr += 1

            # Mirror wing nodes skipping first, but adding junction connectivity
            ChainRulesCore.ignore_derivatives() do
                elemConn[elemCtr, 1] = 1
                elemConn[elemCtr, 2] = nodeCtr
            end
            for nodeIdx in 2:nElem
                mesh[nodeCtr, :] = [0.0, -foilwingMesh[nodeIdx], 0.0]
                ChainRulesCore.ignore_derivatives() do
                    elemConn[nodeCtr, 1] = nodeCtr
                    elemConn[nodeCtr, 2] = nodeCtr + 1
                end
                elemCtr += 1
                nodeCtr += 1
            end

            # Grab end of wing
            mesh[nodeCtr, :] = [0.0, -foilwingMesh[end], 0.0]
            ChainRulesCore.ignore_derivatives() do
                elemConn[elemCtr, 1] = nodeCtr - 1
                elemConn[elemCtr, 2] = nodeCtr
            end
            nodeCtr += 1
            elemCtr += 1

            # in the extreme case of 3 elements, elem conn is wrong
            ChainRulesCore.ignore_derivatives() do
                if (2 * nElem == 2)
                    elemConn[2, 1] = 1
                    elemConn[2, 2] = 3
                end
            end
        end

    elseif config == "t-foil"

        # Simple meshes starting from junction at zero
        # Mesh foil wing
        dl = span / (nElem) # dist btwn nodes
        # foilwingMesh = collect(0:dl:span)
        foilwingMesh = LinRange(0, span, nElem + 1)
        # Mesh strut
        dlStrut = spanStrut / (nElStrut)
        # strutMesh = collect(dlStrut:dlStrut:spanStrut) # don't start at zero since it already exists
        strutMesh = LinRange(dlStrut, spanStrut, nElStrut)
        # This is basically avoiding double counting the nodes

        # ************************************************
        #     Wing mesh
        # ************************************************
        elemCtr = 1 # elem counter
        nodeCtr = 1 # node counter traversing nodes

        # Add foil wing first
        ChainRulesCore.ignore_derivatives() do
            for nodeIdx in 1:nElem
                # mesh[nodeCtr, :] = [0.0, foilwingMesh[nodeIdx], 0.0]
                elemConn[nodeCtr, 1] = nodeIdx
                elemConn[nodeCtr, 2] = nodeIdx + 1
                elemCtr += 1
                nodeCtr += 1
            end
        end

        # Grab end of wing
        foilMesh = hcat(
            zeros(nElem + 1),
            foilwingMesh,
            zeros(nElem + 1)
        )
        # mesh[nodeCtr, :] = [0.0, foilwingMesh[end], 0.0]
        nodeCtr += 1

        # Mirror wing nodes skipping first, but adding junction connectivity
        ChainRulesCore.ignore_derivatives() do
            elemConn[elemCtr, 1] = 1
            elemConn[elemCtr, 2] = nodeCtr
            for nodeIdx in 2:nElem
                # mesh[nodeCtr, :] = [0.0, -foilwingMesh[nodeIdx], 0.0]
                elemConn[nodeCtr, 1] = nodeCtr
                elemConn[nodeCtr, 2] = nodeCtr + 1
                elemCtr += 1
                nodeCtr += 1
            end
        end

        foilMeshPort = hcat(zeros(nElem), -foilwingMesh[2:end], zeros(nElem))
        # Grab end of wing
        # mesh[nodeCtr, :] = [0.0, -foilwingMesh[end], 0.0]
        ChainRulesCore.ignore_derivatives() do
            elemConn[elemCtr, 1] = nodeCtr - 1
            elemConn[elemCtr, 2] = nodeCtr
        end
        nodeCtr += 1
        elemCtr += 1

        # ************************************************
        #     Strut mesh
        # ************************************************
        # Add strut going up in z
        nodeIdx = 1
        ChainRulesCore.ignore_derivatives() do
            for istrut in 1:nElStrut # loop elem, not nodes
                # if nodeIdx <= nElStrut 
                # mesh[nodeCtr, 1:3] = [0.0, 0.0, strutMesh[istrut]]
                if nodeIdx == 1
                    elemConn[elemCtr, 1] = 1
                    elemConn[elemCtr, 2] = nodeCtr
                else
                    elemConn[elemCtr, 1] = nodeCtr - 1
                    elemConn[elemCtr, 2] = nodeCtr
                end
                # end
                nodeIdx += 1
                elemCtr += 1
                nodeCtr += 1
            end
        end

        strutMesh = hcat(
            zeros(nElStrut), # XDIM
            zeros(nElStrut), # YDIM
            strutMesh # ZDIM
        )
        mesh[:, :] = vcat(foilMesh, foilMeshPort, strutMesh)

        # in the extreme case of 3 elements, elem conn is wrong
        ChainRulesCore.ignore_derivatives() do
            if (2 * nElem + nElStrut == 3)
                elemConn[2, 1] = 1
                elemConn[2, 2] = 3
            end
        end


    end

    # ************************************************
    #     Lastly, translate, and rotate the mesh
    # ************************************************
    if config == "t-foil"
        mesh[:, ZDIM] = mesh[:, ZDIM] .- spanStrut # translate
    end

    if transMat != Matrix(I, 3, 3)
        for inode in eachindex(mesh[:, 1])
            # println("pre-transformed pt", meshCopy[inode, :])
            mesh[inode, :] = transMat * mesh[inode, :]
            # println("translated point", transMat * meshCopy[inode, :])
        end
    end
    meshCopy = copy(mesh)

    return meshCopy, elemConn

end


function rotate3d(dataVec, rot; axis="x")
    """
    Rotates a 3D vector about axis by rot radians (RH rule!)
    """
    # rotMat = zeros(Float64, 3, 3)
    # rotMat = @SMatrix zeros(Float64, 3, 3)
    c = cos(rot)
    s = sin(rot)
    if axis == "x"
        rotMat = [
            1 0 0
            0 c -s
            0 s c
        ]
    elseif axis == "y"
        rotMat = [
            c 0 s
            0 1 0
            -s 0 c
        ]
    elseif axis == "z"
        rotMat = [
            c -s 0
            s c 0
            0 0 1
        ]
    else
        println("Only axis rotation implemented")
    end
    transformedVec = rotMat * dataVec
    return transformedVec
end

function assemble(
    StructMesh::StructMesh, abVec::Vector{DTYPE}, x_αbVec::Vector{DTYPE},
    FOIL::DynamicFoil, elemType="bend-twist", constitutive="isotropic";
    config="wing", STRUT=nothing, ab_strut=nothing, x_αb_strut=nothing, verbose=true
)
    """
    Generic function to assemble the global mass and stiffness matrices

        coordMat: 
        elemConn: 2D array of element connectivity (nElem x 2)
    Outputs
    -------
        globalK: global stiffness matrix
        globalM: global mass matrix
        globalF: global force vector
    """

    # --- Local nodal DOF vector ---
    # Determine the number of dofs per node
    qLocal = zeros(DTYPE, NDOF * 2)

    abVec = FOIL.ab
    # x_αbVec = StructMesh.x_αb
    # --- Initialize matrices ---
    nElem = size(StructMesh.elemConn)[1]
    nNodes = nElem + 1
    globalK = zeros(DTYPE, NDOF * (nNodes), NDOF * (nNodes))
    globalM = zeros(DTYPE, NDOF * (nNodes), NDOF * (nNodes))
    globalF = zeros(DTYPE, NDOF * (nNodes))
    # Note: sparse arrays does not work through Zygote without some workarounds (that I haven't figured out yet)
    # globalK::SparseMatrixCSC{Float64,Int64} = spzeros(nnd * (nNodes), nnd * (nNodes))

    # ************************************************
    #     Element loop
    # ************************************************
    # --- Zygote buffer initializations ---
    globalK_z = Zygote.Buffer(globalK)
    globalM_z = Zygote.Buffer(globalM)
    globalF_z = Zygote.Buffer(globalF)
    globalK_z[:, :] = globalK
    globalM_z[:, :] = globalM
    globalF_z[:] = globalF
    populate_matrices!(globalK_z, globalM_z, globalM_z, nElem, StructMesh, FOIL, STRUT, abVec, x_αbVec;
        config=config, constitutive=constitutive, verbose=verbose, elemType=elemType, ab_strut=ab_strut, x_αb_strut=x_αb_strut)


    globalK = copy(globalK_z)
    globalM = copy(globalM_z)
    globalF = copy(globalF_z)

    return globalK, globalM, globalF
end

function populate_matrices!(
    globalK, globalM, globalF,
    nElem::Int64, StructMesh, FOIL::DynamicFoil, STRUT::DynamicFoil, abVec::Vector, x_αbVec::Vector;
    config="wing", constitutive="isotropic", verbose=true, elemType="bend-twist", ab_strut=nothing, x_αb_strut=nothing
)
    nNodes::Int64 = nElem + 1
    elemConn = StructMesh.elemConn
    coordMat = StructMesh.mesh
    # --- Debug printout for initialization ---
    ChainRulesCore.ignore_derivatives() do
        if verbose
            println("+----------------------------------------+")
            println("|        Assembling beam matrices        |")
            println("+----------------------------------------+")
            println("Constitutive law: ", constitutive)
            println("Beam element: ", elemType)
            println("Elements: ", nElem)
            println("Nodes: ", nNodes)
            println("DOFs: ", NDOF * nNodes)
        end
    end

    nElemWing = (FOIL.nNodes - 1)

    for elemIdx ∈ 1:nElem
        # ---------------------------
        #   Extract element info
        # ---------------------------
        n1 = elemConn[elemIdx, 1]
        n2 = elemConn[elemIdx, 2]
        dR1 = (coordMat[n2, XDIM] - coordMat[n1, XDIM])
        dR2 = (coordMat[n2, YDIM] - coordMat[n1, YDIM])
        dR3 = (coordMat[n2, ZDIM] - coordMat[n1, ZDIM])
        lᵉ = sqrt(dR1^2 + dR2^2 + dR3^2) # length of elem
        nVec = [dR1, dR2, dR3] / lᵉ # normalize
        if elemIdx <= nElemWing
            EIₛ = FOIL.EIₛ[elemIdx]
            EIIPₛ = FOIL.EIIPₛ[elemIdx]
            EAₛ = FOIL.EAₛ[elemIdx]
            GJₛ = FOIL.GJₛ[elemIdx]
            Kₛ = FOIL.Kₛ[elemIdx]
            Sₛ = FOIL.Sₛ[elemIdx]
            mₛ = FOIL.mₛ[elemIdx]
            iₛ = FOIL.Iₛ[elemIdx]
            # These are currently DVs
            ab = abVec[elemIdx]
            x_αb = x_αbVec[elemIdx]
        else
            if config == "t-foil"
                if nElemWing < elemIdx <= 2 * nElemWing # half-wing
                    wingElem = elemIdx - nElemWing
                    EIₛ = FOIL.EIₛ[wingElem]
                    EIIPₛ = FOIL.EIIPₛ[wingElem]
                    EAₛ = FOIL.EAₛ[wingElem]
                    GJₛ = FOIL.GJₛ[wingElem]
                    Kₛ = -FOIL.Kₛ[wingElem] # negative to account for the opposite sign in the strut
                    Sₛ = FOIL.Sₛ[wingElem]
                    mₛ = FOIL.mₛ[wingElem]
                    iₛ = FOIL.Iₛ[wingElem]
                    # These are currently DVs
                    ab = abVec[wingElem]
                    x_αb = x_αbVec[wingElem]
                elseif elemIdx > 2 * nElemWing # strut
                    strutElem = elemIdx - 2 * nElemWing
                    EIₛ = STRUT.EIₛ[strutElem]
                    EIIPₛ = STRUT.EIIPₛ[strutElem]
                    EAₛ = STRUT.EAₛ[strutElem]
                    GJₛ = STRUT.GJₛ[strutElem]
                    Kₛ = STRUT.Kₛ[strutElem]
                    Sₛ = STRUT.Sₛ[strutElem]
                    mₛ = STRUT.mₛ[strutElem]
                    iₛ = STRUT.Iₛ[strutElem]
                    # These are currently DVs
                    ab = ab_strut[strutElem]
                    x_αb = x_αb_strut[strutElem]
                end
            elseif config == "full-wing"
                if elemIdx > nElemWing # port wing
                    wingElem = elemIdx - nElemWing
                    EIₛ = FOIL.EIₛ[wingElem]
                    EIIPₛ = FOIL.EIIPₛ[wingElem]
                    EAₛ = FOIL.EAₛ[wingElem]
                    GJₛ = FOIL.GJₛ[wingElem]
                    Kₛ = -FOIL.Kₛ[wingElem] #MATERIAL COUPLING MUST BE NEGATED (I debugged this)
                    Sₛ = FOIL.Sₛ[wingElem]
                    mₛ = FOIL.mₛ[wingElem]
                    iₛ = FOIL.Iₛ[wingElem]
                    ab = abVec[wingElem]
                    x_αb = x_αbVec[wingElem]
                end
            end
        end

        # ---------------------------
        #   Local stiffness matrix
        # ---------------------------
        kLocal = BeamElement.compute_elem_stiff(EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, lᵉ, ab, elemType, constitutive, false)

        # ---------------------------
        #   Local mass matrix
        # ---------------------------
        mLocal = BeamElement.compute_elem_mass(mₛ, iₛ, lᵉ, x_αb, elemType)

        # ---------------------------
        #   Local force vector
        # ---------------------------
        fLocal = zeros(DTYPE, NDOF * 2)

        # ---------------------------
        #   Transform from local to global coordinates
        # ---------------------------
        #  AEROSP510 notes and python code, Engineering Vibration Chapter 8 (Inman 2014)
        # The local coordinate system is {u} while the global is {U}
        # {u} = [Γ] * {U}
        # where [Γ] is the transformation matrix
        Γ = SolverRoutines.get_transMat(dR1, dR2, dR3, lᵉ, elemType)
        kElem = Γ' * kLocal * Γ
        mElem = Γ' * mLocal * Γ
        fElem = Γ' * fLocal
        ChainRulesCore.ignore_derivatives() do
            if any(isnan.(kElem))
                println("NaN in elem stiffness matrix")
            end
        end

        # ---------------------------
        #   Assemble into global matrices
        # ---------------------------
        # The following procedure generally follows:
        @inbounds @fastmath begin
            for nodeIdx ∈ 1:2 # loop over nodes in element
                for dofIdx ∈ 1:NDOF # loop over DOFs in node
                    idxRow = ((elemConn[elemIdx, nodeIdx] - 1) * NDOF + dofIdx) # idx of global dof (row of global matrix)
                    idxRowₑ = (nodeIdx - 1) * NDOF + dofIdx # idx of dof within this element

                    # --- Assemble RHS ---
                    globalF[idxRow] = fElem[idxRowₑ]

                    # --- Assemble LHS ---
                    for nodeColIdx ∈ 1:2 # loop over nodes in element
                        for dofColIdx ∈ 1:NDOF # loop over DOFs in node
                            idxCol = (elemConn[elemIdx, nodeColIdx] - 1) * NDOF + dofColIdx # idx of global dof (col of global matrix)
                            idxColₑ = (nodeColIdx - 1) * NDOF + dofColIdx # idx of dof within this element (column)

                            globalK[idxRow, idxCol] += kElem[idxRowₑ, idxColₑ]
                            globalM[idxRow, idxCol] += mElem[idxRowₑ, idxColₑ]
                        end
                    end
                end
            end
        end
        ChainRulesCore.ignore_derivatives() do
            if any(isnan.(globalK))
                println("NaN in global stiffness matrix")
            end
        end
    end

end


function get_fixed_dofs(elemType::String, BCCond="clamped"; appendageOptions=Dict("config" => "wing"), verbose=true)
    """
    Depending on the elemType, return the indices of fixed nodes
    """
    if BCCond == "clamped"
        if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
            fixedDOFs = Vector(1:BeamElement.NDOF)
        elseif appendageOptions["config"] == "t-foil"
            nElemTot = (appendageOptions["nNodes"] - 1) * 2 + appendageOptions["nNodeStrut"] - 1
            nNodeTot = nElemTot + 1
            fixedDOFs = Vector(nNodeTot*BeamElement.NDOF:-1:nElemTot*BeamElement.NDOF+1)

        else
            error("config not recognized")
        end

    else
        error("BCCond not recognized")
    end

    ChainRulesCore.ignore_derivatives() do
        if verbose
            println("BCType: ", BCCond)
        end
    end

    return fixedDOFs
end

function apply_tip_load!(globalF, elemType, transMat, loadType="force"; solverOptions=Dict("config" => "wing"))
    """
    Routine for applying unit tip load to the end node
        globalF
            vector
        elemType: str
        transMat: 2d array
            Transformation matrix from local into global coordinates
    """

    MAG = 1.0
    m, n = size(transMat)

    if loadType == "force"
        if elemType == "bend"
            FLocalVec = [1.0, 0.0]
        elseif elemType == "bend-twist"
            FLocalVec = [1.0, 0.0, 0.0]
        elseif elemType == "BT2"
            FLocalVec = [1.0, 0.0, 0.0, 0.0]
        elseif elemType == "BEAM3D"
            FLocalVec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        elseif elemType == "COMP2"
            FLocalVec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            error("element not defined")
        end
    elseif loadType == "torque"
        if elemType == "bend-twist"
            FLocalVec = [0.0, 0.0, 1.0]
        elseif elemType == "BT2"
            FLocalVec = [0.0, 0.0, 1.0, 0.0]
        elseif elemType == "BEAM3D"
            FLocalVec = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elseif elemType == "COMP2"
            FLocalVec = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            error("element not defined")
        end
    end

    # --- Transform to global then add into vector ---
    if elemType == "COMP2"
        FLocalVec = transMat[1:m÷2, 1:n÷2]' * FLocalVec
    end
    if solverOptions["config"] == "wing"
        globalF[end-NDOF+1:end] += FLocalVec * MAG
    elseif solverOptions["config"] == "full-wing" || solverOptions["config"] == "t-foil"
        nElemWing = solverOptions["nNodes"] - 1
        # STBD wing
        inodestart = nElemWing + 1
        idofstart = (inodestart - 1) * NDOF + 1
        globalF[idofstart:idofstart+NDOF-1] += FLocalVec * MAG
        # Port wing
        inodestart = nElemWing * 2 + 1
        idofstart = (inodestart - 1) * NDOF + 1
        globalF[idofstart:idofstart+NDOF-1] += FLocalVec * MAG
    else
        error("config not recognized")

    end

end

function apply_tip_mass(globalM, mass, inertia, elemLength, x_αbBulb, transMat, elemType="BT2")
    """
    Apply a tip mass to the global mass matrix

    mass: mass of the tip [kg]
    inertia: moment of inertia of the tip about C.G. [kg-m^2]
    """

    globalM_z = Zygote.Buffer(globalM)
    globalM_z[:, :] = globalM
    # TODO: unit test for the tip mass, verification case from Eirikur's thesis?
    if elemType == "bend-twist"
        println("Does not work")
    elseif elemType == "BT2"
        nDOF = 8
        # --- Get sectional properties ---
        ms = mass / elemLength
        # Parallel axis theorem
        Iea = inertia + mass * (x_αbBulb)^2
        is = Iea / elemLength
        tipMassMat = BeamElement.compute_elem_mass(ms, is, elemLength, x_αbBulb, elemType)
        tipMassMat = transMat' * tipMassMat * transMat

        # --- Assemble into global matrix ---
        globalM_z[end-nDOF+1:end, end-nDOF+1:end] += tipMassMat
    elseif elemType == "COMP2"
        nDOF = 18
        # --- Get sectional properties ---
        ms = mass / elemLength
        # Parallel axis theorem
        Iea = inertia + mass * (x_αbBulb)^2
        is = Iea / elemLength
        tipMassMat = BeamElement.compute_elem_mass(ms, is, elemLength, x_αbBulb, elemType)
        tipMassMat = transMat' * tipMassMat * transMat

        # --- Assemble into global matrix ---
        globalM_z[end-nDOF+1:end, end-nDOF+1:end] += tipMassMat
    else
        error("Not implemented")
    end

    ChainRulesCore.ignore_derivatives() do
        println("+------------------------------------+")
        println("|    Tip mass added!                 |")
        println("+------------------------------------+")
        println("Dist. CG is aft of EA: ", x_αbBulb, " [m]")
    end
    # NOTE: After adding the tip mass, the matrix is no longer positive definite... bug?
    # println("after tip mass:",eigvals(copy(globalM_z)))
    return copy(globalM_z)
end

function apply_inertialLoad!(globalF; gravityVector=[0.0, 0.0, -9.81])
    """
    Applies inertial load and modifies globalF
    """

    println("Adding inertial loads to FEM with gravity vector of", gravityVector)

    # TODO: add gravity vector
end

function apply_BCs(
    K::Matrix, M::Matrix,
    F::Vector,
    globalDOFBlankingList::Vector{Int64}
)
    """
    Applies BCs for nodal displacements and sets them to zero
    """
    # Old method of deleting
    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newF = F[1:end.∉[globalDOFBlankingList]]

    # newK = copy(K)
    # newK[globalDOFBlankingList, :] .= 0.0
    # newK[:, globalDOFBlankingList] .= 0.0
    # newM = copy(M)
    # newM[globalDOFBlankingList, :] .= 0.0
    # newM[:, globalDOFBlankingList] .= 0.0
    # newF = copy(F)
    # newF[globalDOFBlankingList] .= 0.0

    return newK, newM, newF
end

function put_BC_back(q, elemType::String, BCType="clamped"; appendageOptions=Dict("config" => "wing"))
    """
    appends the BCs back into the solution
    """

    if BCType == "clamped"
        if elemType == "BT2"
            uSol = vcat([0, 0, 0, 0], q)
        elseif elemType == "COMP2"
            if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
                uSol = vcat(zeros(9), q)
            elseif appendageOptions["config"] == "t-foil"
                uSol = vcat(q, zeros(9))
            else
                error("Unsuppported config")
            end
        else
            println("Not working")
            exit()
        end
    else
        println("Not working")
    end

    return uSol, length(uSol)
end

function solve_structure(K::Matrix, M::Matrix, F::Vector)
    """
    Solve the structural system
    """

    q = (K) \ F # TODO: should probably replace this with an iterative solver

    return q
end

function compute_modal(K::Matrix, M::Matrix, nEig::Int64)
    """
    Compute the eigenvalues (natural frequencies) and eigenvectors (mode shapes) of the in-vacuum system.
    i.e., this is structural dynamics, not hydroelastics.

    The eigenvalues should be all positive and real
    """

    # use krylov method to get first few smallest eigenvalues
    # Solve [K]{x} = λ[M]{x} where λ = ω²
    eVals, eVecs = SolverRoutines.compute_eigsolve(K, M, nEig)

    naturalFreqs = sqrt.(eVals) / (2π)

    return naturalFreqs, eVecs
end

function compute_proportional_damping(K::Matrix, M::Matrix, ζ::RealOrComplex, nMode::Int64)
    """
    TODO: MAKE SURE THIS CONSTANT STAYS THE SAME THROUGHOUT OPTIMIZATION
    Compute the proportional (Rayleigh) damping matrix

    C = alpha [M] + beta [K]

    where alpha and beta are the mass- and stiffness-
    proportional damping coefficients, respectively.


    Inputs
    ------
        K: stiffness matrix after BCs applied
        M: mass matrix after BCs applied
        Zeta: damping ratio at the first min and max natural frequencies [-]
        OR
        damping ratio at the first natural frequency (vacuum)
    """

    # --- Compute undamped natural frequencies ---
    fns, _ = compute_modal(K, M, nMode)
    ω₁ = fns[1] * 2π
    ω₂ = fns[nMode] * 2π

    # --- Compute proportional coefficients ---
    # # Mass and stiffness proportional damping
    # All damping in between the first two natural frequencies is not overdamped artificially
    # massPropConst = 2 * ζ / (ω₁ + ω₂)
    # stiffPropConst = ω₁ * ω₂ * massPropConst

    # Stiffness proportional damping (grows with increasing response freq)
    massPropConst = 0.0
    stiffPropConst = 2 * ζ / ω₂

    # # Mass proportional damping
    # massPropConst = 2 * ζ * ω₁
    # stiffPropConst = 0.0

    return massPropConst, stiffPropConst
end


end # end module
