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
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using DelimitedFiles
using LinearAlgebra
using StaticArrays

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex, DTYPE
using ..EBBeam: EBBeam as BeamElement, NDOF
using ..SolverRoutines
using ..BeamProperties
using ..DesignConstants: DesignConstants, DynamicFoil, CONFIGS
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE, ELEMTYPE
using ..Preprocessing
using ..MaterialLibrary
using ..Rotations

struct StructMesh{TF,TC,TI}
    """
    Struct to hold the mesh, element connectivity, and node properties
    """
    mesh::AbstractMatrix{TF} # node xyz coords (2D array of coordinates of nodes)
    elemConn::Matrix{TI} # element-node connectivity [elemIdx] => [globalNode1Idx, globalNode2Idx]
    # sectionVectors::TM
    # The stuff below is only stored for output file writing. DO NOT USE IN CALCULATIONS
    chord::Vector{TC}
    toc::Vector{TC}
    ab::Vector{TC}
    x_αb::Vector{TC}
    theta_f::TC # global fiber frame orientation
    idxTip::TI # index of the tip node
    airfoilCoords::AbstractMatrix # airfoil coordinates
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

function make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)
    """

    Replaces make_componentMesh()

    midchords: Location of midchords from the grid file
    nodeConn: Connectivity of the midchords from the grid file

    """
    config = appendageOptions["config"]
    semispan = Preprocessing.compute_structSpan(abs.(midchords), idxTip)
    nElemWing = appendageOptions["nNodes"] - 1
    nElemTot = nothing
    if config == "wing"
        nElemTot = nElemWing
    elseif config == "full-wing"
        nElemTot = 2 * nElemWing
    end
    nNodeTot = nElemTot + 1
    nNodeWing = nElemWing + 1



    # --- Spline quantities ---
    s_loc_q = LinRange(0.0, semispan, nNodeTot)
    if config == "full-wing"
        dx = semispan / nElemWing
        s_loc_q = vcat(LinRange(0, semispan, nNodeWing), LinRange(-dx, -semispan, nNodeWing - 1))
    elseif !(config in CONFIGS)
        error("Invalid configuration")
    end

    # Find where node connectivivity jumps and has index of 1
    junctionIdxs = findall(x -> x == 1, nodeConn)

    s_loc = vec(sqrt.(sum(midchords .^ 2, dims=1))) .* sign.(midchords[YDIM, :])


    midchordXLocs = SolverRoutines.do_linear_interp(s_loc, midchords[XDIM, :], s_loc_q)
    midchordYLocs = SolverRoutines.do_linear_interp(s_loc, midchords[YDIM, :], s_loc_q)
    midchordZLocs = SolverRoutines.do_linear_interp(s_loc, midchords[ZDIM, :], s_loc_q)

    # mesh = zeros(RealOrComplex, nNodeTot, 3)
    mesh = cat(reshape(midchordXLocs, nNodeTot, 1), reshape(midchordYLocs, nNodeTot, 1), reshape(midchordZLocs, nNodeTot, 1), dims=2)
    # mesh[:, XDIM] = midchordXLocs
    # mesh[:, YDIM] = midchordYLocs
    # mesh[:, ZDIM] = midchordZLocs

    elemConn = zeros(Int64, nElemWing, 2)
    elemConn_z = Zygote.Buffer(elemConn)
    # --- Element connectivity ---
    for ee in 1:nElemWing
        elemConn_z[ee, :] = [ee, ee + 1]
    end
    elemConn = copy(elemConn_z)

    # println("span loc query", s_loc_q)
    # for coord in eachrow(mesh)
    #     println(coord)
    # end
    if config == "full-wing"
        modifiedConn = elemConn .+ nNodeWing .- 1
        modifiedConn_z = Zygote.Buffer(modifiedConn)
        modifiedConn_z[:, :] = modifiedConn
        modifiedConn_z[1, 1] = 1
        modifiedConn = copy(modifiedConn_z)
        elemConn = vcat(elemConn, modifiedConn)
    end

    # println("elemConn: ", elemConn)
    return mesh, elemConn

end

function make_componentMesh(
    nElem::Int64, span;
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
    # transMat = SolverRoutines.get_rotate3dMat(rot; axis="y")
    transMat = Rotations.get_rotate3dMat(rot, "y")
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
    mesh = zeros(RealOrComplex, nNodeTot, 3)
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
    mesh_z[:, :] = mesh
    if config == "t-foil"
        for inode in 1:nNodeTot
            mesh_z[inode, ZDIM] = mesh_z[inode, ZDIM] - spanStrut # translate
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
        for ee in 1:nElem
            elemConn[ee, 1] = ee
            elemConn[ee, 2] = ee + 1
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
                @ignore_derivatives() do
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
            @ignore_derivatives() do
                elemConn[elemCtr, 1] = 1
                elemConn[elemCtr, 2] = nodeCtr
            end
            for nodeIdx in 2:nElem
                mesh[nodeCtr, :] = [0.0, -foilwingMesh[nodeIdx], 0.0]
                @ignore_derivatives() do
                    elemConn[nodeCtr, 1] = nodeCtr
                    elemConn[nodeCtr, 2] = nodeCtr + 1
                end
                elemCtr += 1
                nodeCtr += 1
            end

            # Grab end of wing
            mesh[nodeCtr, :] = [0.0, -foilwingMesh[end], 0.0]
            @ignore_derivatives() do
                elemConn[elemCtr, 1] = nodeCtr - 1
                elemConn[elemCtr, 2] = nodeCtr
            end
            nodeCtr += 1
            elemCtr += 1

            # in the extreme case of 3 elements, elem conn is wrong
            @ignore_derivatives() do
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
        @ignore_derivatives() do
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
        @ignore_derivatives() do
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
        @ignore_derivatives() do
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
        @ignore_derivatives() do
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
        @ignore_derivatives() do
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



function assemble(StructMesh, x_αbVec,
    FOIL, elemType="bend-twist", constitutive="isotropic";
    config="wing", STRUT=nothing, x_αb_strut=nothing, verbose=true
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
    abVec = FOIL.ab
    if config == "t-foil"
        ab_strut = STRUT.ab
    else
        ab_strut = nothing
    end
    # x_αbVec = StructMesh.x_αb
    # --- Initialize matrices ---
    nElem = size(StructMesh.elemConn)[1]
    nNodes = nElem + 1
    globalK = zeros(RealOrComplex, NDOF * (nNodes), NDOF * (nNodes))
    globalM = zeros(RealOrComplex, NDOF * (nNodes), NDOF * (nNodes))
    globalF = zeros(RealOrComplex, NDOF * (nNodes))
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
    nElem::Int64, StructMesh, FOIL, STRUT, abVec, x_αbVec;
    config="wing", constitutive="isotropic", verbose=true, elemType="bend-twist", ab_strut=nothing, x_αb_strut=nothing
)
    nNodes::Int64 = nElem + 1
    elemConn = StructMesh.elemConn
    coordMat = StructMesh.mesh
    # --- Debug printout for initialization ---
    @ignore_derivatives() do
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

        # println("elem: $(elemIdx)\t coords", coordMat[n1, :])
        lᵉ = √(dR1^2 + dR2^2 + dR3^2) # length of elem
        # nVec = [dR1, dR2, dR3] / lᵉ # normalize
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
            else
                error("config not recognized")
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
        # Γ = SolverRoutines.get_transMat(dR1, dR2, dR3, lᵉ, elemType)
        Γ = Rotations.get_transMat(dR1, dR2, dR3, lᵉ)
        ΓT = transpose(Γ)
        kElem = ΓT * kLocal * Γ
        mElem = ΓT * mLocal * Γ
        fElem = ΓT * fLocal
        @ignore_derivatives() do
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
        @ignore_derivatives() do
            if any(isnan.(globalK))
                println("NaN in global stiffness matrix")
            end
        end
    end

end


function get_fixed_dofs(elemType::String, BCCond="clamped"; appendageOptions=Dict("config" => "wing"), verbose=false)
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

    @ignore_derivatives() do
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

    @ignore_derivatives() do
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
    TODO: add gravity vector
    """

    println("Adding inertial loads to FEM with gravity vector of", gravityVector)

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

    # q = (K) \ F # TODO: should probably replace this with an iterative solver
    q = inv(K) * F

    return q
end

function init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut,
    appendageParams, appendageOptions, solverOptions)
    """
    similar to above but shortcircuiting the hydroside
    """

    idxTip = Preprocessing.get_tipnode(real.(LECoords))
    midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    # ---------------------------
    #   Geometry
    # ---------------------------
    eb::Vector{RealOrComplex} = 0.25 * chordLengths .+ ab
    t::Vector{RealOrComplex} = toc .* chordLengths

    # ---------------------------
    #   Structure
    # ---------------------------
    ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(appendageOptions["material"])

    # --- Compute the structural properties for the foil ---
    nNodes = appendageOptions["nNodes"]

    if haskey(appendageOptions, "path_to_struct_props") && !isnothing(appendageOptions["path_to_struct_props"])
        println("Reading structural properties from file: ", appendageOptions["path_to_struct_props"])
        EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = Preprocessing.get_1DBeamPropertiesFromFile(appendageOptions["path_to_struct_props"])
    else
        EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodes, chordLengths, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f, constitutive; solverOptions=solverOptions)
    end

    # ---------------------------
    #   Build final model
    # ---------------------------
    wingModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, eb, ab, chordLengths, appendageOptions["nNodes"], constitutive)

    # ************************************************
    #     Strut properties
    # ************************************************
    if appendageOptions["config"] == "t-foil" && !solverOptions["use_nlll"]
        c_strut = chordLengths # TODO fix me later
        # Do it again using the strut properties
        nNodesStrut = appendageOptions["nNodeStrut"]
        ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive = MaterialLibrary.return_constitutive(appendageOptions["strut_material"])
        t_strut = toc_strut .* c_strut
        eb_strut = 0.25 * c_strut .+ ab_strut

        EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ = BeamProperties.compute_beam(nNodesStrut, c_strut, t_strut, ab_strut, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f_strut, constitutive; solverOptions=solverOptions)

        # ---------------------------
        #   Build final model
        # ---------------------------
        strutModel = DesignConstants.Foil(mₛ, Iₛ, EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, eb_strut, ab_strut, c_strut, appendageOptions["nNodeStrut"], constitutive)

    elseif appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
        strutModel = nothing
    elseif !(appendageOptions["config"] in CONFIGS)
        error("Unsupported config: ", appendageOptions["config"])
    end


    return wingModel, strutModel

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

    naturalFreqs = .√(eVals) / (2π)

    return naturalFreqs, eVecs
end


function compute_proportionalDampingConstants(FEMESH, x_αbVec, FOIL, elemType, appendageParams, appendageOptions, solverOptions)
    """
    A routine to be called in flutter initialization to set α and β for proportional damping
    """

    globalKs, globalMs, globalF = assemble(FEMESH, x_αbVec, FOIL, elemType, FOIL.constitutive; config=appendageOptions["config"])

    # It's OK to assume clamped BCs for the damping computation
    globalDOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped"; appendageOptions=appendageOptions)

    Ks, Ms, _ = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

    αStruct, βStruct = FEMMethods.compute_proportional_damping(Ks, Ms, appendageParams["zeta"], solverOptions["nModes"])

    return αStruct, βStruct
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


    println("Natural frequencies for struct. damping:\nMode\t[Hz]")
    for (ii, fn) in enumerate(fns)
        println("$(ii)\t$(fn)")
    end

    return massPropConst, stiffPropConst
end



end # end module
