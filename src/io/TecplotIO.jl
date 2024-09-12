# --- Julia 1.7---
"""
@File    :   TecplotIO.jl
@Time    :   2023/03/05
@Author  :   Galen Ng
@Desc    :   Interface for writing an output Tecplot can read
When in doubt, refer to the Tecplot Data Format Guide
"""


module TecplotIO
# --- PACKAGES ---
using Printf

# --- DCFoil modules ---
using ..SolverRoutines: get_rotate3dMat
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..EBBeam: UIND, VIND, WIND, ΦIND, ΘIND, ΨIND, NDOF
using ..Utilities: Utilities

function write_mesh(DVDict::Dict, FEMESHLIST, solverOptions::Dict, outputDir::String, fname="mesh.dat")
    """
    Top level routine to write the structural mesh file
    """

    outfile = @sprintf("%s%s", outputDir, fname)
    @printf("Writing mesh file %s...\n", outfile)

    io = open(outfile, "w")
    write(io, "TITLE = \"Mesh Data\"\n")
    write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \n")

    for icomp in eachindex(FEMESHLIST)
        options = solverOptions["appendageList"][icomp]

        FEMESH = FEMESHLIST[icomp]

        # Offset by xmount just for the visualization purpose. 
        # Solution still happens in the local foil frame
        FEMESHCopy = deepcopy(FEMESH)
        FEMESHCopy.mesh[:, XDIM] .+= options["xMount"]
        write_1Dfemmesh(io, FEMESHCopy)
        # write_strips(io, DVDict, FEMESH; config=solverOptions["config"], nNodeWing=options["nNodes"])
        # write_oml(io, DVDict, FEMESH; config=solverOptions["config"], nNodeWing=options["nNodes"])
    end

    close(io)
end

function write_hydromesh(LLMesh, uvec, outputDir::String, fname="hydromesh.dat")
    """
    Write the lifting line mesh to a tecplot file
    """
    outfile = @sprintf("%s%s", outputDir, fname)
    @printf("Writing hydro mesh file %s...\n", outfile)

    io = open(outfile, "w")
    write(io, "TITLE = \"Hydrodynamic Mesh Data\"\n")
    write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \n")

    write_LLmesh(io, LLMesh, uvec)

    close(io)

end



function transform_airfoil(foilCoords, localChord, pretwist=0.0)
    """
    Unit airfoil to DCFoil frame
    considering pretwist where positive tilts up
    """
    # Translate airfoil to be centered at the midchord
    foilCoordsXform = copy(foilCoords)
    foilCoordsXform[:, XDIM] .+= -0.5
    # Scale airfoil
    foilCoordsXform = localChord * foilCoordsXform
    # 2D rotation matrix
    s = sin(-pretwist)
    c = cos(-pretwist)
    rmat = [
        c -s
        s c
    ]
    # Rotate airfoil
    for ii in 1:size(foilCoordsXform)[1]
        foilCoordsXform[ii, :] = (rmat * foilCoordsXform[ii, :])
    end
    return foilCoordsXform
end

function write_airfoils(io, DVDict::Dict, mesh, u, v, w, phi, theta, psi; appendageOptions=Dict("config" => "wing"))
    """
    TODO generalize to take in a normal vector in spanwise direction
    """

    function write_slice(io, unode, iairfoilpt, foilCoords, nodeLoc::Vector, dus, dvs, dws)
        write(io, @sprintf(
            "%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
            foilCoords[iairfoilpt, XDIM] + nodeLoc[XDIM],
            nodeLoc[YDIM],
            foilCoords[iairfoilpt, YDIM] + nodeLoc[ZDIM],
            unode[1] + dus[iairfoilpt],
            unode[2] + dvs[iairfoilpt],
            unode[3] + dws[iairfoilpt],
            unode[4],
            unode[5],
            unode[6])
        )
    end

    foilCoords = Utilities.generate_naca4dig(DVDict["toc"][1])
    baserake = deg2rad(DVDict["alfa0"])
    rake = deg2rad(DVDict["rake"])
    if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing" || appendageOptions["config"] == "t-foil"
        for ii in 1:appendageOptions["nNodes"] # iterate over span
            nodeLoc = mesh[ii, :]
            localChord = DVDict["c"][ii]
            foilCoordsXform = transform_airfoil(foilCoords, localChord, rake + baserake)

            # Get u, v, w based on rotations
            nAirfoilPts = size(foilCoordsXform)[1]
            # uAirfoil = u[ii] * ones(size(foilCoordsScaled)[1])
            dws = -foilCoordsXform[:, XDIM] * sin(theta[ii]) # airfoil twist
            dvs = foilCoordsXform[:, YDIM] * sin(phi[ii]) # airfoil OOP bend
            dus = -foilCoordsXform[:, XDIM] * sin(psi[ii]) # airfoil IP bend

            # --- Header ---
            write(io, @sprintf("ZONE T = \"Airfoil midchord (%.8f, %.8f, %.8f)\" \n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM]))
            write(io, @sprintf("NODES = %d, ", nAirfoilPts))
            write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nAirfoilPts - 1))
            write(io, "DATAPACKING = POINT\n")
            # --- Values ---
            for jj in 1:nAirfoilPts
                unode = [u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii]]
                write_slice(io, unode, jj, foilCoordsXform, nodeLoc, dus, dvs, dws)
            end
            # --- Connectivities ---
            for jj in 1:nAirfoilPts-1
                write(io, @sprintf("%d\t%d\n", jj, jj + 1))
            end
        end
        if appendageOptions["config"] == "full-wing" || appendageOptions["config"] == "t-foil"
            for ii in appendageOptions["nNodes"]+1:2*appendageOptions["nNodes"]-1 # iterate over span
                nodeLoc = mesh[ii, :]
                localChord = DVDict["c"][ii-appendageOptions["nNodes"]]
                foilCoordsXform = transform_airfoil(foilCoords, localChord, rake + baserake)

                # Get u, v, w based on rotations
                nAirfoilPts = size(foilCoordsXform)[1]
                dws = -foilCoordsXform[:, XDIM] * sin(theta[ii]) # airfoil twist
                dvs = foilCoordsXform[:, YDIM] * sin(phi[ii]) # airfoil OOP bend
                dus = -foilCoordsXform[:, XDIM] * sin(psi[ii]) # airfoil IP bend

                # --- Header ---
                write(io, @sprintf("ZONE T = \"Airfoil midchord (%.8f, %.8f, %.8f)\" \n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM]))
                write(io, @sprintf("NODES = %d, ", nAirfoilPts))
                write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nAirfoilPts - 1))
                write(io, "DATAPACKING = POINT\n")
                # --- Values ---
                for jj in 1:nAirfoilPts
                    unode = [u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii]]
                    write_slice(io, unode, jj, foilCoordsXform, nodeLoc, dus, dvs, dws)
                end
                # --- Connectivities ---
                for jj in 1:nAirfoilPts-1
                    write(io, @sprintf("%d\t%d\n", jj, jj + 1))
                end
            end
        end

        if appendageOptions["config"] == "t-foil"

            foilCoords = Utilities.generate_naca4dig(DVDict["toc_strut"][1])
            for ii in appendageOptions["nNodes"]*2:(appendageOptions["nNodes"]*2+appendageOptions["nNodeStrut"]-2) # iterate over strut
                spanLoc = mesh[ii, :]
                localChord = DVDict["c_strut"][ii-(appendageOptions["nNodes"]*2-1)]
                foilCoordsXform = transform_airfoil(foilCoords, localChord)

                # Get u, v, w based on rotations
                nAirfoilPts = size(foilCoordsXform)[1]
                dws = foilCoordsXform[:, XDIM] * sin(theta[ii]) # airfoil twist
                dvs = foilCoordsXform[:, YDIM] * sin(phi[ii]) # airfoil OOP bend
                dus = -foilCoordsXform[:, XDIM] * sin(psi[ii]) # airfoil IP bend

                # --- Header ---
                write(io, @sprintf("ZONE T = \"Airfoil midchord (%.8f, %.8f, %.8f)\" \n", spanLoc[XDIM], spanLoc[YDIM], spanLoc[ZDIM]))
                write(io, @sprintf("NODES = %d, ", nAirfoilPts))
                write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nAirfoilPts - 1))
                write(io, "DATAPACKING = POINT\n")
                # --- Values ---
                for jj in 1:nAirfoilPts
                    # THIS PART CHANGED BECAUSE THE STRUT IS VERTICAL
                    # Strut rake doesn't really show up in the airfoil drawing
                    write(io, @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
                        foilCoordsXform[jj, XDIM] + spanLoc[XDIM],
                        foilCoordsXform[jj, YDIM],
                        spanLoc[ZDIM],
                        u[ii] + dus[jj],
                        v[ii] + dvs[jj],
                        w[ii] + dws[jj],
                        phi[ii],
                        theta[ii],
                        psi[ii]))
                end
                # --- Connectivities ---
                for jj in 1:nAirfoilPts-1
                    write(io, @sprintf("%d\t%d\n", jj, jj + 1))
                end
            end
        end
    else
        error("Unsupported config: ", appendageOptions["config"])
    end
end
# ==============================================================================
#                         1D Stick Routines
# ==============================================================================
function write_deflections(
    DVDict, STATICSOL, FEMESH, outputDir::String, basename="static";
    appendageOptions=Dict("config" => "wing"), solverOptions=Dict(), iComp=1
)
    """
    This function writes the structural output of the 1D stick solver to a tecplot file
    """
    fTractions = STATICSOL.fHydro
    deflections = STATICSOL.structStates
    mesh = copy(FEMESH.mesh)
    mesh[:, XDIM] .+= appendageOptions["xMount"]
    elemConn = FEMESH.elemConn
    nNode = length(mesh[:, 1])
    nElem = length(elemConn[:, 1])

    @printf("Writing deformed structure to %s_<>.dat\n", basename)

    outfile = @sprintf("%s%s_comp%03d.dat", outputDir, basename, iComp)
    io = open(outfile, "w")

    # ************************************************
    #     Header
    # ************************************************
    write(io, @sprintf("TITLE = \"STATIC DEFLECTION Uinf = %.8f m/s\"\n", solverOptions["Uinf"]))
    write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \"u\" \"v\" \"w\" \"phi\" \"theta\" \"psi\"\n")
    write(io, "ZONE T = \"1D BEAM\" \n")
    write(io, @sprintf("NODES = %d, ", nNode))
    write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nNode - 1))
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    # ---------------------------
    #   Values
    # ---------------------------
    u = deflections[UIND:NDOF:end]
    v = deflections[VIND:NDOF:end]
    w = deflections[WIND:NDOF:end]
    phi = deflections[ΦIND:NDOF:end]
    theta = deflections[ΘIND:NDOF:end]
    psi = deflections[ΨIND:NDOF:end]
    # --- Write them ---
    for ii in 1:nNode
        nodeLoc = mesh[ii, :]

        stringData = @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM], u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii])
        write(io, stringData)
    end
    # ---------------------------
    #   Connectivities
    # ---------------------------
    for ii in 1:nElem
        write(io, @sprintf("%d\t%d\n", elemConn[ii, 1], elemConn[ii, 2]))
    end

    # ************************************************
    #     Airfoils
    # ************************************************
    write_airfoils(io, DVDict, mesh, u, v, w, phi, theta, psi; appendageOptions=appendageOptions)

    close(io)
end

function write_hydroLoads(
    LLOutputs, FlowCond, LLMesh, outputDir::String, basename="hydroloads";
)
    """
    Writes the hydrodynamic loads to a tecplot file
    """
    ϱ = FlowCond.rhof
    U∞ = FlowCond.Uinf
    dimTerm = 0.5 * ϱ * U∞^2 * LLMesh.SRef
    fTractions = LLOutputs.Fdist #* dimTerm
    mesh = transpose(copy(LLMesh.collocationPts))
    nNode = length(mesh[:, 1])

    @printf("Writing loads to %s_<>.dat\n", basename)

    outfile = @sprintf("%s%s.dat", outputDir, basename)
    io = open(outfile, "w")

    # ************************************************
    #     Header
    # ************************************************
    write(io, @sprintf("TITLE = \"STATIC LOADS Uinf = %.8f m/s\"\n", U∞))
    write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \"fx\" \"fy\" \"fz\" \n")
    write(io, "ZONE T = \"1D BEAM\" \n")
    write(io, @sprintf("NODES = %d, ", nNode))
    write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nNode - 1))
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    # ---------------------------
    #   Values
    # ---------------------------
    fx = fTractions[UIND:3:end]
    fy = fTractions[VIND:3:end]
    fz = fTractions[WIND:3:end]
    # --- Write them ---
    for ii in 1:nNode
        nodeLoc = mesh[ii, :]

        stringData = @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n",
            nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM], fx[ii], fy[ii], fz[ii])
        write(io, stringData)
    end
    # ---------------------------
    #   Connectivities
    # ---------------------------
    for ii in 1:nNode-1
        write(io, @sprintf("%d\t%d\n", ii, ii + 1))
    end



    close(io)

end

function write_LLmesh(io, LLMesh, uvec)
    """
    Write lifting line outline
    Uinfvec is for TV joints
    """

    mesh = LLMesh.nodePts
    jointPts = LLMesh.jointPts
    nNodes = LLMesh.npt_wing + 1
    localChords = LLMesh.localChords
    # ************************************************
    #     Leading edge
    # ************************************************
    write(io, "ZONE T = \"Leading Edges\" \n")
    # write(io, @sprintf("NODES = %d,", nNodes))
    # write(io, @sprintf("ELEMENTS = %d ZONETYPE=FELINESEG\n", nNodes - 1))
    write(io, "DATAPACKING = POINT\n")

    for (ii, nodeLoc) in enumerate(eachcol(mesh))
        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM] - 0.25 * localChords[ii], nodeLoc[YDIM], nodeLoc[ZDIM])
        write(io, stringData)
    end


    # ************************************************
    #     Trailing edge
    # ************************************************
    write(io, "ZONE T = \"Trailing edges\" \n")
    # write(io, @sprintf("NODES = %d,", nNodes))
    # write(io, @sprintf("ELEMENTS = %d ZONETYPE=FELINESEG\n", nNodes - 1))
    write(io, "DATAPACKING = POINT\n")

    for (ii, nodeLoc) in enumerate(eachcol(mesh))
        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM] + 0.75 * localChords[ii], nodeLoc[YDIM], nodeLoc[ZDIM])
        write(io, stringData)
    end

    # ************************************************
    #     Panel edges
    # ************************************************
    for (ii, nodeLoc) in enumerate(eachcol(mesh))
        write(io, "ZONE T = \"Panel edge $(ii)\" \n")
        write(io, "DATAPACKING = POINT\n")
        stringDataLE = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM] - 0.25 * localChords[ii], nodeLoc[YDIM], nodeLoc[ZDIM])
        stringDataTE = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM] + 0.75 * localChords[ii], nodeLoc[YDIM], nodeLoc[ZDIM])
        write(io, stringDataLE)
        write(io, stringDataTE)
    end

    # ************************************************
    #     Trailing vortices
    # ************************************************
    distance = 1.5 * LLMesh.rootChord * uvec
    # println("Distance: ", distance)
    for (ii, nodeLoc) in enumerate(eachcol(mesh))
        jointLoc = jointPts[:, ii]
        write(io, "ZONE T = \"Trailing vortex $(ii)\" \n")
        write(io, "DATAPACKING = POINT\n")
        stringDataA = @sprintf("%.16f\t%.16f\t%.16f\n",
            nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM])
        stringDataB = @sprintf("%.16f\t%.16f\t%.16f\n",
            jointLoc[XDIM], jointLoc[YDIM], jointLoc[ZDIM])
        stringDataC = @sprintf("%.16f\t%.16f\t%.16f\n",
            jointLoc[XDIM] + distance[XDIM], jointLoc[YDIM] + distance[YDIM], jointLoc[ZDIM] + distance[ZDIM])
        write(io, stringDataA)
        write(io, stringDataB)
        write(io, stringDataC)
    end
end

function write_1Dfemmesh(io, FEMESH)
    """
    Write the jig shape FEM stick mesh to tecplot
    """

    mesh = FEMESH.mesh
    nNodes = length(mesh[:, 1])
    nElem = length(FEMESH.elemConn[:, 1])
    # ************************************************
    #     Header
    # ************************************************
    write(io, "ZONE T = \"Mesh\"\n")
    write(io, @sprintf("NODES = %d,", nNodes))
    write(io, @sprintf("ELEMENTS = %d ZONETYPE=FELINESEG\n", nElem))
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(mesh) == 1 # 1D beam in a straight line
        for nodeLoc in mesh
            # Right now only 1D beam
            stringData = @sprintf("0.0\t%.16f\t0.0\n", nodeLoc)
            write(io, stringData)
        end
    elseif ndims(mesh) == 2 # 1D beam in 3D space
        # Loop by nodes
        for ii in 1:nNodes
            nodeLoc = mesh[ii, :]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM])
            write(io, stringData)
        end

        # Loop by elements to write connectivities
        for ii in 1:nElem
            write(io, @sprintf("%d\t%d\n", FEMESH.elemConn[ii, 1], FEMESH.elemConn[ii, 2]))
        end
    end

end

function write_hydroelastic_mode(DVDict, FLUTTERSOL, mesh, outputDir::String, basename="mode"; solverOptions=Dict("config" => "wing"))
    """
    Write the mode shape to tecplot
    Currently writes out a NACA 4-digit airfoil
    """

    true_eigs_r = FLUTTERSOL.eigs_r
    true_eigs_i = FLUTTERSOL.eigs_i
    R_eigs_r = FLUTTERSOL.R_eigs_r
    R_eigs_i = FLUTTERSOL.R_eigs_i
    iblank = FLUTTERSOL.iblank
    flowHistory = FLUTTERSOL.flowHistory
    nModes = FLUTTERSOL.NTotalModesFound

    @printf("Writing hydroelastic mode shape files for %i modes to %s_<>.dat\n", nModes, basename)

    nDOF = size(R_eigs_r)[1] ÷ 2 # no BC here right now and twice the size
    nDOFPerNode = 9
    nNode = nDOF ÷ nDOFPerNode + 1
    u = zeros(nNode)
    v = zeros(nNode)
    w = zeros(nNode)
    phi = zeros(nNode)
    theta = zeros(nNode)
    psi = zeros(nNode)
    for qq in 1:FLUTTERSOL.nFlow
        for mm in 1:nModes
            if iblank[mm, qq] == 0
                continue
            else
                outfile = @sprintf("%s%s_m%03i_q%03i.dat", outputDir, basename, mm, qq)
                io = open(outfile, "w")

                # ************************************************
                #     Header
                # ************************************************
                write(io, @sprintf("TITLE = \"MODE SHAPE %.8f HZ\"\n", true_eigs_i[mm, qq] / (2 * pi)))
                write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \"u\" \"v\" \"w\" \"phi\" \"theta\" \"psi\"\n")
                write(io, "ZONE T = \"1D BEAM\" \n")
                write(io, @sprintf("NODES = %d, ", nNode))
                write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nNode - 1))
                write(io, "DATAPACKING = POINT\n")

                # ************************************************
                #     Write contents
                # ************************************************
                # ---------------------------
                #   Values
                # ---------------------------
                # --- Store displacements ---
                if ndims(mesh) == 1 # 1D beam in a straight line
                    inode = 1
                    u, v, w = R_eigs_r[:, mm, qq], R_eigs_i[:, mm, qq]
                    for nodeLoc in mesh
                        # Right now only 1D beam
                        stringData = @sprintf("0.0\t%.16f\t0.0\t%.16f\t%.16f\t%.16f\n", nodeLoc)
                        write(io, stringData)
                        inode += 1
                    end
                elseif ndims(mesh) == 2 # 1D beam in 3D space
                    dim = 1 # iterate over the first dimension
                    # with zero in the front
                    u_r = [0; R_eigs_r[1:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    u_i = [0; R_eigs_i[1:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    v_r = [0; R_eigs_r[2:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    v_i = [0; R_eigs_i[2:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    w_r = [0; R_eigs_r[3:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    w_i = [0; R_eigs_i[3:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    phi_r = [0; R_eigs_r[4:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    phi_i = [0; R_eigs_i[4:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    theta_r = [0; R_eigs_r[5:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    theta_i = [0; R_eigs_i[5:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    psi_r = [0; R_eigs_r[6:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    psi_i = [0; R_eigs_i[6:nDOFPerNode:nNode*nDOFPerNode, mm, qq]]
                    for ii in 1:size(mesh)[dim]
                        u[ii] = sqrt(u_r[ii]^2 + u_i[ii]^2)
                        v[ii] = sqrt(v_r[ii]^2 + v_i[ii]^2)
                        w[ii] = sqrt(w_r[ii]^2 + w_i[ii]^2)
                        phi[ii] = sqrt(phi_r[ii]^2 + phi_i[ii]^2)
                        theta[ii] = sqrt(theta_r[ii]^2 + theta_i[ii]^2)
                        psi[ii] = sqrt(psi_r[ii]^2 + psi_i[ii]^2)
                    end
                end
                # --- Write them ---
                for ii in 1:size(mesh)[dim]
                    nodeLoc = mesh[ii, :]
                    stringData = @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM], u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii])
                    write(io, stringData)
                end
                # ---------------------------
                #   Connectivities
                # ---------------------------
                for ii in 1:nNode-1
                    write(io, @sprintf("%d\t%d\n", ii, ii + 1))
                end

                # ************************************************
                #     Airfoils
                # ************************************************
                write_airfoils(io, DVDict, mesh, u, v, w, phi, theta, psi; solverOptions=solverOptions)

                close(io)
            end
        end
    end
end

function write_natural_mode(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, mesh, outputDir::String; solverOptions=Dict("config" => "wing"))
    """
    Write the mode shape to tecplot
    Currently writes out a NACA 4-digit airfoil
    """

    nModes = length(structNatFreqs)

    nDOF = size(structModeShapes)[1] # has BC
    nDOFPerNode = 9
    nNode = nDOF ÷ nDOFPerNode
    u = zeros(nNode)
    v = zeros(nNode)
    w = zeros(nNode)
    phi = zeros(nNode)
    theta = zeros(nNode)
    psi = zeros(nNode)
    basename = "drymode"
    @printf("Writing natural mode shape files for %i modes to %s_<>.dat\n", nModes, basename)
    for mm in 1:nModes
        outfile = @sprintf("%s%s_m%03i.dat", outputDir, basename, mm)
        io = open(outfile, "w")

        modeShape = structModeShapes[:, mm]

        # ************************************************
        #     Header
        # ************************************************
        write(io, @sprintf("TITLE = \"MODE SHAPE %.8f HZ\"\n", structNatFreqs[mm]))
        write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \"u\" \"v\" \"w\" \"phi\" \"theta\" \"psi\"\n")
        write(io, "ZONE T = \"1D BEAM\" \n")
        write(io, @sprintf("NODES = %d, ", nNode))
        write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nNode - 1))
        write(io, "DATAPACKING = POINT\n")

        # ************************************************
        #     Write contents
        # ************************************************
        # ---------------------------
        #   Values
        # ---------------------------
        # --- Store displacements ---
        if ndims(mesh) == 2 # 1D beam in 3D space
            dim = 1 # iterate over the first dimension
            # with zero in the front
            u = modeShape[1:nDOFPerNode:nNode*nDOFPerNode]
            v = modeShape[2:nDOFPerNode:nNode*nDOFPerNode]
            w = modeShape[3:nDOFPerNode:nNode*nDOFPerNode]
            phi = modeShape[4:nDOFPerNode:nNode*nDOFPerNode]
            theta = modeShape[5:nDOFPerNode:nNode*nDOFPerNode]
            psi = modeShape[6:nDOFPerNode:nNode*nDOFPerNode]
        end
        # --- Write them ---
        for ii in 1:size(mesh)[dim]
            nodeLoc = mesh[ii, :]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM], u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii])
            write(io, stringData)
        end
        # ---------------------------
        #   Connectivities
        # ---------------------------
        for ii in 1:nNode-1
            write(io, @sprintf("%d\t%d\n", ii, ii + 1))
        end

        # ************************************************
        #     Airfoils
        # ************************************************
        write_airfoils(io, DVDict, mesh, u, v, w, phi, theta, psi; solverOptions=solverOptions)

        close(io)
    end

    basename = "wetmode"
    @printf("Writing natural mode shape files for %i modes to %s_<>.dat\n", nModes, basename)
    for mm in 1:nModes
        outfile = @sprintf("%s%s_m%03i.dat", outputDir, basename, mm)
        io = open(outfile, "w")

        modeShape = wetModeShapes[:, mm]

        # ************************************************
        #     Header
        # ************************************************
        write(io, @sprintf("TITLE = \"MODE SHAPE %.8f HZ\"\n", wetNatFreqs[mm]))
        write(io, "VARIABLES = \"X\" \"Y\" \"Z\" \"u\" \"v\" \"w\" \"phi\" \"theta\" \"psi\"\n")
        write(io, "ZONE T = \"1D BEAM\" \n")
        write(io, @sprintf("NODES = %d, ", nNode))
        write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nNode - 1))
        write(io, "DATAPACKING = POINT\n")

        # ************************************************
        #     Write contents
        # ************************************************
        # ---------------------------
        #   Values
        # ---------------------------
        # --- Store displacements ---
        if ndims(mesh) == 2 # 1D beam in 3D space
            dim = 1 # iterate over the first dimension
            # with zero in the front
            u = modeShape[1:nDOFPerNode:nNode*nDOFPerNode]
            v = modeShape[2:nDOFPerNode:nNode*nDOFPerNode]
            w = modeShape[3:nDOFPerNode:nNode*nDOFPerNode]
            phi = modeShape[4:nDOFPerNode:nNode*nDOFPerNode]
            theta = modeShape[5:nDOFPerNode:nNode*nDOFPerNode]
            psi = modeShape[6:nDOFPerNode:nNode*nDOFPerNode]
        end
        # --- Write them ---
        for ii in 1:size(mesh)[dim]
            nodeLoc = mesh[ii, :]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM], u[ii], v[ii], w[ii], phi[ii], theta[ii], psi[ii])
            write(io, stringData)
        end
        # ---------------------------
        #   Connectivities
        # ---------------------------
        for ii in 1:nNode-1
            write(io, @sprintf("%d\t%d\n", ii, ii + 1))
        end

        # ************************************************
        #     Airfoils
        # ************************************************
        write_airfoils(io, DVDict, mesh, u, v, w, phi, theta, psi; solverOptions=solverOptions)

        close(io)
    end
end

# ==============================================================================
#                         Hydro Routines
# ==============================================================================
function write_strips(io, DVDict, FEMESH; config="wing", nNodeWing=10)
    """
    Write lifting line
    """
    # ************************************************
    #     Header
    # ************************************************
    nnode = size(FEMESH.mesh)[1]
    write(io, "ZONE T=\"Hydrodynamic Strips\" \n")
    write(io, @sprintf("NODES = %d,", nnode * 2))
    write(io, @sprintf("ELEMENTS = %d ZONETYPE=FELINESEG\n", nnode))
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(FEMESH.mesh) == 1 # 1D beam in a straight line
        ii = 1 # spanwise counter
        for nodeLoc in FEMESH.mesh
            localChord = DVDict["c"][ii]
            b = localChord * 0.5

            # ---------------------------
            #   Write aero strip
            # ---------------------------
            XYZCoords1 = [-b, nodeLoc, 0.0]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords1[1], XYZCoords1[2], XYZCoords1[3])
            write(io, stringData)
            XYZCoords2 = [b, nodeLoc, 0.0]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords2[1], XYZCoords2[2], XYZCoords2[3])
            write(io, stringData)

            ii += 1
        end
    elseif ndims(FEMESH.mesh) == 2 # 1D beam in 3D space
        dim = 1 # iterate over the first dimension

        for ii in 1:nnode
            if ii <= nNodeWing
                localChord = DVDict["c"][ii]
                b = localChord * 0.5
                nodeLoc = FEMESH.mesh[ii, :]

                # ---------------------------
                #   Write aero strip
                # ---------------------------
                XYZCoords1 = nodeLoc - [b, 0.0, 0.0]
                stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords1[XDIM], XYZCoords1[YDIM], XYZCoords1[ZDIM])
                write(io, stringData)
                XYZCoords2 = nodeLoc + [b, 0.0, 0.0]
                stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords2[XDIM], XYZCoords2[YDIM], XYZCoords2[ZDIM])
                write(io, stringData)
            else
                if config == "t-foil"
                    if ii <= (nNodeWing * 2 - 1)
                        iwing = ii - nNodeWing
                        localChord = DVDict["c"][iwing]
                        b = localChord * 0.5
                        nodeLoc = FEMESH.mesh[iwing, :]
                        nodeLoc[YDIM] *= -1.0

                        # ---------------------------
                        #   Write aero strip
                        # ---------------------------
                        XYZCoords1 = nodeLoc - [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords1[XDIM], XYZCoords1[YDIM], XYZCoords1[ZDIM])
                        write(io, stringData)
                        XYZCoords2 = nodeLoc + [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords2[XDIM], XYZCoords2[YDIM], XYZCoords2[ZDIM])
                        write(io, stringData)
                    else
                        istrut = ii - nNodeWing * 2 + 1
                        localChord = DVDict["c_strut"][istrut]
                        b = localChord * 0.5
                        nodeLoc = FEMESH.mesh[ii, :]

                        # ---------------------------
                        #   Write aero strip
                        # ---------------------------
                        XYZCoords1 = nodeLoc - [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords1[XDIM], XYZCoords1[YDIM], XYZCoords1[ZDIM])
                        write(io, stringData)
                        XYZCoords2 = nodeLoc + [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords2[XDIM], XYZCoords2[YDIM], XYZCoords2[ZDIM])
                        write(io, stringData)
                    end
                end
            end


        end
    end
    # ************************************************
    #     Connectivities of aero strips
    # ************************************************
    jj = 1 # node counter
    for ii in 1:nnode # here, ii is line number
        write(io, @sprintf("%d\t%d\n", jj, jj + 1))
        jj += 2
    end

end

function write_oml(io, DVDict, FEMESH; config="wing", nNodeWing=10)
    """
    There is abug in this code that writes the OML a little funky
    """
    # ************************************************
    #     Header
    # ************************************************
    write(io, "ZONE T=\"OML\" \n")
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(FEMESH.mesh) == 1
        ii = 1 # spanwise counter
        for nodeLoc in mesh
            localChord = DVDict["c"][ii]
            b = localChord * 0.5

            XYZCoords = [-b, nodeLoc, 0.0]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
            write(io, stringData)

            ii += 1
        end
        ii = 0
        for nodeLoc in reverse(mesh)

            localChord = DVDict["c"][end-ii]
            b = localChord * 0.5
            XYZCoords = [b, nodeLoc, 0.0]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
            write(io, stringData)

            ii += 1
        end
    elseif ndims(FEMESH.mesh) == 2
        dim = 1 # iterate over the first dimension
        # Iterate over LE and TE
        for factor in [-1, 1]
            for ii in 1:size(FEMESH.mesh)[dim]

                if ii <= nNodeWing
                    localChord = DVDict["c"][ii]
                    b = localChord * 0.5

                    nodeLoc = FEMESH.mesh[ii, :]
                    if factor == -1
                        nodeLoc = FEMESH.mesh[end-ii+1, :]
                    end

                    XYZCoords = nodeLoc + factor * [b, 0.0, 0.0]
                    stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
                    write(io, stringData)
                else
                    if ii <= (nNodeWing * 2 - 1)
                        iwing = ii - nNodeWing
                        localChord = DVDict["c"][iwing]
                        b = localChord * 0.5

                        nodeLoc = FEMESH.mesh[iwing, :]
                        nodeLoc[YDIM] *= -1.0
                        if factor == -1
                            nodeLoc = FEMESH.mesh[end-iwing+1, :]
                        end

                        XYZCoords = nodeLoc + factor * [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
                        write(io, stringData)
                    else
                        istrut = ii - 2 * nNodeWing + 1
                        localChord = DVDict["c_strut"][istrut]
                        b = localChord * 0.5

                        nodeLoc = FEMESH.mesh[ii, :]
                        if factor == -1
                            nodeLoc = FEMESH.mesh[end-ii+1, :]
                        end

                        XYZCoords = nodeLoc + factor * [b, 0.0, 0.0]
                        stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
                        write(io, stringData)
                    end
                end
            end
        end
    end

end

end # module