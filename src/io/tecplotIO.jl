# --- Julia 1.7---
"""
@File    :   TecplotIO.jl
@Time    :   2023/03/05
@Author  :   Galen Ng
@Desc    :   Interface for writing an output Tecplot can read
When in doubt, refer to the Tecplot Data Format Guide
"""


module TecplotIO

using Printf

# --- Globals ---
global XDIM = 1
global YDIM = 2
global ZDIM = 3

function write_mesh(DVDict::Dict, mesh, outputDir::String, fname="mesh.dat")
    """
    Top level routine to write the mesh file
    """

    outfile = @sprintf("%s%s", outputDir, fname)
    @printf("Writing mesh file %s...\n", outfile)

    io = open(outfile, "w")
    write(io, "TITLE = \"Mesh Data\"\n")
    write(io, "VARIABLES = \"X\" \"Y\" \"Z\"\n")

    write_1Dfemmesh(io, mesh)
    write_strips(io, DVDict, mesh)
    write_oml(io, DVDict, mesh)

    close(io)
end


function generate_naca4dig(toc)
    """
    Simple naca 
    """
    # --- Thickness distribution naca 4dig equation---
    C5 = 0.1015  # type I equation
    x = range(0, 1, length=50)

    # Thickness distribution (upper)
    yt = 5 * toc * (0.2969 * x .^ 0.5 - 0.126 * x - 0.3516 * x .^ 2 + 0.2843 * x .^ 3 - C5 * x .^ 4)
    lower_yt = -yt
    # Make CCW
    upper_yt = reverse(yt)
    y_coords = vcat(upper_yt, lower_yt)
    x_coords = vcat(reverse(x), x)
    foil_coords = hcat(x_coords, y_coords)
    return foil_coords
end

function transform_airfoil(foilCoords, localChord)
    """
    Unit airfoil to DCFoil frame
    """
    # Translate airfoil to be centered at the midchord
    foilCoordsXform = copy(foilCoords)
    foilCoordsXform[:, XDIM] .+= -0.5
    # Scale airfoil
    foilCoordsXform = localChord * foilCoordsXform
    return foilCoordsXform
end

function write_airfoils(io, DVDict, mesh, dim, u, v, w, phi, theta, psi)
    """
    TODO generalize to take in a normal vector in spanwise direction
    """

    foilCoords = generate_naca4dig(DVDict["toc"])

    for ii in 1:size(mesh)[dim] # iterate over span
        spanLoc = mesh[ii, :]
        localChord = DVDict["c"][ii]
        foilCoordsXform = transform_airfoil(foilCoords, localChord)

        # Get u, v, w based on rotations
        nAirfoilPts = size(foilCoordsXform)[1]
        # uAirfoil = u[ii] * ones(size(foilCoordsScaled)[1])
        dws = foilCoordsXform[:, XDIM] * sin(theta[ii]) # airfoil twist
        dvs = foilCoordsXform[:, YDIM] * sin(phi[ii]) # airfoil OOP bend
        dus = foilCoordsXform[:, XDIM] * sin(psi[ii]) # airfoil IP bend

        # --- Header ---
        write(io, @sprintf("ZONE T = \"Airfoil midchord (%.8f, %.8f, %.8f)\" \n", spanLoc[XDIM], spanLoc[YDIM], spanLoc[ZDIM]))
        write(io, @sprintf("NODES = %d, ", nAirfoilPts))
        write(io, @sprintf("ELEMENTS = %d, ZONETYPE=FELINESEG\n", nAirfoilPts - 1))
        write(io, "DATAPACKING = POINT\n")
        # --- Values ---
        for jj in 1:nAirfoilPts
            write(io, @sprintf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", foilCoordsXform[jj, XDIM], spanLoc[YDIM], foilCoordsXform[jj, YDIM], u[ii] + dus[jj], v[ii] + dvs[jj], w[ii] + dws[jj], phi[ii], theta[ii], psi[ii]))
        end
        # --- Connectivities ---
        for jj in 1:nAirfoilPts-1
            write(io, @sprintf("%d\t%d\n", jj, jj + 1))
        end
    end
end
# ==============================================================================
#                         1D Stick Routines
# ==============================================================================
function write_1Dfemmesh(io, mesh)
    """
    Write the jig shape FEM stick mesh to tecplot
    """

    # ************************************************
    #     Header
    # ************************************************
    write(io, "ZONE T = \"Mesh\"\n")
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
        dim = 1 # iterate over the first dimension

        for ii in 1:size(mesh)[dim]
            nodeLoc = mesh[ii, :]
            stringData = @sprintf("%.16f\t%.16f\t%.16f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM])
            write(io, stringData)
        end
    end

end # end write_mesh

function write_hydroelastic_mode(DVDict, FLUTTERSOL, mesh, outputDir::String, basename="mode")
    """
    Write the mode shape to tecplot
    # TODO: should mode 3 be in-plane?? No it shouldn't. Why does it appear?
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
                write_airfoils(io, DVDict, mesh, dim, u, v, w, phi, theta, psi)

                close(io)
            end
        end
    end
end # end write_hydroelastic_mode

function write_natural_mode(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, mesh, outputDir::String)
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
        write_airfoils(io, DVDict, mesh, dim, u, v, w, phi, theta, psi)

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
        write_airfoils(io, DVDict, mesh, dim, u, v, w, phi, theta, psi)

        close(io)
    end
end # end write_natural_mode


# ==============================================================================
#                         Hydro Routines
# ==============================================================================
function write_strips(io, DVDict, mesh)
    # ************************************************
    #     Header
    # ************************************************
    nnode = size(mesh)[1]
    write(io, "ZONE T=\"Hydrodynamic Strips\" \n")
    write(io, @sprintf("NODES = %d,", nnode * 2))
    write(io, @sprintf("ELEMENTS = %d ZONETYPE=FELINESEG\n", nnode))
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(mesh) == 1 # 1D beam in a straight line
        ii = 1 # spanwise counter
        for nodeLoc in mesh
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
    elseif ndims(mesh) == 2 # 1D beam in 3D space
        dim = 1 # iterate over the first dimension

        for ii in 1:nnode
            localChord = DVDict["c"][ii]
            b = localChord * 0.5
            nodeLoc = mesh[ii, :]

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
    # ************************************************
    #     Connectivities of aero strips
    # ************************************************
    jj = 1 # node counter
    for ii in 1:nnode # here, ii is line number
        write(io, @sprintf("%d\t%d\n", jj, jj + 1))
        jj += 2
    end

end

function write_oml(io, DVDict, mesh)
    # ************************************************
    #     Header
    # ************************************************
    write(io, "ZONE T=\"OML\" \n")
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(mesh) == 1
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
    elseif ndims(mesh) == 2
        dim = 1 # iterate over the first dimension
        # Iterate over LE and TE
        for factor in [-1, 1]
            for ii in 1:size(mesh)[dim]
                localChord = DVDict["c"][ii]
                b = localChord * 0.5

                nodeLoc = mesh[ii, :]
                if factor == -1
                    nodeLoc = mesh[end-ii+1, :]
                end

                XYZCoords = nodeLoc + factor * [b, 0.0, 0.0]
                stringData = @sprintf("%.16f\t%.16f\t%.16f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
                write(io, stringData)
            end
        end
    end

end

end # module