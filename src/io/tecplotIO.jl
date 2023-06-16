# --- Julia 1.7---
"""
@File    :   tecplotIO.jl
@Time    :   2023/03/05
@Author  :   Galen Ng
@Desc    :   Interface for writing an output Tecplot can read
"""


module tecplotIO

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
    write(io, "VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\"\n")

    write_1Dfemmesh(io, mesh)
    write_strips(io, DVDict, mesh)
    write_oml(io, DVDict, mesh)

    close(io)
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
    write(io, "ZONE T=\"Mesh\"\n")
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    if ndims(mesh) == 1 # 1D beam in a straight line
        for nodeLoc in mesh
            # Right now only 1D beam
            stringData = @sprintf("0.0\t%.8f\t0.0\n", nodeLoc)
            write(io, stringData)
        end
    elseif ndims(mesh) == 2 # 1D beam in 3D space
        dim = 1 # iterate over the first dimension

        for ii in 1:size(mesh)[dim]
            nodeLoc = mesh[ii, :]
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", nodeLoc[XDIM], nodeLoc[YDIM], nodeLoc[ZDIM])
            write(io, stringData)
        end
    end

end # end write_mesh

function write_mode_shape(mesh, outputDir::String, fname="mode.dat")
end

# ==============================================================================
#                         Hydro Routines
# ==============================================================================
function write_strips(io, DVDict, mesh)
    # ************************************************
    #     Header
    # ************************************************
    nnode = size(mesh)[1]
    write(io, "ZONE T=\"Hydrodynamic Strips\" \n")
    write(io, @sprintf("Nodes = %d,", nnode * 2))
    write(io, @sprintf("Elements = %d ZONETYPE=FELINESEG\n", nnode))
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
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords1[1], XYZCoords1[2], XYZCoords1[3])
            write(io, stringData)
            XYZCoords2 = [b, nodeLoc, 0.0]
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords2[1], XYZCoords2[2], XYZCoords2[3])
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
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords1[XDIM], XYZCoords1[YDIM], XYZCoords1[ZDIM])
            write(io, stringData)
            XYZCoords2 = nodeLoc + [b, 0.0, 0.0]
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords2[XDIM], XYZCoords2[YDIM], XYZCoords2[ZDIM])
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
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords[1], XYZCoords[2], XYZCoords[3])
            write(io, stringData)

            ii += 1
        end
        ii = 0
        for nodeLoc in reverse(mesh)

            localChord = DVDict["c"][end-ii]
            b = localChord * 0.5
            XYZCoords = [b, nodeLoc, 0.0]
            stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords[1], XYZCoords[2], XYZCoords[3])
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
                stringData = @sprintf("%.8f\t%.8f\t%.8f\n", XYZCoords[XDIM], XYZCoords[YDIM], XYZCoords[ZDIM])
                write(io, stringData)
            end
        end
    end

end

end # module