# --- Julia 1.7---
"""
@File    :   tecplotIO.jl
@Time    :   2023/03/05
@Author  :   Galen Ng
@Desc    :   Interface for writing an output Tecplot can read
"""


module tecplotIO

using Printf

function write_mesh(mesh, outputDir::String, fname="mesh.dat")
    """
    Write the jig shape FEM mesh to an ASCII file
    """

    outfile = @sprintf("%s%s", outputDir, fname)
    @printf("Writing mesh file %s...\n", outfile)
    # ************************************************
    #     Header
    # ************************************************
    io = open(outfile, "w")
    write(io, "TITLE = \"Mesh\"\n")
    write(io, "VARIABLES = \"CoordinateX\" \"CoordinateY\" \"CoordinateZ\"\n")
    write(io, "ZONE T=\"Mesh\"\n")
    write(io, "DATAPACKING = POINT\n")

    # ************************************************
    #     Write contents
    # ************************************************
    for nodeLoc in mesh
        # Right now only 1D beam
        stringData = @sprintf("0.0\t%.8f\t0.0\n", nodeLoc)
        write(io, stringData)
    end
    close(io)

end

end # module