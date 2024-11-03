# --- Julia 1.11---
"""
@File          :   MeshIO.jl
@Date created  :   2024/10/15
@Last modified :   2024/10/15
@Author        :   Galen Ng
@Desc          :   Interface for reading mesh files
"""


module MeshIO

NSKIP = 1 # Number of lines to skip in the mesh file

struct Grid{TF,TI}
    """
    Struct to hold the top-level grid data for the program
    """
    LEMesh::AbstractMatrix{TF} # Mesh for the LE; 2D (3 x npts) array of (x,y,z) coordinates
    nodeConn::Matrix{TI} # Connectivity for the LE mesh; 
    TEMesh::AbstractMatrix{TF}
end

function add_meshfiles(gridFiles, options)

    # Define so they're not stuck in local scope
    nNodes = 0
    LEMesh = nothing
    TEMesh = nothing
    nodeConn = nothing
    for (ii, gridFile) in enumerate(gridFiles)

        meshGrid = add_mesh(gridFile)

        if ii == 1 # first mesh
            nNodes = size(meshGrid.LEMesh)[2]
            nodeConn = meshGrid.nodeConn
            LEMesh = meshGrid.LEMesh
            TEMesh = meshGrid.TEMesh

        else # append to the existing mesh

            if options["junction-first"] # the first node is the junction
                println("Junction first")

                modifiedConn = meshGrid.nodeConn .+ nNodes .- 1
                modifiedConn[1, 1] = 1

                nodeConn = hcat(nodeConn, modifiedConn)

                # Remove the first node from the LE mesh
                LEMesh = hcat(LEMesh, meshGrid.LEMesh[:, 2:end])
                TEMesh = hcat(TEMesh, meshGrid.TEMesh[:, 2:end])
            else
                error("Not implemented yet")
            end

            # New shape
            nNodes = size(LEMesh)[2]
        end


        println("-"^50)
        println("Added mesh file: $gridFile with $(size(meshGrid.LEMesh)[2]) nodes")
        # println("Total nodes: $nNodes")
        # println("LEMesh: $(size(LEMesh))")
        # for (ii, node) in enumerate(eachcol(LEMesh))
        #     println("Node: $(node)")
        # end
        # println("Node connectivity: $(size(nodeConn))")
        # for (ii, conn) in enumerate(eachcol(nodeConn))
        #     println("$(conn)")
        # end

    end
    GridStruct = Grid(LEMesh, nodeConn, TEMesh)

    return GridStruct
end

function add_mesh(gridFile)
    """
    Read a single mesh file and add it to the grid struct
    """

    if occursin(".dcf", gridFile)

        LEMesh, nodeConn, TEMesh = read_dcf(gridFile)

        GridStruct = Grid(LEMesh, nodeConn, TEMesh)
    else
        error("File type not supported for mesh")
    end

    return GridStruct
end

function read_dcf(gridFile)
    """
    Read the mesh file in the *.DCF format
    The component mesh should be in the format:
    <component-name>
    LE
    <x y z>
    TE
    <x y z>

    Make sure that these points go in the direction you want the composite fiber angles to be along
    """

    f = open(gridFile, "r")

    # --- Read the file ---
    # Loop through the file and read the data
    LEMesh = []
    nodeConn = []
    TEMesh = []

    isLE = false
    isTE = false
    LEctr = 1
    TEctr = 1
    for (ii, line) in enumerate(eachline(f))
        if ii > NSKIP

            # --- Boolean checking ---
            if occursin("TE", uppercase(line))
                isTE = true
                isLE = false
                continue
            elseif occursin("LE", uppercase(line))
                isLE = true
                isTE = false
                continue
            end


            # Add to data structs
            if isLE
                push!(LEMesh, [parse(Float64, x) for x in split(line, " ")])

                if LEctr > NSKIP
                    push!(nodeConn, [LEctr - 1, LEctr])
                end

                LEctr += 1

            elseif isTE

                push!(TEMesh, [parse(Float64, x) for x in split(line, " ")])

                TEctr += 1

            end

        end
    end
    # Turn data structs into matrices
    LEMesh = hcat(LEMesh...)
    nodeConn = hcat(nodeConn...)
    TEMesh = hcat(TEMesh...)

    return LEMesh, nodeConn, TEMesh

end

end