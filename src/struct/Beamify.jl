# --- Julia 1.9---
"""
@File    :   Beamify.jl
@Time    :   2024/01/18
@Author  :   Galen Ng
@Desc    :   Module to take a NASTRAN BDF file and reduce its properties to a beam
"""

module Beamify

    include("../io/NastranIO.jl")
    using .NastranIO

    function get_oml_from_fem(femFile)

        NastranIO.read_bdf()
        
    end
end