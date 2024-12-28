# --- Julia 1.9---
"""
@File    :   DesignVariables.jl
@Time    :   2024/11/18
@Author  :   Galen Ng
@Desc    :   Store all possible design variables that aren't the mesh
"""

module DesignVariables


const allDesignVariables::Vector{String} = [
    "alfa0",
    "theta_f",
    "toc",
]


end