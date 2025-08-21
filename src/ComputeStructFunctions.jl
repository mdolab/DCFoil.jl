# --- Julia 1.11---
"""
@File          :   ComputeStructFunctions.jl
@Date created  :   2025/02/18
@Last modified :   2025/02/18
@Author        :   Galen Ng
@Desc          :   Compute cost functions for structure
"""

function compute_maxtipbend(states)
    W = states[WIND:NDOF:end]

    return W[end]
end

function compute_maxtiptwist(states)
    Theta = states[ΘIND:NDOF:end]

    return Theta[end]
end

