# --- Julia 1.11---
"""
@File          :   ComputeForcedFunctions.jl
@Date created  :   2024/11/22
@Last modified :   2025/02/10
@Author        :   Galen Ng
@Desc          :   Compute cost functions from the forced vibration response
"""


function compute_PSDArea(PSD, fSweep, meanChord)
    """
    Compute the area under the PSD curve nondimensionalized
    Inputs
    ------
    PSD : [m^2 - sec] Power Spectral Density of the deformations

    """

    # df = fSweep[2] - fSweep[1]
    dfs = fSweep[2:end] - fSweep[1:end-1] # [Hz] frequency increments

    ω_char = √(GRAV / (0.5 * meanChord)) # [Hz] characteristic frequency for nondimensionalization


    PSDArea = sum(PSD) * df / ω_char

    return
end

function compute_responsePeak(dynDeflections, limit, solverOptions)

    ksmax = compute_KS(dynDeflections, solverOptions["rhoKS"])
    return ksmax - limit
end
