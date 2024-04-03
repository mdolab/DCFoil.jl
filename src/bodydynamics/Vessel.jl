# --- Julia 1.9---
"""
@File    :   Vessel.jl
@Time    :   2024/01/26
@Author  :   Galen Ng
@Desc    :   Marine dynamics of the foiling craft module
"""

module Vessel

# --- PACKAGES ---
using LinearAlgebra

# --- DCFoil modules ---
using ..SolverRoutines
using ..SolutionConstants: XDIM, YDIM, ZDIM


function compute_cm(STATSOLLIST::Vector, rho_s, chordVec, xAreaVec, hullCM, hullMass, rVec)
    """
    Get center of mass including foils

    This function should be called after foils and hull are initialized. This function 
    returns the center of mass of the vessel from the initial origin of the 
    foil reference frame.

    Parameters
    ----------
    structMesh : array
        structural mesh of foils
    rho_s : float
        density of foil material
    chordVec : array
        chord length of each foil node
    xAreaVec : array
        cross-sectional area of each foil node (unit chord based)
    hullCM : array
        CM of vessel
    hullMass : float
        mass of vessel
    rVec : array
        vector of foil mounting locations wrt hull CM (vessel w/o foils)

    Returns
    -------
    xCM : array
        center of mass of vessel from initial origin of hull CM.
        This should be lower since the foils are mounted below the hull
    Mass : float
        total mass of vessel
    """

    # --- CM of foils ---
    CMfoil = zeros(3)
    Mfoil = 0.0
    for inode in 1:(length(structMesh[:, 1])-1)
        dA = chordVec[inode] * xAreaVec
        xElem = 0.5 * (structMesh[inode+1, :] + structMesh[inode, :]) # element centroid
        dl = norm(structMesh[inode+1, :] - structMesh[inode, :])
        dV = dA * dl
        dm = rho_s * dV
        CMfoil[:] += dm * xElem
        Mfoil += dm
    end

    # Vector addition to get position of CM of foil in the vessel frame
    # Add vector of mounting location wrt CM of hull to the position vector CM of foil
    CMfoil[:] = rVec + CMfoil

    # --- CM of hull + foils ---
    # It is just a weighted average
    Mass = hullMass + Mfoil
    xCM = (hullMass * hullCM + Mfoil * CMfoil) / Mass

    return xCM, Mass

end

function compute_gravloads(HULL, solverOptions::Dict)
    """
    compute f_gB  (m_gB = 0)
    """

    f_gB = HULL.mass * solverOptions["gravityVector"]

    return f_gB
end


end

