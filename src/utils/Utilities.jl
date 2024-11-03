"""
@File    :   Utilities.jl
@Time    :   2024/04/12
@Author  :   Galen Ng
@Desc    :   Some routines that are useful for the solver and do not depend on Anything
"""


module Utilities

using ..DesignConstants: SORTEDDVS
using ..DCFoil: RealOrComplex

function unpack_dvdict(DVDict::Dict)

    nDVs::Int64 = 0

    for dvKey in keys(DVDict)
        nDVs += length(DVDict[dvKey])
    end
    DVVec = zeros(nDVs)
    DVLengths = zeros(Int64, length(SORTEDDVS))

    # --- Unpack DVs using sorted keys ---
    iDV::Int64 = 1
    for (ii, dvKey) in enumerate(SORTEDDVS)

        dvValues = DVDict[dvKey]
        lDV = length(dvValues)

        if lDV == 1
            DVVec[iDV] = dvValues
        elseif lDV > 1
            DVVec[iDV:iDV+lDV-1] = dvValues
        end

        DVLengths[ii] = lDV
        iDV += lDV
    end
    return DVVec, DVLengths
end

function repack_dvdict(DVVec, DVLengths::Vector{Int64})
    """
    Repack DVVec into a dictionary
    """

    DVDict = Dict()
    iDV::Int64 = 1
    for (ii, dvKey) in enumerate(SORTEDDVS)

        lDV = DVLengths[ii]

        if lDV == 1
            DVDict[dvKey] = DVVec[iDV]
        elseif lDV > 1
            DVDict[dvKey] = DVVec[iDV:iDV+lDV-1]
        end

        iDV += lDV
    end
    return DVDict

end

function pack_funcsSens(funcsSens::Dict, funcKey, dvKey, dfdx)
    """
    We want the function sensitivities dictionary 
    to be the same way ADflow stores its data.
    That is,
    funcsSens = Dict(
    costFunc1 => Dict(
        dv1 => array[sens1], # if DV is scalar
        dv2 => array[sens1, sens2, ...], # if DV is vector
        ...
    ),
    """

    funcsSens[funcKey][dvKey] = dfdx

    return funcsSens
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

end