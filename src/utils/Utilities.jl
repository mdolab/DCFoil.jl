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

end