# --- Julia 1.11---
"""
@File          :   LDTransfer.jl
@Date created  :   2025/01/30
@Last modified :   2025/01/30
@Author        :   Galen Ng
@Desc          :   Load and displacement transfer routines
"""


module LDTransfer

using ..EBBeam: UIND, VIND, WIND, ΦIND, ΘIND, ΨIND, NDOF
using ..SolverRoutines


function transfer_LD(inputVector, XStruct, XHydro; mode="D")
    """
    Inputs
    ------
        mode : string
            'D' or 'L' to transfer displacements or loads
    """

    xferMat = compute_transferMatrix()

    if mode == "D"
        # Given displacements in the structural frame, 
        # get the displacements in the hydrodynamic model

        # u_LL = T u

        return xferMat * inputVector

    elseif mode == "L"
        # Given the hydrodynamic loads coming from the lifting line,
        # compute the traction loads to the structural model

        # f = Tᵀ f_LL
        return transpose(xferMat) * inputVector
    end
end

function compute_transferMatrix(XStruct, XHydro)
    """
    The transfer function is defined
        T = ∂u_LL / ∂ u = ∂X_LL / ∂u
    and
        Tᵀ = ∂f / ∂ f_LL
    """
    
    function transfer_value(sLoc, sEval, vals)
        """
        sLoc is the location to evaluate
        sEval is the vector of locations we have

        """
        valq = SolverRoutines.do_linear_interp(sEval, vals, sLoc)
        return valq
    end

    # Linear interpolations of locations
    xferMat = zeros(length(XLL), length(allStructStates))

    for ii in 1:length(allStructStates)

    end

    return xferMat
end


end