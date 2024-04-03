# --- Julia 1.9---
"""
@File    :   CustomRules.jl
@Time    :   2024/02/24
@Author  :   Galen Ng
@Desc    :   Contains custom chain rules for FUNDAMENTAL operations
This is not a module on purpose so it's in the namespace when it's first included
"""

# --- PACKAGES ---
using ChainRulesCore
using LinearAlgebra

# # --- DCFoil modules ---
# using ..SolverRoutines: cmplxStdEigValProb2

# ==============================================================================
#                         Constructors
# ==============================================================================
function ChainRulesCore.rrule(::Type{LinRange}, start::RealOrComplex, stop::RealOrComplex, N::Integer)
    
    function LinRange_pullback(yb)
        """
        The input to the pullback is
        a vector of backward seeds of ones
        """
        # Do dot product so the output is the right structure
        startb = dot(yb, LinRange(1.0, 0.0, length(yb)))
        stopb = dot(yb, LinRange(0.0, 1.0, length(yb)))
        # println("This is yb", yb)

        return NoTangent(), startb, stopb, NoTangent()
    end

    
    return LinRange(start, stop, N), LinRange_pullback
end

# ==============================================================================
#                         Functions
# ==============================================================================
function ChainRulesCore.rrule(::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex})
    """
    MATRIX MULTIPLY RULE
    """
    function times_pullback(ΔΩ)
        ∂A = @thunk(ΔΩ * B')
        ∂B = @thunk(A' * ΔΩ)
        return (NoTangent(), ∂A, ∂B)
    end
    return A * B, times_pullback
end

function ChainRulesCore.rrule(::typeof(inv), A::Matrix{<:RealOrComplex})
    """
    MATRIX INVERSE
    """
    Ω = inv(A)

    function inv_pullback(ΔΩ)

        ∂A = -Ω' * ΔΩ * Ω'

        return (NoTangent(), ∂A)
    end

    return Ω, inv_pullback
end
