# --- Julia ---

# @File    :   GovDiffEqns.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Two modules containing the governing differential equations recasting as a linear system
#              q' = f(q(y)) 
#              where q = [w, ψ, w', ψ', w'', ψ'', w''', ψ''']ᵀ
#              The functions in this module are the 'f' in the above equation
# 
# TODO: It might be better to redo the solution algorithm to be the finite element method
# i.e. one that solve A\b = {u}


module Steady
"""
Steady differential equations module
All time derivative terms ∂/∂t() = 0 and C(k=0) = 1
"""

# --- Libraries ---
using FLOWMath: linear
using LinearAlgebra
using DifferentialEquations

export compute_∂q∂y

function compute_∂q∂y(qn, η, foil)
    """
    Compute the derivative of the column state vector q 
    with respect to the spatial variable y at parametric location η
    """
    # TODO: DEBUG ALL OF THIS
    # --- First interpolate all necessary values based on spanwise location ---
    y = LinRange(-foil.s, 0, foil.neval)
    yⁿ = -η * foil.s
    clα = linear(y, foil.clα, yⁿ)
    c = linear(y, foil.c, yⁿ)
    b = 0.5 * c # semichord for more readable code
    ab = linear(y, foil.ab, yⁿ)
    eb = linear(y, foil.eb, yⁿ)
    EIₛ = linear(y, foil.EIₛ, yⁿ)
    GJₛ = linear(y, foil.GJₛ, yⁿ)
    Kₛ = linear(y, foil.Kₛ, yⁿ)
    Sₛ = linear(y, foil.Sₛ, yⁿ)
    q = copy(qn)
    q[2] += foil.α₀ * π / 180 # update the angle of attack to be total
    L = foil.s

    # --- Compute governing matrix equations ---
    # NOTE: the convention is [w, ψ]ᵀ for the indexing
    qf = 0.5 * foil.ρ_f * foil.U∞^2 # dynamic pressure
    # Fluid de-stiffening (disturbing)
    K_f = qf * cos(foil.Λ)^2 *
          [
              0.0 -2*b*clα
              0.0 -2*eb*b*clα
          ]

    # Sweep correction matrix
    E_f = qf * sin(foil.Λ) * cos(foil.Λ) * b *
          [
              2*clα -clα*b*(1-ab/b)
              clα*b*(1+ab/b) π*b^2-0.5*clα*b^2*(1-(ab/b)^2)
          ]

    # --- Build the linear system ---
    # 4th deriv terms: w'''', ψ''''
    A = (1 / L^4) *
        [
        EIₛ 0.5*ab*EIₛ
        0.5*ab*EIₛ Sₛ
    ]
    # 3rd deriv terms: w''', ψ'''
    B = (1 / L^3) *
        [
        0 Kₛ
        -Kₛ 0
    ]
    # 2nd deriv terms: w'', ψ''
    C = (1 / L^2) *
        [
        0 0
        0 -GJₛ
    ]
    # 0th deriv terms: w, ψ
    D = K_f
    # 1st deriv terms: w', ψ'
    E = 1 / L * E_f

    bVec = -(B * q[7:8] + C * q[5:6] + D * q[1:2] + E * q[3:4])

    x = A \ bVec

    # --- Solution ---
    # reset the angle of attack because julia is tracking this modification to the input throughout the run
    q[2] -= foil.α₀ * π / 180
    ∂q∂y = zeros(Float64, 8)
    ∂q∂y[1:6] .= qn[3:end]
    ∂q∂y[7:8] .= x

    return ∂q∂y

end

function compute_g(ya, yb, foil)
    """
    Boundary condition function at a and b
    """
    neqns = length(ya)
    g = zeros(neqns)
    idxTip = lastindex(foil.Kₛ)

    EIₛ = foil.EIₛ[idxTip]
    GJₛ = foil.GJₛ[idxTip]
    Kₛ = foil.Kₛ[idxTip]
    cTip = foil.c[idxTip]
    abTip = foil.ab[idxTip]
    L = foil.s

    g[1:4] .= ya[1:4]
    g[5] = (yb[5] + Kₛ * L / (EIₛ) * yb[4])
    g[6] = (yb[6])
    g[7] = (yb[7] + 0.5 * abTip * (GJₛ - Kₛ^2 / EIₛ) * L^2 / (EIₛ * cTip^2 / 12.0) * yb[4])
    g[8] = (yb[8] - (GJₛ - Kₛ^2 / EIₛ) * L^2 / (EIₛ * cTip^2 / 12.0) * yb[4])
    return g
end

end # end module

# module DynamicDiffEqns

# end # end module

# using .DynamicDiffEqns

# ==============================================================================
#                         ODE Solver
# ==============================================================================
module Solver
# Contains routines for solving an ODE and a BVP
using LinearAlgebra

function solve_rk4(dudt, u0, t0, tf, nnode, foil=nothing)
    """
    4-stage Runge-Kutta integrator
    TODO: there should be a better way to do this without passing 'foil' through

    Parameters
    ----------
        dudt - callable function
    """
    # TODO: type declaration for performance
    # m::Int64 # no of eqns
    # ncounter::Int64 # node counter
    # dt::Float64 
    # tsol::Array{Float64} 
    m = length(u0)
    ncounter = 1
    dt = (tf - t0) / (nnode - 1) # time step
    tsol = t0:dt:tf
    usol = zeros(m, nnode)

    # ---------------------------
    #   Initialize
    # ---------------------------
    t = t0
    u = copy(u0)
    usol[:, begin] = copy(u0)

    # ---------------------------
    #   Integration from node #2 onwards
    # ---------------------------
    for nn in 1:nnode-1
        tsol[nn]
        # --- multi-stage evaluation pts ---
        t1 = t + 0.5 * dt
        t2 = t + 0.5 * dt
        t3 = t + dt

        # --- evaluate at current node ---
        if foil == nothing
            f0 = dudt(u, t)
        else
            f0 = dudt(u, t, foil)
        end

        # --- state @ 1st stage ---
        u1 = u + 0.5 * dt * f0
        if foil == nothing
            f1 = dudt(u1, t1)
        else
            f1 = dudt(u1, t1, foil)
        end

        # --- state @ 2nd stage ---
        u2 = u + 0.5 * dt * f1
        if foil == nothing
            f2 = dudt(u2, t2)
        else
            f2 = dudt(u2, t2, foil)
        end

        # --- state @ 3rd stage ---
        u3 = u + dt * f2
        if foil == nothing
            f3 = dudt(u3, t3)
        else
            f3 = dudt(u3, t3, foil)
        end

        # --- take step ---
        u += dt / 6 * (f0 + 2 * f1 + 2 * f2 + f3) # RK4 formula
        t += dt
        ncounter += 1
        usol[:, ncounter] = copy(u)
    end


    return tsol, usol
end

function solve_bvp(dudt, u0, t0, tf, nnode, compute_g, foil=nothing)
    """
    Shooting method for solving a 2pt BVP using a Newton Solver + an RK4 solver

    """
    # ************************************************
    #     Initializations
    # ************************************************
    # TODO: declare types ahead?
    m = length(u0)
    n = m ÷ 2 # no of free parameters
    jac = zeros(m, m)
    u = u0
    du = 1e-8
    tol = 1e-14
    maxIter = 20
    iter = 1

    tsol, usol = solve_rk4(dudt, u, t0, tf, nnode, foil)
    res = compute_g(usol[:, begin], usol[:, end], foil)

    # --- Newton iterations ---
    while norm(res) > tol
        upert = copy(u)

        # --- Fill in Jacobian with FD ---
        for col = 1:m
            upert[col] += du

            tsol, usolPert = solve_rk4(dudt, upert, t0, tf, nnode, foil)
            resPert = compute_g(usolPert[:, begin], usolPert[:, end], foil)

            jac[:, col] = (resPert - res) / du

            upert = copy(u) # reset
        end
        jac[1:n, 1:n] = Matrix(1.0I, n, n)  # Identity matrix of Float64 type

        # --- Newton step ---
        step = -jac \ res
        u += step

        # --- Recalculate g(u) with the new u ---
        tsol, usol = solve_rk4(dudt, u, t0, tf, nnode, foil)
        res = compute_g(usol[:, begin], usol[:, end], foil)
        iter += 1
        if iter > maxIter
            println("No solution in max iters")
            break
        end

        # # --- Print residual ---
        # if (iter % 10 == 0) || (iter == 2)
        #     println("Residual")
        # end
        # println(norm(res))
    end

    return tsol, usol
end
end # end module