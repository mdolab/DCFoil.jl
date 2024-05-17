# --- Julia 1.9---
"""
@File    :   VPM.jl
@Time    :   2024/05/16
@Author  :   Galen Ng
@Desc    :   Vortex panel method for the circulation distribution over an airfoil surface
"""


module VPM

# --- PACKAGES ---
using LinearAlgebra
using Plots
using FLOWMath: atan_cs_safe

# --- DCFoil modules ---
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..DCFoil: DTYPE
using ..SolverRoutines: compute_anglesFromVector

struct Airfoil{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    vortexXY::TM # vortex points [x,y] for each panel, size [2, n] where n is the number of vertices
    controlXY::TM # control points [x,y] for each panel, size [2, n-1] where n is the number of vertices
    panelLengths::TA # panel lengths [m]
    n::TI # number of panel vertices
    sweep::TF # sweep angle [rad]
end

function initialize(xx, yy, control_xy, sweep=0.0)

    # Quick error check
    if size(control_xy)[1] != 2
        error("Control points must be in the form [x,y]")
    end

    nodeCt = length(xx)
    vortex_xy = copy(hcat(xx .* cos(sweep), yy)')

    panelLengths = sqrt.(diff(vortex_xy[XDIM, :]) .^ 2 .+ diff(vortex_xy[YDIM, :]) .^ 2)

    AIRFOIL = Airfoil(vortex_xy, control_xy, panelLengths, nodeCt, sweep)

    P11, P12, P21, P22 = compute_panelMatrix(AIRFOIL)

    Amat = zeros(nodeCt, nodeCt)
    Amat[1:end-1, 1:end-1] += diff(AIRFOIL.vortexXY[XDIM, :]) .* P21 ./ AIRFOIL.panelLengths
    .-diff(AIRFOIL.vortexXY[YDIM, :]) .* P11 ./ AIRFOIL.panelLengths

    Amat[1:end-1, 2:end] += diff(AIRFOIL.vortexXY[XDIM, :]) .* P22 ./ AIRFOIL.panelLengths
    .-diff(AIRFOIL.vortexXY[YDIM, :]) .* P12 ./ AIRFOIL.panelLengths

    # Kutta condition
    Amat[end, 1] = 1.0
    Amat[end, end] = 1.0

    return AIRFOIL, Amat
end

function solve(Airfoil, Amat, V, chord=1.0, Vref=1.0)
    """
    Solve vortex strength and lift and moment
    """

    alpha, Vinf = compute_sweepCorr(Airfoil.sweep, V)

    RHS = zeros(DTYPE, Airfoil.n)
    RHS[1:end-1] = Vinf ./ Airfoil.panelLengths .* (diff(Airfoil.vortexXY[YDIM, :]) .* cos(alpha)
                                                    .-
                                                    diff(Airfoil.vortexXY[XDIM, :] .* sin(alpha)))

    # Airfoil surface vorticity strengths
    γi = Amat \ RHS

    # Total circulation for the airfoil
    Γi = sum(0.5 * (γi[1:end-1] + γi[2:end]) .* Airfoil.panelLengths)

    cℓ = 2.0 * Vinf * Γi / (chord * cos(Airfoil.sweep) * Vref^2)
    cpDist = 1.0 .- (γi ./ Vref) .^ 2
    cm = -sum(
        (
            (
                2.0 * Airfoil.vortexXY[XDIM, 1:end-1] .* γi[1:end-1]
                + Airfoil.vortexXY[XDIM, 1:end-1] .* γi[2:end]
                + Airfoil.vortexXY[XDIM, 2:end] .* γi[1:end-1]
                + 2.0 * Airfoil.vortexXY[XDIM, 2:end] .* γi[2:end]
            )
            .*
            cos(alpha)
            .+
            (
                2.0 * Airfoil.vortexXY[YDIM, 1:end-1] .* γi[1:end-1]
                + Airfoil.vortexXY[YDIM, 1:end-1] .* γi[2:end]
                + Airfoil.vortexXY[YDIM, 2:end] .* γi[1:end-1]
                + 2.0 * Airfoil.vortexXY[YDIM, 2:end] .* γi[2:end]
            )
            .*
            sin(alpha)
        )
        .*
        Airfoil.panelLengths
    ) * Vinf / (3.0 * chord^2 * cos(Airfoil.sweep) * Vref^2)


    return cℓ, cm, Γi, cpDist
end

function compute_panelMatrix(Airfoil)
    """
    Computes the panel matrix for the VPM
    """

    # Control points
    xc = Airfoil.controlXY[XDIM, :]
    yc = Airfoil.controlXY[YDIM, :]

    # Starting vertex of panel
    xx = Airfoil.vortexXY[XDIM, 1:end-1]
    yy = Airfoil.vortexXY[YDIM, 1:end-1]

    # Ending vertex of panel
    x1 = Airfoil.vortexXY[XDIM, 2:end]
    y1 = Airfoil.vortexXY[YDIM, 2:end]

    l = Airfoil.panelLengths

    # Code is slightly different from Reid 2020 b/c Julia is column-major
    matrix1 = (x1 .- xx)' .* (xc .- xx')
    matrix2 = (y1 .- yy)' .* (yc .- yy')
    matrix3 = -(y1 .- yy)' .* (xc .- xx')
    matrix4 = (x1 .- xx)' .* (yc .- yy')
    # matSum = matrix1 .+ matrix2
    η = zeros(DTYPE, size(matrix3))
    ξ = zeros(DTYPE, size(matrix3))
    for ii in 1:size(matrix3)[1]
        ξ[ii, :] = (1 ./ l) .* (matrix1[ii, :] .+ matrix2[ii, :])
        η[ii, :] = (1 ./ l) .* (matrix3[ii, :] .+ matrix4[ii, :])
    end
    
    # Matrix
    arg1 = zeros(DTYPE, size(η))
    arg2 = zeros(DTYPE, size(η))
    
    # Matrix
    numerator = zeros(DTYPE, size(η))
    denominator = zeros(DTYPE, size(η))
    for ii in 1:size(η)[1]
        arg1[ii, :] = η[ii, :] .* l
        arg2[ii, :] = η[ii, :] .^ 2 .+ ξ[ii, :] .^ 2 .- ξ[ii, :] .* l
        numerator[ii, :] = η[ii, :] .^ 2 + ξ[ii, :] .^ 2
        denominator[ii, :] = η[ii, :] .* η[ii, :] .+ (ξ[ii, :] .- l) .* (ξ[ii, :] .- l)
    end
    # Everything up to here is debugged, but the calculations for P11, P12, P21, P22 are not correct
    # TODO: I think Φ is wrong
    Φ = atan_cs_safe.(arg1, arg2)
    Ψ = 0.5 * log.(numerator ./ denominator)
    # Vector
    constant = 2 .* π .* l .^ 2
    
    
    # Vectors
    XY11 = (x1 .- xx) ./ constant
    XY12 = -(y1 .- yy) ./ constant
    XY21 = (y1 .- yy) ./ constant
    XY22 = (x1 .- xx) ./ constant

    P11_pre = zeros(DTYPE, size(η))
    P12_pre = zeros(DTYPE, size(η))
    P21_pre = zeros(DTYPE, size(η))
    P22_pre = zeros(DTYPE, size(η))

    mat1 = zeros(DTYPE, size(η))
    mat2 = zeros(DTYPE, size(η))
    vec1 = zeros(DTYPE, size(η))
    for ii in 1:size(η)[1]
        vec1[ii, :] = (l .- ξ[ii, :])
        mat1[ii, :] = vec1[ii, :] .* Φ[ii, :]
        mat2[ii, :] = η[ii, :] .* Ψ[ii, :]
        P11_pre[ii, :] = mat1[ii, :] + mat2[ii, :]
        P12_pre[ii, :] = (ξ[ii, :] .* Φ[ii, :]) .- (η[ii, :] .* Ψ[ii, :])
        P21_pre[ii, :] = η[ii, :] .* Φ[ii, :] .- (l[ii] .- ξ[ii, :]) .* Ψ[ii, :] .- l[ii]
        P22_pre[ii, :] = -(η[ii, :] .* Φ[ii, :]) .- ξ[ii, :] .* Ψ[ii, :] .+ l[ii]
    end
    # P11_pre = (l .- ξ) .* Φ .+ η .* Ψ
    # P12_pre = (ξ .* Φ) .- (η .* Ψ)
    # P21_pre = η .* Φ .- (l .- ξ) .* Ψ .- l
    # P22_pre = -(η .* Φ) .- ξ .* Ψ .+ l


    P11 = XY11 .* P11_pre .+ XY12 .* P21_pre
    P12 = XY11 .* P12_pre .+ XY12 .* P22_pre
    P21 = XY21 .* P11_pre .+ XY22 .* P21_pre
    P22 = XY21 .* P12_pre .+ XY22 .* P22_pre

    # Debug code
    # η::Matrix{DTYPE} = (1 ./ l) .* (matrix1) .+ (1 ./ l) .* (matrix2)
    plot(eachindex(η[:, 1]), mat1[1, :], label="row 0")
    # plot!(eachindex(η[:, 1]), (1 ./ l) .* matrix1[1, :], label="matrix1")
    # plot!(eachindex(η[:, 1]), (1 ./ l) .* matrix2[1, :], label="matrix2")
    plot!(eachindex(η[:, 1]), mat1[11, :], label="row 10")
    savefig("eta.png")

    return P11, P12, P21, P22
end

function compute_sweepCorr(angle, V)
    """

    """
    alpha, beta, Vinf = compute_anglesFromVector(V)
    Ca = cos(alpha)
    Sa = sin(alpha)
    Cb = cos(beta)
    Sb = sin(beta)
    sqrtSaSb = sqrt(1.0 - Sa * Sa * Sb * Sb)
    alphaCorr = atan_cs_safe(tan(alpha) * Cb, cos(angle - beta))
    VinfCorr = Vinf * sqrt(Ca^2 * cos(angle - beta)^2 + Sa^2 * Cb^2) / sqrtSaSb
    return alphaCorr, VinfCorr
end
end