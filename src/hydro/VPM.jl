# --- Julia 1.9---
"""
@File    :   VPM.jl
@Time    :   2024/05/16
@Author  :   Galen Ng
@Desc    :   Vortex panel method for the circulation distribution over an airfoil surface.
             This code more accurately computes the sectional lift (cℓ) and 
             sectional lift slope (∂cℓ / ∂α ≐ 1 / rad) than thin airfoil theory and introduces the
             "nonlinear" lift slope into the nonlinear lifting line
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

struct AirfoilMesh{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    """
    Struct to hold the airfoil geometry discretization
    """
    vortexXY::TM        # vortex points [x,y] for each panel, size [2, n] where n is the number of vertices
    controlXY::TM       # control points [x,y] for each panel, size [2, n-1] where n is the number of vertices
    panelLengths::TA    # panel lengths [m]
    n::TI               # number of panel vertices
    sweep::TF           # sweep angle [rad]
end

function setup(xx, yy, control_xy, sweep=0.0)
    """
    Discretize the airfoil into panels and setup the VPM linear systems to solve

    xx: x-coordinates of the airfoil vertices
    yy: y-coordinates of the airfoil vertices
    control_xy: control points for the VPM (center of panels)
    """

    # Quick error check
    if size(control_xy)[1] != 2
        error("Control points must be in the form [x,y]")
    end

    nodeCt = length(xx)
    vortex_xy = copy(transpose(hcat(xx .* cos(sweep), yy)))

    panelLengths = sqrt.(diff(vortex_xy[XDIM, :]) .^ 2 .+ diff(vortex_xy[YDIM, :]) .^ 2)

    AIRFOIL = AirfoilMesh(vortex_xy, control_xy, panelLengths, nodeCt, sweep)

    P11, P12, P21, P22 = compute_panelMatrix(AIRFOIL)

    Amat = zeros(nodeCt, nodeCt)
    dx = (diff(AIRFOIL.vortexXY[XDIM, :]))
    dy = (diff(AIRFOIL.vortexXY[YDIM, :]))
    mat1 = (dx) .* P21 ./ (AIRFOIL.panelLengths)
    mat2 = (dy) .* P11 ./ (AIRFOIL.panelLengths)
    Amat[1:end-1, 1:end-1] += mat1 .- mat2

    # # Debug code
    # matplot = mat1
    # plot(eachindex(matplot[:, 1]), matplot[1, :], label="row 0")
    # plot!(eachindex(matplot[:, 1]), matplot[11, :], label="row 10")
    # plot!(eachindex(matplot[:, 1]), matplot[end-9, :], label="row -10")
    # plot!(eachindex(matplot[:, 1]), matplot[end, :], label="row end")
    # savefig("eta.png")

    mat1 = (dx) .* P22 ./ (AIRFOIL.panelLengths)
    mat2 = (dy) .* P12 ./ (AIRFOIL.panelLengths)
    Amat[1:end-1, 2:end] += mat1 .- mat2


    # Kutta condition
    Amat[end, 1] = 1.0
    Amat[end, end] = 1.0

    return AIRFOIL, Amat
end

function solve(Airfoil, Amat, V, chord=1.0, Vref=1.0)
    """
    Solve vortex strength and lift and moment

    Airfoil: AirfoilMesh struct
    Amat: Panel matrix of influences
    V: Freestream velocity vector [U, V, W]
    """

    if length(V) != 3
        error("Velocity must be a 3D vector [U, V, W]")
    end

    alpha, Vinf = compute_sweepCorr(Airfoil.sweep, V)

    RHS = zeros(DTYPE, Airfoil.n)
    RHS[1:end-1] = Vinf ./ Airfoil.panelLengths .* (diff(Airfoil.vortexXY[YDIM, :]) .* cos(alpha)
                                                    .-
                                                    diff(Airfoil.vortexXY[XDIM, :] .* sin(alpha)))

    # Airfoil surface vorticity strengths
    γi = Amat \ RHS

    # # Debug code
    # plot(eachindex(Amat[1,:]), Amat[1,:], label="row 0")
    # plot!(eachindex(Amat[1,:]), Amat[11,:], label="row 10")
    # plot!(eachindex(Amat[1,:]), Amat[end-10,:], label="row 10")
    # plot!(eachindex(Amat[1,:]), Amat[end,:], label="row end")
    # ylims!(-1, 1)
    # savefig("eta.png")

    # Total circulation for the airfoil
    Γi = sum(0.5 * (γi[1:end-1] .+ γi[2:end]) .* Airfoil.panelLengths)

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
    mat1 = transpose(x1 .- xx) .* (xc .- transpose(xx))
    mat2 = transpose(y1 .- yy) .* (yc .- transpose(yy))
    mat3 = -transpose(y1 .- yy) .* (xc .- transpose(xx))
    mat4 = transpose(x1 .- xx) .* (yc .- transpose(yy))

    # η = (1 / l) * ( (x1 - xx)(yc - yy) - (y1 - yy)(xc - xx))
    η = zeros(DTYPE, size(mat3))
    ξ = zeros(DTYPE, size(mat3))
    for ii in 1:size(mat3)[1]
        ξ[ii, :] = (1 ./ l) .* (mat1[ii, :] .+ mat2[ii, :])
        η[ii, :] = (1 ./ l) .* (mat3[ii, :] .+ mat4[ii, :])
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
    # Φ = tan⁻¹( ηl / (η² + (ξ - l)²))
    Φ = atan_cs_safe.(arg1, arg2)
    Ψ = 0.5 * log.(numerator ./ denominator)
    # Vector
    constant = 2π .* l .^ 2


    # Vectors
    XY11 = (x1 .- xx) ./ constant
    XY12 = -(y1 .- yy) ./ constant
    XY21 = (y1 .- yy) ./ constant
    XY22 = (x1 .- xx) ./ constant

    P11_pre = zeros(DTYPE, size(η))
    P12_pre = zeros(DTYPE, size(η))
    P21_pre = zeros(DTYPE, size(η))
    P22_pre = zeros(DTYPE, size(η))

    vec1 = zeros(DTYPE, size(η))
    for ii in 1:size(η)[1]
        vec1[ii, :] = (l .- ξ[ii, :])
        mat1[ii, :] = vec1[ii, :] .* Φ[ii, :]
        mat2[ii, :] = η[ii, :] .* Ψ[ii, :]
        P11_pre[ii, :] = mat1[ii, :] + mat2[ii, :]
        P12_pre[ii, :] = (ξ[ii, :] .* Φ[ii, :]) .- (η[ii, :] .* Ψ[ii, :])


        P21_pre[ii, :] = η[ii, :] .* Φ[ii, :] .- (l .- ξ[ii, :]) .* Ψ[ii, :] .- l
        P22_pre[ii, :] = -(η[ii, :] .* Φ[ii, :]) .- ξ[ii, :] .* Ψ[ii, :] .+ l
    end
    # P11_pre = (l .- ξ) .* Φ .+ η .* Ψ
    # P12_pre = (ξ .* Φ) .- (η .* Ψ)
    # P21_pre = η .* Φ .- (l .- ξ) .* Ψ .- l
    # P22_pre = -(η .* Φ) .- ξ .* Ψ .+ l

    P11 = transpose(XY11) .* P11_pre .+ transpose(XY12) .* P21_pre
    P12 = transpose(XY11) .* P12_pre .+ transpose(XY12) .* P22_pre
    P21 = transpose(XY21) .* P11_pre .+ transpose(XY22) .* P21_pre
    P22 = transpose(XY21) .* P12_pre .+ transpose(XY22) .* P22_pre

    # # Debug code
    # matplot = P21_pre
    # plot(eachindex(matplot[:, 1]), matplot[1, :], label="row 0")
    # plot!(eachindex(matplot[:, 1]), matplot[11, :], label="row 10")
    # plot!(eachindex(matplot[:, 1]), matplot[end-9, :], label="row -10")
    # plot!(eachindex(matplot[:, 1]), matplot[end, :], label="row end")
    # savefig("eta.png")

    return P11, P12, P21, P22
end

function compute_sweepCorr(angle, V)
    """
    Correct the angle of attack and freestream velocity for the sweep angle

    Inputs:
    -------
    angle - Sweep angle [rad]
    V - Velocity vector [U, V, W]
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