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

function setup_VPM(xx, yy, control_xy, sweep=0.0)
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
    # println("control_xy pre: $(control_xy[XDIM, :])\n")
    # control_xy_work[XDIM, :] .= copy(control_xy[XDIM, :]) .* cos(sweep) # change control points to sweep angle
    control_xy_xmod = reshape(copy(control_xy[XDIM, :]) .* cos(sweep), 1, nodeCt - 1) # change control points to sweep angle
    control_xy = cat(control_xy_xmod, reshape(control_xy[YDIM, :], 1, nodeCt - 1), dims=1)
    # println("sweep: $(sweep)\n")
    # println("control_xy post: $(control_xy[XDIM, :])\n")

    panelLengths = .√(diff(vortex_xy[XDIM, :]) .^ 2 .+ diff(vortex_xy[YDIM, :]) .^ 2)

    AIRFOIL = AirfoilMesh(vortex_xy, control_xy, panelLengths, nodeCt, sweep)

    P11, P12, P21, P22 = compute_panelMatrix(AIRFOIL)

    dx = (diff(AIRFOIL.vortexXY[XDIM, :]))
    dy = (diff(AIRFOIL.vortexXY[YDIM, :]))
    mat1 = (dx) .* P21 ./ (AIRFOIL.panelLengths)
    mat2 = (dy) .* P11 ./ (AIRFOIL.panelLengths)

    # Amat = zeros(nodeCt, nodeCt)
    # Amat[1:end-1, 1:end-1] += mat1 .- mat2
    step1mat = mat1 .- mat2
    step1Amat = cat(step1mat, zeros(1, size(step1mat)[1]), dims=1) #row on bottom
    step1Amat = cat(step1Amat, zeros(size(step1Amat)[1], 1), dims=2) #col on right

    # # Debug code
    # matplot = mat1
    # plot(eachindex(matplot[:, 1]), matplot[1, :], label="row 0")
    # plot!(eachindex(matplot[:, 1]), matplot[11, :], label="row 10")
    # plot!(eachindex(matplot[:, 1]), matplot[end-9, :], label="row -10")
    # plot!(eachindex(matplot[:, 1]), matplot[end, :], label="row end")
    # savefig("eta.png")

    mat11 = (dx) .* P22 ./ (AIRFOIL.panelLengths)
    mat22 = (dy) .* P12 ./ (AIRFOIL.panelLengths)

    # Amat[1:end-1, 2:end] += mat11 .- mat22
    step2mat = mat11 .- mat22
    step2Amat = cat(step2mat, zeros(1, size(step2mat)[1]), dims=1) # row on bottom
    step2Amat = cat(zeros(size(step2Amat)[1], 1), step2Amat, dims=2) # col on LEFT

    # Kutta condition
    kutta = transpose(vcat(1.0, zeros(nodeCt - 2), 1.0))
    kuttaMat = cat(zeros(nodeCt - 1, size(kutta)[2]), kutta, dims=1) # Kutta row on bottom

    Amat = step1Amat + step2Amat + kuttaMat

    return AIRFOIL, Amat
end

function solve_VPM(Airfoil, Amat, V, chord=1.0, Vref=1.0, hcRatio=50.0)
    """
    Solve vortex strength and lift and moment

    Airfoil: AirfoilMesh struct
    Amat: Panel matrix of influences
    V: Freestream velocity vector [U, V, W]
    Outputs:
    cℓ: Sectional lift coefficient
    cm: Moment coefficient about leading edge
    """

    if length(V) != 3
        error("Velocity must be a 3D vector [U, V, W]")
    end

    alpha, Vinf = compute_sweepCorr(Airfoil.sweep, V)

    calpha = cos(alpha)
    salpha = sin(alpha)
    csweep = cos(Airfoil.sweep)
    # RHS = zeros(typeof(Vinf), Airfoil.n)
    # RHS[1:end-1] = Vinf ./ Airfoil.panelLengths .* (diff(Airfoil.vortexXY[YDIM, :]) .* cos(alpha)
    #                                                 .-
    #                                                 diff(Airfoil.vortexXY[XDIM, :] .* sin(alpha)))
    RHS_noTE = Vinf ./ Airfoil.panelLengths .* (diff(Airfoil.vortexXY[YDIM, :]) .* calpha
                                                .-
                                                diff(Airfoil.vortexXY[XDIM, :] .* salpha))
    RHS = vcat(RHS_noTE, 0.0)

    # Airfoil surface vorticity strengths
    γi = Amat \ RHS # bottleneck code

    # Total circulation for the airfoil
    Γi = sum(0.5 * (γi[1:end-1] .+ γi[2:end]) .* Airfoil.panelLengths)

    # --- Correct to total section circulation via the FS effect ---
    corrFactor = (1.0 + 16.0 * hcRatio^2) / (2.0 + 16.0 * hcRatio^2)
    Γi *= corrFactor

    cℓ = 2.0 * Vinf * Γi / (chord * csweep * Vref^2)
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
            calpha
            .+
            (
                2.0 * Airfoil.vortexXY[YDIM, 1:end-1] .* γi[1:end-1]
                + Airfoil.vortexXY[YDIM, 1:end-1] .* γi[2:end]
                + Airfoil.vortexXY[YDIM, 2:end] .* γi[1:end-1]
                + 2.0 * Airfoil.vortexXY[YDIM, 2:end] .* γi[2:end]
            )
            .*
            salpha
        )
        .*
        Airfoil.panelLengths
    ) * Vinf / (3.0 * chord^2 * csweep * Vref^2)


    return cℓ, cm, Γi, cpDist
end

function compute_panelMatrix(Airfoil)
    """
    Computes the panel matrix for the VPM
    """

    nctrl = Airfoil.n - 1

    # Control points
    xc = Airfoil.controlXY[XDIM, :]
    yc = Airfoil.controlXY[YDIM, :]

    # Starting vertex of panel
    xx = Airfoil.vortexXY[XDIM, 1:end-1]
    yy = Airfoil.vortexXY[YDIM, 1:end-1]

    # Ending vertex of panel
    x1 = Airfoil.vortexXY[XDIM, 2:end]
    y1 = Airfoil.vortexXY[YDIM, 2:end]

    ℓ = Airfoil.panelLengths
    ℓmat = transpose(repeat(ℓ, 1, nctrl)) # ℓ matrix

    # Code is slightly different from Reid 2020 b/c Julia is column-major
    mat1 = transpose(x1 .- xx) .* (xc .- transpose(xx))
    mat2 = transpose(y1 .- yy) .* (yc .- transpose(yy))
    mat3 = -transpose(y1 .- yy) .* (xc .- transpose(xx))
    mat4 = transpose(x1 .- xx) .* (yc .- transpose(yy))

    # η = (1 / l) * ( (x1 - xx)(yc - yy) - (y1 - yy)(xc - xx))
    divℓ = 1.0 ./ ℓmat
    ξ = divℓ .* (mat1 .+ mat2)
    η = divℓ .* (mat3 .+ mat4)
    ηSq = η .^ 2
    ξSq = ξ .^ 2

    # Matrix
    ellmatxi1 = ℓmat .- ξ
    arg1 = η .* ℓmat
    arg2 = ηSq .+ ξSq .- ξ .* ℓmat
    numerator = ηSq + ξSq
    denominator = ηSq .+ (-ellmatxi1) .^ 2

    # Φ = tan⁻¹( ηl / (η² + (ξ - l)²))
    Φ = atan_cs_safe.(arg1, arg2)
    Ψ = 0.5 * log.(numerator ./ denominator)

    # Vector
    constant = 2π .* ℓ .^ 2

    # Vectors
    XY11 = (x1 .- xx) ./ constant
    XY12 = -(y1 .- yy) ./ constant
    XY21 = (y1 .- yy) ./ constant
    XY22 = (x1 .- xx) ./ constant

    mat1 = ellmatxi1 .* Φ
    mat2 = η .* Ψ
    P11_pre = mat1 .+ mat2
    P12_pre = (ξ .* Φ) .- (η .* Ψ)
    P21_pre = η .* Φ .- ellmatxi1 .* Ψ .- ℓmat
    P22_pre = -(η .* Φ) .- ξ .* Ψ .+ ℓmat

    P11 = transpose(XY11) .* P11_pre .+ transpose(XY12) .* P21_pre
    P12 = transpose(XY11) .* P12_pre .+ transpose(XY12) .* P22_pre
    P21 = transpose(XY21) .* P11_pre .+ transpose(XY22) .* P21_pre
    P22 = transpose(XY21) .* P12_pre .+ transpose(XY22) .* P22_pre

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
    alpha, beta, Vinf = compute_cartAnglesFromVector(V)
    Ca = cos(alpha)
    Sa = sin(alpha)
    Cb = cos(beta)
    Sb = sin(beta)
    SaSb = √(1.0 - Sa * Sa * Sb * Sb)
    alphaCorr = atan_cs_safe(tan(alpha) * Cb, cos(angle - beta))
    VinfCorr = Vinf * √(Ca^2 * cos(angle - beta)^2 + Sa^2 * Cb^2) / SaSb

    return alphaCorr, VinfCorr
end
