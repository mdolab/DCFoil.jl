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

using DelimitedFiles, Debugger, Plots

"""
Struct to hold the airfoil geometry discretization
"""
struct AirfoilMesh{TI,T<:Number,TM<:AbstractMatrix}
    vortexXY::TM        # vortex points [x,y] for each panel, size [2, n] where n is the number of vertices
    controlXY::TM       # control points [x,y] for each panel, size [2, n-1] where n is the number of vertices
    panelLengths::AbstractVector    # panel lengths [m]
    n::TI               # number of panel vertices
    sweep::T           # sweep angle [rad]
end

"""
Discretize the airfoil into panels and setup the VPM linear systems to solve

xx: x-coordinates of the airfoil vertices
yy: y-coordinates of the airfoil vertices
control_xy: control points for the VPM (center of panels)
"""
function setup_VPM(xx, yy, control_xy, sweep=0.0)

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

    step1mat = mat1 .- mat2
    step1Amat = cat(step1mat, zeros(1, size(step1mat)[1]), dims=1) #row on bottom
    step1Amat = cat(step1Amat, zeros(size(step1Amat)[1], 1), dims=2) #col on right

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
    # [A_ij][γ] = [RHS_i] or expanded out
    #  [               [  γ1
    # 
    #   1 0 ... 0 1]     γn+1]
    γi = Amat \ RHS # bottleneck code

    # Total circulation for the airfoil
    Γi = sum(0.5 * (γi[1:(end-1)] .+ γi[2:end]) .* Airfoil.panelLengths)

    # --- Correct to total section circulation via the FS effect ---
    corrFactor = correct_FS(hcRatio)
    Γi *= corrFactor

    cℓ = 2.0 * Vinf * Γi / (chord * csweep * Vref^2)
    cpDist = 1.0 .- (γi ./ Vref) .^ 2
    cm = -sum(
        (
            (
                2.0 * Airfoil.vortexXY[XDIM, 1:(end-1)] .* γi[1:(end-1)]
                + Airfoil.vortexXY[XDIM, 1:(end-1)] .* γi[2:end]
                + Airfoil.vortexXY[XDIM, 2:end] .* γi[1:(end-1)]
                + 2.0 * Airfoil.vortexXY[XDIM, 2:end] .* γi[2:end]
            )
            .*
            calpha
            .+
            (
                2.0 * Airfoil.vortexXY[YDIM, 1:(end-1)] .* γi[1:(end-1)]
                + Airfoil.vortexXY[YDIM, 1:(end-1)] .* γi[2:end]
                + Airfoil.vortexXY[YDIM, 2:end] .* γi[1:(end-1)]
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

function correct_FS(hcRatio)
    """Use a high Fn theoretical correction value to the section lift slope and lift based on Wadlin et al. 1955"""

    hcTerm = 16.0 * hcRatio^2
    corrFactor = (1.0 + hcTerm) / (2.0 + hcTerm)
    return corrFactor
end


function compute_panelMatrix(Airfoil)
    """
    Computes the panel matrix for the VPM
        I think you need to read Phillip's or Anderson's textbook
    """

    nctrl = Airfoil.n - 1

    # Control points
    xc = Airfoil.controlXY[XDIM, :]
    yc = Airfoil.controlXY[YDIM, :]

    # Starting vertex of panel
    xx = Airfoil.vortexXY[XDIM, 1:(end-1)]
    yy = Airfoil.vortexXY[YDIM, 1:(end-1)]

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

# ==============================================================================
#                         Chen 2012 vortex method
# ==============================================================================
function setup_linsys(xx, yy, control_xy, ν, μ, Uinf)
    Npan = size(control_xy, 2)
    aij = zeros(Npan, Npan)
    bi = zeros(Npan)
    kSweep = 0:0.01:100
    M = length(kSweep)

    Aij = zeros(ComplexF64, size(control_xy, 2), size(control_xy, 2))
    Bij = zeros(ComplexF64, size(control_xy, 2), size(control_xy, 2))
    Ci = zeros(ComplexF64, size(control_xy, 2))
    z1 = xx[1] + yy[1] * im # needed for Kutta condition
    z̄1 = conj(z1)
    zp1 = xx[2] + yy[2] * im
    zN = xx[end-1] + yy[end-1] * im
    zN1 = xx[end] + yy[end] * im

    t1 = (zp1 - z1) / abs(zp1 - z1)
    tN = (zN1 - zN) / abs(zN1 - zN)
    # println("t1: ", t1, " tN: ", tN)

    println("$Uinf m/s")

    pplot = scatter([real(z1)], [imag(z1)], color=:blue, markershape=:circle)
    pplot = scatter([real(z̄1)], [imag(z̄1)], color=:green, markershape=:circle)

    # Loop over all panels
    for ii in 1:Npan # first loop rows

        # Location of the control point where we evaluate the influence
        ζi = control_xy[XDIM, ii] + control_xy[YDIM, ii] * im

        bin = 0.0 # bi

        # println("panel: $ii")
        # println("panel location: $(real(ζi)), $(imag(ζi))")
        scatter!(pplot, [real(ζi)], [imag(ζi)], color=:red, markershape=:x, legend=false, xlabel="X", ylabel="Y")

        for (jj, panel) in enumerate(eachcol(control_xy))

            # Mesh point
            zj = xx[jj] + yy[jj] * im
            # Neighbor point
            zj1 = xx[jj+1] + yy[jj+1] * im

            # Images of the mesh points
            z̄j = conj(zj)
            z̄j1 = conj(zj1)

            # Tangent vectors
            tj = (zj1 - zj) / abs(zj1 - zj)
            t̄j = conj(tj)

            # ζj panel center
            ζj = panel[XDIM] + 1im*panel[YDIM]
            ζ̄j = panel[XDIM] - 1im*panel[YDIM]

            # Evaluate the wave integral numerically
            SUM = 0.0
            SUM2 = 0.0
            # for m = 1:(M-1)
            #     km = kSweep[m]
            #     km1 = kSweep[m+1]

            #     eⁱ1 = exp(-1im * km * (ζi - z̄j1))
            #     eⁱ2 = exp(-1im * km * (ζi - z̄j))
            #     eⁱ = exp(-1im * km * (ζi - z̄1))

            #     SUM += (eⁱ1 - eⁱ2) * log((km1 - ν + 1im * μ) / (km - ν + 1im * μ))
            #     SUM2 += (eⁱ) * log((km1 - ν + 1im * μ) / (km - ν + 1im * μ))
            # end
            # TODO: leave zeroed out for now
            SUM = 0.0
            SUM2 = 0.0

            lnζz = log((ζi - zj1) / (ζi - zj))
            if ii == jj
                lnζz = -1im*π
            end
            # TODO: PICKUP HERE, SOMETHING IS NOT RIGHT BECAUSE THE VALUE SHOULD GO TO ZERO AT ZERO ALPHA
            self = -1im * t̄j * lnζz
            image = 1im * tj * (1im * ν - μ) / (1im * ν + μ) * log((ζi - z̄j1) / (ζi - z̄j))

            # What I think it is?
            self = -1im * tj * log((ζi - zj1) / (ζi - zj))
            image = 1im * t̄j * (1im * ν - μ) / (1im * ν + μ) * log((ζi - z̄j1) / (ζi - z̄j))

            waveTerm = -2 * tj * ν / (1im * ν + μ) * SUM

            Aij[ii, jj] = self
            + image
            + waveTerm
            println("zj: ", zj)
            println("zj1: ", zj1)
            println("Aij[$ii, $jj]: $(Aij[ii, jj])")
            println("t̄j: ", t̄j)
            println("ln(ζ-z) term: ", lnζz)
            bug

            self = -t̄j * log((ζi - zj1) / (ζi - zj))
            image = tj * (1im * ν - μ) / (1im * ν + μ) * log((ζi - z̄j1) / (ζi - z̄j))

            # What I think it is?
            self = -tj * log((ζi - zj1) / (ζi - zj))
            image = t̄j * (1im * ν - μ) / (1im * ν + μ) * log((ζi - z̄j1) / (ζi - z̄j))

            Bij[ii, jj] = self
            +image
            +im*waveTerm

            self = 1 / (ζi - z1)
            image = 1 / (ζi - z̄1)
            waveTerm3 = 2 * im * ν * SUM2
            Ci[ii] = 1im * (
                self
                + image
                + waveTerm3
            )
            bin += real(im * tj) * Bij[ii, jj]

            # plot the matrices as color maps
            p1 = heatmap(real(Aij)', yflip=true, title="re(Aij)", color=:RdBu, clims=(-4, 4))
            xlabel!("Column j")
            ylabel!("Row i")
            savefig(p1, "Aij.png")
            p1 = heatmap(real(Bij)', yflip=true, title="re(Bij)", color=:RdBu, clims=(-4, 4))
            xlabel!("Column j")
            ylabel!("Row i")
            savefig(p1, "Bij.png")

            println("Green func loc: $(real(zj)), $(imag(zj))")


            scatter!(pplot, [real(zj)], [imag(zj)], color=:blue, markershape=:circle)
            scale = 20
            plot!(pplot, [real(ζj), real(ζj)+real(tj)/scale], [imag(ζj), imag(ζj)+imag(tj)/scale], arrow=true, color=:black)

            scatter!(pplot, [real(z̄j)], [imag(z̄j)], color=:green, markershape=:circle, size=(600, 1200))
            plot!(pplot, [real(ζ̄j), real(ζ̄j)+real(t̄j)/scale], [imag(ζ̄j), imag(ζ̄j)+imag(t̄j)/scale], arrow=true, color=:black)
            savefig(pplot, "panel_locations_$(ii).png")
        end


        aij[ii, :] = real.(Aij[ii, :])
        bi[ii] = Uinf * real(bin)
    end

    ainp1 = -real(Ci)

    writedlm("reAij.csv", real(Aij), ',')
    writedlm("reBij.csv", real(Bij), ',')
    writedlm("reCi.csv", real(Ci), ',')


    Bvec = vcat(bi, -Uinf * (real(t1) + real(tN)))

    # Add the column
    Amat = hcat(aij, ainp1)

    # println(size(Amat))
    an1 = zeros(size(aij, 1)+1)
    an1[1] = 1.0
    an1[end-1] = 1.0

    # Add the row
    Amat = vcat(Amat, transpose(an1))

    return Amat, Bvec
end

function solve_gamma(aij, ain1, bi)

    N = length(bi)
    Aij = zeros(N + 1, N + 1)
    Bi = zeros(N + 1)

    Aij[1:N, 1:N] = aij
    Aij[:, N+1] = ain1
    Aij[N+1, 1] = 1
    Aij[N+1, N] = 1

    Bi[1:N] = bi
    Bi[N+1] = -Uinf * (real(t1) - real(tN))

    gammas = Aij \ Bi

    return gammas
end

"""
ζ - field point in imaginary units
z - location of the Green function
ν - nondimensional wave no.
μ - dissipation

Computes this Green function
G = ln(ζ - z) + νi - μ               2iν    /
                ------ ln(ζ - z̄) + ------  |
                νi + μ             iν + μ  /
"""
function compute_DissipativeKelvinGreenVortex(ζ, z, ν, μ)


    function compute_kInt(k)
        # TODO: make this wave integral more efficient
        int = exp(-im * k * (ζ - conj(z))) /
              (k - ν + μ * im)
        return int
    end
    dk = 0.01
    kSweep = 0:dk:100
    heights = compute_kInt.(kSweep)
    waveInt = sum(heights * dk)

    # H is called F in paper
    H = (im * ν - μ) / (im * ν + μ) * log(ζ - conj(z))
    + 2 * im * ν / (im * ν + μ) * waveInt

    ln = log(ζ - z)
    Gsource = ln + H

    Gvort = im * Gsource

    return Gsource, Gvort, ln, H
end

"""
Chen and Price approach
"""
function setup_linsysdipole(xx, yy, control_xy, ν, μ, Uinf)
    Npan = size(control_xy, 2)
    aij = zeros(Npan, Npan)
    bi = zeros(Npan)
    kSweep = 0:0.01:100
    M = length(kSweep)

    Aij = zeros(Float64, size(control_xy, 2), size(control_xy, 2))
    # Bij = zeros(ComplexF64, size(control_xy, 2), size(control_xy, 2))
    Ai = zeros(Float64, size(control_xy, 2))
    z1 = xx[1] + yy[1] * im # needed for Kutta condition
    # z̄1 = conj(z1)
    # zp1 = xx[2] + yy[2] * im
    zN = xx[end-1] + yy[end-1] * im
    zN1 = xx[end] + yy[end] * im
    z̄N1 = conj(zN1)
    # @bp

    tN = (zN1 - zN) / abs(zN1 - zN)
    tT = tN
    t̄T = conj(tT)

    # Loop over all panels
    for ii in 1:Npan # first loop rows

        # Location of the control point where we evaluate the influence
        # zi = xx[ii] + yy[ii] * im
        # zi1 = xx[ii+1] + yy[ii+1] * im
        # ζi = 0.5*(zi + zi1)
        ζi = control_xy[XDIM, ii] + control_xy[YDIM, ii] * im

        SUM2 = 0.0
        for m = 1:(M-1)
            km = kSweep[m]
            km1 = kSweep[m+1]
            eⁱ = exp(-1im * km * (ζi - z̄N1))
            SUM2 += (eⁱ) * log((km1 - ν + 1im * μ) / (km - ν + 1im * μ))
        end
        # println("SUM2: ", SUM2)

        self = 1im*log(-tT/(ζi - zN1))
        image = 1im * (1im*ν-μ) / (1im*ν + μ) * log(-(ζi - z̄N1)/t̄T)
        waveTerm2 = real(1im*2im*ν/(1im * ν + μ) * SUM2)
        Ai[ii] = real(self + image) + waveTerm2
        # println("image: ", image)
        # println("waveTerm2: ", waveTerm2)
        lnzlist = []
        lnz̄list = []

        for jj in 1:Npan
            # Mesh point
            zj = xx[jj] + yy[jj] * im

            # Neighbor point
            zj1 = xx[jj+1] + yy[jj+1] * im

            # Images of the mesh points
            z̄j = conj(zj)
            z̄j1 = conj(zj1)

            # Evaluate the wave integral numerically
            SUM = 0.0
            for m = 1:(M-1)
                km = kSweep[m]
                km1 = kSweep[m+1]

                eⁱ1 = exp(-1im * km1 * (ζi - z̄j1))
                eⁱ2 = exp(-1im * km1 * (ζi - z̄j))

                SUM += (eⁱ1 - eⁱ2) * log((km1 - ν + 1im * μ) / (km - ν + 1im * μ))
            end


            lnζz = log((ζi - zj1) / (ζi - zj))
            lnζz̄ = log((ζi - z̄j1) / (ζi - z̄j))

            # KEEP THIS TERM OTHERWISE IT'S OSCILLATORY WITH THE JUMP IN THE LOG FUNCTION
            if ii == jj
                # lnζz = -1im*π
                lnζz = 1im*π
            end

            # --- Paper ---
            self = 1im * lnζz
            image = (ν + 1im*μ) / (1im * ν + μ) * lnζz̄

            waveTerm = real(2 * ν / (1im * ν + μ) * SUM)

            Aij[ii, jj] = real(self+image) + waveTerm
            # println("image: ", image)
            # println("waveTerm: ", waveTerm)
        end
    end



    aij[:, :] .= Aij[:, :]

    for ii in 1:Npan
        aij[ii, end] += Ai[ii]
        aij[ii, 1] -= Ai[ii]
        ξ = (xx[ii] + xx[ii+1]) * 0.5
        bi[ii] = 2π*Uinf*ξ
    end

    aij += I(Npan)*2π

    Amat = copy(aij)

    Bvec = copy(bi)


    return Amat, Bvec
end