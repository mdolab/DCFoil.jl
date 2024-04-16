# --- Julia 1.9---
"""
@File    :   Interpolation.jl
@Time    :   2024/04/12
@Author  :   Galen Ng
@Desc    :   Interpolation schemes
"""

module Interpolation

using Zygote
using ..DCFoil: DTYPE

function lagrangeArrInterp(x0, y0, m::Int64, n::Int64, d::Int64, x)
    """
    Interpolate/extrapolate polynomials of order 'd-1'
    Providing 'd' points of array of size m x n, we obtain inter/extrapolant order 'd-1'
    Comes from Eirikur's DLM4PY code

    Inputs
    ------
        x0 - input array size(d) (domain)
        y0 - input array y0(x0) size(m,n,d) (values)
        m, n  - size of array
        d - number of points to use for interpolation
        x  - the location we want to inter/extrapolate at - scalar
    Outputs
    -------
        y  - the inter/extrapolated array at x, or y(x)
    """

    # 2 dimensional array interpolation
    y = zeros(DTYPE, m, n)

    # @simd for ii in 1:d
    @inbounds @fastmath begin
        for ii in 1:d
            L = 1.0
            for jj in 1:d
                if jj != ii
                    L *= (x - x0[jj]) / (x0[ii] - x0[jj])
                end
            end
            y += y0[:, :, ii] .* L
        end
    end

    return y
end

function lagrangeInterp(x0, y0, n, x)
    """
    Interpolate/extrapolate polynomial of order 'm'
    Providing 'n' points gives us inter/extrapolant of order m = n-1

    Inputs
    ------
        x0 - input vector
        y0 - input vector y0(x0)
        n  - size of array
        x  - the location we want to inter/extrapolate at
    Outputs
    -------
        y  - the inter/extrapolated value at x, or y(x)
    """
    y = 0.0

    for ii in 1:n # loop over points
        L = 1.0 # Lagrange weight
        for kk in 1:n
            if kk != ii
                # This is the lagrange polynomial
                L *= (x - x0[kk]) / (x0[ii] - x0[kk])
            end
        end
        y += y0[ii] * L
    end

    return y
end

function abs_smooth(x, Δx)
    """
    Absolute value function with quadratic in valley for C1 continuity
    """
    y = 0.0
    if (x >= Δx)
        y = x
    elseif (x <= -Δx)
        y = -x
    else
        y = x^2 / (2.0 * Δx) + Δx / 2.0
    end

    return y
end


# The following functions are based off of Andrew Ning's publicly available akima spline code
# Except the derivatives are generated implicitly using Zygote RAD
function setup_akima(npt, xpt, ypt, Δx)
    """
    Setup for the akima spline
    Returns spline coefficients
    """
    eps = 1e-30

    # --- Output ---
    p0 = zeros(npt - 1)
    p1 = zeros(npt - 1)
    p2 = zeros(npt - 1)
    p3 = zeros(npt - 1)

    # --- Local working vars ---
    t = zeros(npt)
    m = zeros(npt + 3) # segment slopes
    # There are two extra end points and beginning and end
    # x---x---o--....--o---x---x
    # estimate             estimate

    # Zygote buffers
    p0_z = Zygote.Buffer(p0)
    p1_z = Zygote.Buffer(p1)
    p2_z = Zygote.Buffer(p2)
    p3_z = Zygote.Buffer(p3)
    t_z = Zygote.Buffer(t)
    m_z = Zygote.Buffer(m)

    # --- Compute segment slopes ---
    for ii in 1:npt-1
        m_z[ii+2] = (ypt[ii+1] - ypt[ii]) / (xpt[ii+1] - xpt[ii])
    end
    # Estimations
    m_z[2] = 2.0 * m_z[3] - m_z[4]
    m_z[1] = 2.0 * m_z[2] - m_z[3]
    m_z[npt+2] = 2.0 * m_z[npt+1] - m_z[npt]
    m_z[npt+3] = 2.0 * m_z[npt+2] - m_z[npt+1]
    m = copy(m_z)

    # --- Slope at points ---
    for ii in 1:npt
        m1 = m[ii]
        m2 = m[ii+1]
        m3 = m[ii+2]
        m4 = m[ii+3]
        w1 = abs_smooth(m4 - m3, Δx)
        w2 = abs_smooth(m2 - m1, Δx)
        if (w1 < eps && w2 < eps)
            t_z[ii] = 0.5 * (m2 + m3)  # special case to avoid divide by zero
        else
            t_z[ii] = (w1 * m2 + w2 * m3) / (w1 + w2)
        end
    end
    t = copy(t_z)

    # --- Polynomial coefficients ---
    for ii in 1:npt-1
        dx = xpt[ii+1] - xpt[ii]
        t1 = t[ii]
        t2 = t[ii+1]
        p0_z[ii] = ypt[ii]
        p1_z[ii] = t1
        p2_z[ii] = (3.0 * m[ii+2] - 2.0 * t1 - t2) / dx
        p3_z[ii] = (t1 + t2 - 2.0 * m[ii+2]) / dx^2
    end

    return copy(p0_z), copy(p1_z), copy(p2_z), copy(p3_z)
end

function interp_akima(npt, n, x, xpt, p0, p1, p2, p3,
    dp0dxpt, dp1dxpt, dp2dxpt, dp3dxpt, dp0dypt, dp1dypt, dp2dypt, dp3dypt,
)
    """
    Evaluate Akima spline and its derivatives

    Returns
    y - interpolated value
    dydx - derivative of y wrt x
    dydxpt, dydypt - derivative of y wrt xpt and ypt
    """
    # --- Outputs ---
    y = zeros(n)
    dydx = zeros(n)
    dydxpt = zeros(n, npt)
    dydypt = zeros(n, npt)
    # Zygote buffers
    y_z = Zygote.Buffer(y)
    dydx_z = Zygote.Buffer(dydx)
    dydxpt_z = Zygote.Buffer(dydxpt)
    dydypt_z = Zygote.Buffer(dydypt)


    # --- Interpolate at each point ---
    for ii in 1:n

        # --- Find location of spline in array (uses end segments if out of bounds) ---
        jj = 1 # give jj an initial value
        if x[ii] < xpt[1]
            jj = 1
        else
            # Linear search
            for jj in npt-1:-1:1
                if x[ii] >= xpt[jj]
                    break
                end
            end
        end

        # --- Evaluate poly and derivative ---
        dx = (x[ii] - xpt[jj])
        y_z[ii] = p0[jj] + p1[jj] * dx + p2[jj] * dx^2 + p3[jj] * dx^3
        dydx_z[ii] = p1[jj] + 2.0 * p2[jj] * dx + 3.0 * p3[jj] * dx^2


        for kk in 1:npt
            dydxpt_z[ii, kk] = dp0dxpt[jj, kk] + dp1dxpt[jj, kk] * dx + dp2dxpt[jj, kk] * dx^2 + dp3dxpt[jj, kk] * dx^3
            if (kk == jj)
                dydxpt_z[ii, kk] = dydxpt[ii, kk] - dydx_z[ii]
            end
            dydypt_z[ii, kk] = dp0dypt[jj, kk] + dp1dypt[jj, kk] * dx + dp2dypt[jj, kk] * dx^2 + dp3dypt[jj, kk] * dx^3
        end
    end

    return copy(y_z), copy(dydx_z), copy(dydxpt_z), copy(dydypt_z)
end

function do_akima_interp(xpt, ypt, xq, Δx=1e-7)
    npt = length(xpt)
    n = length(xq)
    p0, p1, p2, p3 = setup_akima(npt, xpt, ypt, Δx)
    zeros_in = zeros(npt - 1, npt)
    y, _, _, _ = interp_akima(npt, n, xq, xpt, p0, p1, p2, p3, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in)

    if n == 1 # need it returned as a float
        return y[1]
    else
        return y
    end
end


end # module
