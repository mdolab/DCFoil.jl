# --- Julia 1.9---
"""
@File    :   Interpolation.jl
@Time    :   2024/04/12
@Author  :   Galen Ng
@Desc    :   Interpolation schemes
"""

module Interpolation

using Zygote
using ChainRulesCore
using ..DCFoil: DTYPE

function lagrangeArrInterp(xj, yj, m::Int64, n::Int64, d::Int64, x)
    """
    Interpolate/extrapolate polynomials of order 'd-1'
    Providing 'd' points of array of size m x n, we obtain inter/extrapolant order 'd-1'
    Comes from Eirikur's DLM4PY code

    Inputs
    ------
        xj - input array size(d) (domain)
        yj - input array yj(xj) size(m,n,d) (values)
        m, n  - size of array
        d - number of points to use for interpolation
        x  - the location we want to inter/extrapolate at - scalar
    Outputs
    -------
        y  - the inter/extrapolated array at x, or y(x)
    """

    # ---------------------------
    #   Method 1
    # ---------------------------
    # # 2 dimensional array interpolation
    # y = zeros(m, n)
    # @inbounds @fastmath begin
    #     for ii in 1:d
    #         L = 1.0
    #         for jj in 1:d
    #             if jj != ii
    #                 L *= (x - xj[jj]) /
    #                      (xj[ii] - xj[jj])
    #             end
    #         end
    #         y += yj[:, :, ii] .* L # matrix version so we don't use Lagrange interp
    #     end
    # end

    # ---------------------------
    #   Method 2
    # ---------------------------
    # # Might need to zygote buffer this thing
    # y_z = Zygote.Buffer(y)
    # for ii in 1:m
    #     for jj in 1:n
    #         yvals = yj[ii, jj, :]
    #         # y[ii, jj] = lagrangeInterp(xj, yvals, d, x)
    #         y_z[ii, jj] = lagrangeInterp(xj, yvals, d, x)
    #     end
    # end
    # y = copy(y_z)

    # ---------------------------
    #   Method 3
    # ---------------------------
    yout = lagrangeArrInterp_differentiable(xj, yj, m, n, d, x)
    y = reshape(yout, m, n)

    return y
end

function lagrangeArrInterp_differentiable(xj, yj, m, n, d, x)
    """
    Unroll the output

    Outputs
    -------
        y  - the inter/extrapolated array at x, or y(x)...unrolled
    """

    # 2 dimensional array interpolation
    y = zeros(DTYPE, m, n)

    @inbounds @fastmath begin
        for ii in 1:d
            L = 1.0
            for jj in 1:d
                if jj != ii
                    L *= (x - xj[jj]) /
                         (xj[ii] - xj[jj])
                end
            end
            y += yj[:, :, ii] .* L # matrix version so we don't use Lagrange interp
        end
    end

    yout = vec(y)

    return yout
end

function lagrangeInterp(xj, yj, n::Int, x)
    """
    Interpolate/extrapolate polynomial of order 'm'
    Providing 'n' points gives us inter/extrapolant of order m = n-1

    Inputs
    ------
        xj - input vector
        yj - input vector of points yj(xj)
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
                L *= (x - xj[kk]) / (xj[ii] - xj[kk])
            end
        end
        y += yj[ii] * L
    end

    return y
end

function ChainRulesCore.rrule(::typeof(lagrangeInterp), xj, yj, n, x)
    """
    Derivative of lagrangeInterp
    """

    y = lagrangeInterp(xj, yj, n, x)

    function lagrangeInterp_pullback(ȳ)
        """
        Pullback for lagrangeInterp

        ȳ - the seed for the pullback
        """

        # 2 dimensional array interpolation
        # dydxj = zeros(size(xj))
        dydyj = zeros(size(yj))

        # y = 0.0
        dydx = 0.0
        for jj in 1:n # loop over points
            L = 1.0 # Lagrange weight

            dx = 0.0 # derivative of Lagrange weight wrt x

            for ii in 1:n
                if ii != jj
                    # This is the lagrange polynomial
                    L *= (x - xj[ii]) /
                         (xj[jj] - xj[ii])

                    dx += 1.0 / (x - xj[ii])
                end
            end

            dydx += yj[jj] * L * dx
            dydyj[jj] = L

        end

        # --- Vector-matrix products should happen here ---
        x0b = NoTangent() # this is incorrect but will work for now, could use Tapenade to inform...
        y0b = ȳ .* dydyj # Correct
        xb = ȳ .* dydx # Correct

        # println("back seed size", length(ȳ))

        return (NoTangent(), x0b, y0b, NoTangent(), xb)
    end


    return y, lagrangeInterp_pullback

end

# function ChainRulesCore.rrule(::typeof(lagrangeArrInterp_differentiable), xj, yj, m, n, d, x)
#     """
#     Derivative of lagrangeArrInterp
#     """

#     # --- Primal ---
#     y = lagrangeArrInterp_differentiable(xj, yj, m, n, d, x)

#     function lagrangeArrInterp_pullback(ȳ)
#         """
#         Pullback for lagrangeArrInterp

#         ȳ - the seed for the pullback, size(m, n)
#         """

#         dydxArr = zeros(m, n)
#         dydyjArr = zeros(m, n, d)

#         # Loop over all dimensions
#         for kk in 1:m
#             for ll in 1:n

#                 # --- Now do core lagrange derivative routine ---
#                 dydyj = zeros(d)
#                 dydx = 0.0
#                 for jj in 1:d # loop over points
#                     L = 1.0 # Lagrange weight

#                     dx = 0.0 # derivative of Lagrange weight wrt x

#                     for ii in 1:d
#                         if ii != jj
#                             # This is the lagrange polynomial
#                             L *= (x - xj[ii]) /
#                                  (xj[jj] - xj[ii])

#                             dx += 1.0 / (x - xj[ii])
#                         end
#                     end

#                     dydx += yj[kk, ll, jj] * L * dx
#                     dydyj[jj] = L
#                 end

#                 dydxArr[kk, ll] = dydx
#                 dydyjArr[kk, ll, :] = dydyj

#             end
#         end
#         # --- Reshape stuff ---
#         dydxVec = reshape(dydxArr, m * n)
#         dydyjVec = reshape(dydyjArr, m * n, d)

#         # --- Vector-matrix products should happen here ---
#         x0b = NoTangent() # this is incorrect but will work for now, could use Tapenade to inform...
#         y0b = ȳ .* dydyjVec
#         xb = ȳ .* dydxVec

#         # --- Reshaping again to fit input var shape ---
#         y0b = reshape(y0b, m, n, d)
#         xb = reshape(xb, m, n)

#         return (NoTangent(), x0b, y0b, NoTangent(), NoTangent(), NoTangent(), xb)
#     end

#     return y, lagrangeArrInterp_pullback

# end

# 2024/11/05 Kind of working but weird outputs
# The derivative of L(x) wrt xj[jj] is not correct
# # Testing lagrange derivatives
# using ChainRulesTestUtils
# test_rrule(lagrangeInterp, rand(3), rand(3), 3, 0.5)
# lagrangeInterp(0:0.1:1, 1:0.1:2, 3, 0.5)

# using AbstractDifferentiation: AbstractDifferentiation as AD
# using Zygote, FiniteDifferences, ForwardDiff, ReverseDiff

# backend = AD.FiniteDifferencesBackend()
# # AD.gradient(backend, x -> lagrangeInterp(0:0.2:1, 1:0.1:2, 3, x), 0.5)
# # # # # AD.gradient(backend, x -> lagrangeInterp(x, 1:0.1:2, 3, 0.5), collect(0:0.2:1.0))
# # # # dfdxfd = AD.gradient(backend, x -> lagrangeInterp(0:0.2:1.0, x, 3, 0.5), collect(1:0.1:2))
# yjtest = zeros(3, 3, 2)
# yjtest[:, :, 1] = [1 2 3; 2 3 4; 1 1 1]
# yjtest[:, :, 2] = [1 1 1; 5 3 4; 2 2 1]
# dfdxfd, = AD.jacobian(backend, x -> lagrangeArrInterp(0:0.2:1, yjtest, 3, 3, 2, x)[1], 0.5)
# # 2024/12/30 the array version works and agrees with the FD mode

# xj = 0:0.2:1
# yj = cat(repeat([1 2 3; 2 3 4; 1 1 1], 1, 1, length(xj) ÷ 2), repeat([1 1 1; 2 3 4; 2 2 1], 1, 1, length(xj) ÷ 2), dims=3)
# dfdxfd = AD.jacobian(backend, x -> lagrangeArrInterp(xj, yj, 3, 3, length(xj), x), 0.75)
# # dfdyjfd = AD.jacobian(backend, x -> lagrangeArrInterp(xj, x, 3, 3, length(xj), 0.5), yj)

# # backend = AD.ForwardDiffBackend()
# # # AD.gradient(backend, x -> lagrangeInterp(0:0.2:1, 1:0.1:2, 3, x), 0.75)
# # # dfdxjfwd = AD.gradient(backend, x -> lagrangeInterp(x, ones(5), 3, 0.5), collect(0:0.2:1.0))
# # # dfdxfwd = AD.gradient(backend, x -> lagrangeInterp(0:0.2:1.0, x, 3, 0.5), collect(1:0.1:2))


# backend = AD.ZygoteBackend()
# AD.jacobian(backend, x -> lagrangeInterp(0:0.2:1, 1:0.1:2, 3, x), 0.5)
# # # # dfdxjrad = AD.jacobian(backend, x -> lagrangeInterp(x, ones(5), 3, 0.5), collect(0:0.2:1.0))
# # dfdxrad = AD.jacobian(backend, x -> lagrangeInterp(0:0.2:1.0, x, 3, 0.5), collect(1:0.1:2))
# dfdxrad = AD.jacobian(backend, x -> lagrangeArrInterp(xj, yj, 3, 3, length(xj), x), 0.75) # SHape of the output doesn't fully work yet

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
