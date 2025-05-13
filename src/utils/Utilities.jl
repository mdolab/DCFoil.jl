"""
@File    :   Utilities.jl
@Time    :   2024/04/12
@Author  :   Galen Ng
@Desc    :   Some routines that are useful for the solver and do not depend on Anything
"""



using FLOWMath: abs_cs_safe
using LinearAlgebra
using Zygote
using ChainRulesCore


function unpack_dvdict(DVDict::AbstractDict)

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

function pack_funcsSens(funcsSens::AbstractDict, funcKey, dvKey, dfdx)
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

function unpack_coords(LECoords, TECoords)
    """
    Unpack the coordinates into a vector
    """
    allPts = cat(LECoords, TECoords, dims=2)
    m, n = size(allPts)
    LineCoordsVec = vec(allPts)

    return LineCoordsVec, m, n
end

function repack_coords(LineCoordsVec, m, n)
    """
    Repack the line coordinates into a matrix
    """
    LineCoords = reshape(LineCoordsVec, m, n)

    LECoords = LineCoords[:, 1:div(n, 2)]
    TECoords = LineCoords[:, div(n, 2)+1:end]
    return LECoords, TECoords
end

function unpack_appendageParams(appendageParams, appendageOptions)

    if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
        print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

        α₀ = appendageParams["alfa0"]
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]

        toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
    else
        rake = appendageParams["rake"]
        # span = appendageParams["s"] * 2
        toc::Vector{RealOrComplex} = appendageParams["toc"]
        ab::Vector{Real} = appendageParams["ab"]
        x_ab::Vector{Real} = appendageParams["x_ab"]
        zeta = appendageParams["zeta"]
        theta_f = appendageParams["theta_f"]
        beta = appendageParams["beta"]
        s_strut = appendageParams["s_strut"]
        c_strut = appendageParams["c_strut"]
        toc_strut = appendageParams["toc_strut"]
        ab_strut = appendageParams["ab_strut"]
        x_ab_strut = appendageParams["x_ab_strut"]
        theta_f_strut = appendageParams["theta_f_strut"]
    end

    return rake, toc, ab, x_ab, zeta, theta_f, beta, s_strut, c_strut, toc_strut, ab_strut, x_ab_strut, theta_f_strut
end


function set_defaultOptions!(solverOptions)
    """
    Set default options
    """

    function check_key!(solverOptions, key, default)
        if !haskey(solverOptions, key)
            println("Setting default option: $(key) to ", default)
            solverOptions[key] = default
        end
    end
    keys = [
        # ************************************************
        #     I/O
        # ************************************************
        "name",
        "outputDir",
        "debug",
        "writeTecplotSolution",
        "gridFile",
        # ************************************************
        #     Flow
        # ************************************************
        "Uinf",
        "rhof",
        "nu",
        "use_freeSurface",
        "use_cavitation",
        "use_ventilation",
        "use_dwCorrection",
        "use_nlll",
        # ************************************************
        #     Hull properties
        # ************************************************
        "hull",
        # ************************************************
        #     Solver modes
        # ************************************************
        "run_static",
        "res_jacobian",
        "onlyStructDerivs",
        "run_forced",
        "run_modal",
        "run_flutter",
        "run_body",
        "rhoKS",
        "maxQIter",
        "fRange",
        "tipForceMag",
        "nModes",
        "uRange",
    ]
    defaults = [
        # ************************************************
        #     I/O
        # ************************************************
        "default",
        "./OUTPUT/",
        false,
        false,
        nothing,
        # ************************************************
        #     Flow
        # ************************************************
        1.0,
        1000.0,
        1.1892E-06, # kinematic viscosity of seawater at 15C
        false,
        false,
        false,
        false,
        false, # use_nlll
        # ************************************************
        #     Hull properties
        # ************************************************
        nothing,
        # ************************************************
        #     Solver modes
        # ************************************************
        false,
        "analytic", # residual jacobian
        false,
        false,
        false,
        false,
        false, # run_body
        80.0,
        100, # maxQIter
        [0.1, 10.0], # fRange
        0.0, # tipForceMag
        10, # nModes
        [1.0, 2.0] # uRange
    ]
    for ii in eachindex(keys)
        check_key!(solverOptions, keys[ii], defaults[ii])
    end
end


function generate_naca4dig(toc)
    """
    Simple naca 
    """
    # --- Thickness distribution naca 4dig equation---
    C5 = 0.1015  # type I equation
    x = range(0, 1, length=50)

    # Thickness distribution (upper)
    yt = 5 * toc * (0.2969 * x .^ 0.5 - 0.126 * x - 0.3516 * x .^ 2 + 0.2843 * x .^ 3 - C5 * x .^ 4)
    lower_yt = -yt
    # Make CCW
    upper_yt = reverse(yt)
    y_coords = vcat(upper_yt, lower_yt)
    x_coords = vcat(reverse(x), x)
    foil_coords = hcat(x_coords, y_coords)
    return foil_coords
end

function compute_KS(g, ρKS)
    """
    Compute the KS function

    Inputs
    ------
    g - flutter constraints
    ρKS - KS parameter. Float

    Outputs
    -------
    gKS - KS function. Float
    """

    gmax = maximum(g)
    # gmax = maximum(abs_cs_safe.(g)) # DON'T DO THIS

    Σ = 0.0 # sum
    for gval in g
        Σ += exp(ρKS * (gval - gmax))
    end

    # --- Compute the KS function ---
    gKS = gmax + 1 / ρKS * log(Σ)

    return gKS
end # compute_KS

function normalize_3Dvector(r)
    rhat = r ./ √(r[XDIM]^2 + r[YDIM]^2 + r[ZDIM]^2)
    return rhat
end

function cross3D(arr1, arr2)
    """
    Cross product of two 3D arrays
    where the first dimension is length 3
    """
    @assert size(arr1, 1) == 3
    @assert size(arr2, 1) == 3
    M, N = size(arr1, 2), size(arr1, 3)

    arr1crossarr2 = zeros(Real, 3, M, N)
    # arr1crossarr2 = zeros(DTYPE, 3, M, N) # doesn't actually affect the result
    arr1crossarr2_z = Zygote.Buffer(arr1crossarr2)

    for jj in 1:M
        for kk in 1:N
            # arr1crossarr2_z[:, jj, kk] = cross(arr1[:, jj, kk], arr2[:, jj, kk])
            arr1crossarr2_z[:, jj, kk] = myCrossProd(arr1[:, jj, kk], arr2[:, jj, kk])
        end
    end
    arr1crossarr2 = copy(arr1crossarr2_z)

    return arr1crossarr2

end

# I can't believe I have to write my own matrix-matrix multiply that's AD safe
function my_matmul(A::AbstractMatrix, B::AbstractMatrix)
    """
    Matrix matrix multiplication
    """

    C = A * B
    return C
end

function ChainRulesCore.rrule(::typeof(my_matmul), A::AbstractMatrix, B::AbstractMatrix)
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

function ChainRulesCore.frule((_, ΔA, ΔB), ::typeof(my_matmul),
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    Ω = A * B
    ∂Ω = ΔA * B + A * ΔB
    return (Ω, ∂Ω)
end

function myCrossProd(vec1, vec2)
    v1crossv2 = vec([
        vec1[1] * vec2[3] - vec1[3] * vec2[2],
        vec1[3] * vec2[1] - vec1[1] * vec2[3],
        vec1[1] * vec2[2] - vec1[2] * vec2[1]
    ])
    return v1crossv2
end

function find_signChange(x)
    """
    Find the location where a sign changes in an array
    Inputs
    ------
        x - array which signchange is to be found. Size(n)
    Outputs
    -------
        locs - array of size 2 containing the location of the sign change
    """

    # Get signs of each element in x
    sgn = sign.(x)
    n = length(sgn)

    for ii in 1:n-1
        @fastmath @inbounds begin
            if sgn[ii+1] != sgn[ii]
                return ii, ii + 1
            else
                continue
            end
        end
    end

end

function compute_sigmoid(x, xtr, λ, k=20)
    """
    Compute the sigmoid function
    Inputs
    ------
        xtr - value at which to transition the sigmoid function
        λ - shift parameter
        x - value to evaluate the sigmoid function at
        k - steepness of the sigmoid function
    Outputs
    -------
        sig - sigmoid function value
    """

    sig = 1 / (1 + exp(2 * k * (xtr - x + λ)))

    return sig

end
