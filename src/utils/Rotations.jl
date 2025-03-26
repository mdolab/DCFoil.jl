# --- Julia 1.11---
"""
@File          :   Rotations.jl
@Date created  :   2025/01/26
@Last modified :   2025/02/05
@Author        :   Galen Ng
@Desc          :   Better rotations for the finite element beam solver and lifting line routines
"""


function get_rotate3dMat(ψ, axis="x")
    """
    Give rotation matrix about axis by 
    ψ radians (RH rule!)
    """
    rotMat::AbstractMatrix = zeros(DTYPE, 3, 3)

    cosψ = cos(ψ)
    sinψ = sin(ψ)
    n̂ = zeros(DTYPE, 3)

    if axis isa String
        if lowercase(axis) == "x"
            rotMat = [
                1 0 0
                0 cosψ -sinψ
                0 sinψ cosψ
            ]
            n̂ = [1.0, 0.0, 0.0]
        elseif lowercase(axis) == "y"
            rotMat = [
                cosψ 0 sinψ
                0 1 0
                -sinψ 0 cosψ
            ]
            n̂ = [0.0, 1.0, 0.0]
        elseif lowercase(axis) == "z"
            rotMat = [
                cosψ -sinψ 0
                sinψ cosψ 0
                0 0 1
            ]
            n̂ = [0.0, 0.0, 1.0]
        end
    else
        length(axis) == 3 || error("Axis must be string or vector of length 3")
        n̂ = axis / √(axis[XDIM]^2 + axis[YDIM]^2 + axis[ZDIM]^2) # normalize
    end

    # Rodrigues' rotation formula about arbitrary axis
    ψVec = ψ * n̂

    ψ̃ = [000 -ψVec[ZDIM] ψVec[YDIM]
        ψVec[ZDIM] 000 -ψVec[XDIM]
        -ψVec[YDIM] ψVec[XDIM] 000
    ]
    if abs_cs_safe(real(ψ)) < 1e-6
        rotMat = I(3) + ψ̃ + 0.5 * ψ̃^2 # Cartesian rotation vector O(ψ³) because we truncated it
    else
        rotMat = I(3) + sinψ / ψ * ψ̃ + (1 - cosψ) / ψ^2 * ψ̃^2 # Cayley-Hamilton's theorem
    end

    return rotMat
end

function get_transMat(dR1, dR2, dR3, l)
    """
    Find T such that
    [1; 0; 0] = T * [dR1; dR2; dR3]

    The reference vector vB = [1; 0; 0] is along the longitudinal axis of the beam.
    Based on Rodrigues' rotation formula (Palacios and Cesnik Appendix C)

    Inputs
    -------
        dR: vector along beam length
        l: length of element
        elemType: element type
    Outputs
    -------
        Γ: transformation matrix for the finite element
    """

    vA = [dR1, dR2, dR3] / l # beam vector
    vB = [1.0, 0.0, 0.0] # reference vector


    # nVec = cross(vA, vB) # normal vector to plane of rotation
    nVec = myCrossProd(vA, vB) # normal vector to plane of rotation
    n̂ = nVec / √(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # normalize
    cosψ = dot(vA, vB) # cosine of angle between vA and vB
    ψ = acos(cosψ) # angle between vA and vB
    sinψ = √(1 - cosψ^2) # sine of angle between vA and vB

    ψVec = ψ * n̂

    ψ̃ = [000 -ψVec[ZDIM] ψVec[YDIM]
        ψVec[ZDIM] 000 -ψVec[XDIM]
        -ψVec[YDIM] ψVec[XDIM] 000
    ]

    if abs_cs_safe(real(ψ)) < MEPSLARGE && real(nVec) == [0.0, 0.0, 0.0] # if no rotation

        T = I(3) # Identity matrix

    elseif abs_cs_safe(real(ψ)) < 1e-6
        T = I(3) + ψ̃ + 0.5 * ψ̃^2 # Cartesian rotation vector O(ψ³) because we truncated it for small angles
    else
        T = I(3) + sinψ / ψ * ψ̃ + (1 - cosψ) / ψ^2 * ψ̃^2 # Cayley-Hamilton's theorem
    end

    Z = zeros(DTYPE, 3, 3)

    Γ = [
        T Z Z Z Z Z
        Z T Z Z Z Z
        Z Z T Z Z Z
        Z Z Z T Z Z
        Z Z Z Z T Z
        Z Z Z Z Z T
    ]

    return Γ
end

function compute_cartAnglesFromVector(V)
    """
    Compute angles from a vector where V1 is streamwise
    For lifting line code
    """
    V1 = V[XDIM]
    V2 = V[YDIM]
    V3 = V[ZDIM]
    return atan_cs_safe(V3, V1), atan_cs_safe(V2, V1), √(V1^2 + V2^2 + V3^2)
end

function compute_vectorFromAngles(alpha, beta, Uinf)
    """
    Defines a flow vector from flow angles and magnitude .

    Parameters
    ----------
    alpha : scalar , optional
        Angle between the freestream and the x-y plane (rad ).
    beta : scalar , optional
        Angle between the freestream and the x-z plane (rad).
    Uinf : scalar , optional
        The magnitude of the freestream velocity .
    Returns
    -------
    V : array_like
        An array of size [3] containing the x, y, and z components of the
        freestream velocity .
    """

    cosa = cos(alpha)
    sina = sin(alpha)
    cosb = cos(beta)
    sinb = sin(beta)
    sinasinb = √(1.0 - sina^2 * sinb^2)

    # OLD WAY return Uinf * [cosa * cosb, sina * cosb, cosa * sinb] / sinasinb
    return Uinf * [cosa * cosb, cosa * sinb, sina * cosb] / sinasinb
end


# using .Rotations
# Rotations.get_transMat(1, 0, 0, 1)
# Rotations.get_transMat(1, 1, 0, √2)
# testVec = [1, 1, 1] / √3
# T = Rotations.get_transMat(testVec[1], testVec[2], testVec[3], √(testVec[1]^2 + testVec[2]^2 + testVec[3]^2))