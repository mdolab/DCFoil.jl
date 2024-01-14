# --- Julia 1.7---
"""
@File    :   FiniteElements.jl
@Time    :   2022/08/04
@Author  :   Galen Ng
@Desc    :   Finite element library

# KNOWN BUGS:
    span derivative may result in array size being wrong
"""

# ==============================================================================
#                         METHODS FOR ELEMENT MATRICES
# ==============================================================================
module LinearBeamElem
"""
    ELEMENTS THAT WORK IN 3D SPACE
     - BEAM3D

    2-noded 2nd order linear beam element

        o---------o
        1  E,ρₛ,I  2

        {q} = [q₁, q₂, q₃, q₄, q₅, q₆, q₇, q₈]ᵀ

    2-noded 1st order linear beam spatial element with 12 DOF (BEAM3D)
                               z ^ y
                                 |/
            o------------o       +----> x (local coords)
            1  E,ρₛ,A,I  2

        {q} = [q₁, q₂, q₃, q₄, q₅, q₆, ...,q12 ]ᵀ

    The shape function for the element with the {q} vector) is
        w(x, t) = <something-already-derived-in-textbooks>

    You impose the geometric BC's to get the α coeffs since the function must be 'admissible'
        w(x,t) = [N(x)]{q(t)}

    This has already been derived so no need to redo getting [N] in the code

"""

function compute_elem_stiff(EIᵉ, EIIPᵉ, GJᵉ, BTᵉ, Sᵉ, EAᵉ, lᵉ, abᵉ, elemType="bend-twist", constitutive="isotropic", useTimoshenko=false)
    """
    Output
    ------
    Kᵉ: Stiffness matrix, size depends on element type

    Inputs
    ------
    EIᵉ : Float64
        out-of-plane (OOP) bending stiffness of the element [N m²]
    EIIPᵉ : Float64
        in-plane (IP) bending stiffness of the element [N m²]
    GJᵉ : Float64
        torsional stiffness of the element [N-m²]
    BTᵉ : Float64
        this is Kₛ from the paper (material bend-twist coupling, +ve for nose-down BTC) [N-m²]
    Sᵉ : Float64
        structural warping (cross-sections do not retain shape) [N-m⁴]
    lᵉ : Float64
        length of the element [m]
    abᵉ : Float64
        distance from midchord to EA (+ve if EA aft of midchord) [m]
    elemType : String
        which element stiffness matrix to use
    constitutive : String
        which constitutive model to use (isotropic or orthotropic)
    useTimoshenko : Bool
        whether to use Timoshenko beam theory (default is Euler-Bernoulli), only works for the bending element

    The internal strain energy of a beam is
        U = some-integral-function-derived-from-energy-principles = 0.5{q(t)}ᵀ[Kᵉ]{q(t)}

    Element stiffness matrix from the strain energies
    """

    # ************************************************
    #     Handy identities
    # ************************************************
    if useTimoshenko
        # NOTE: only used for the bending only elem. ATM
        κ = 5 / 6 # for a rect section according to wikipedia
        ϕ = 12 / lᵉ^2 * (EIᵉ / κ * GA) # don't know what GA is yet
    else
        ϕ = 0
    end
    kb = EIᵉ / (lᵉ^3 * (1 + ϕ))
    kt = GJᵉ / lᵉ
    coeff = 1 / lᵉ^3
    ax = EAᵉ * lᵉ^2
    az = 12 * EIᵉ
    bz = 6 * EIᵉ * lᵉ
    cz = 1200 * EIᵉ / 70
    dz = 600 * EIᵉ * lᵉ / 70
    ez = 4 * EIᵉ * lᵉ^2
    fz = 2 * EIᵉ * lᵉ^2
    gz = 192 * EIᵉ * lᵉ^2 / 35
    hz = 216 * EIᵉ * lᵉ^2 / 70
    iz = 30 * EIᵉ * lᵉ^2 / 70
    jz = 22 * EIᵉ * lᵉ^3 / 70
    kz = 8 * EIᵉ * lᵉ^3 / 70
    lz = 6 * EIᵉ * lᵉ^4 / 70
    mz = EIᵉ * lᵉ^4 / 70
    ay = 12 * EIIPᵉ
    by = 6 * EIIPᵉ * lᵉ
    cy = 1200 * EIIPᵉ / 70
    dy = 600 * EIIPᵉ * lᵉ / 70
    ey = 4 * EIIPᵉ * lᵉ^2
    fy = 2 * EIIPᵉ * lᵉ^2
    gy = 192 * EIIPᵉ * lᵉ^2 / 35
    hy = 216 * EIIPᵉ * lᵉ^2 / 70
    iy = 30 * EIIPᵉ * lᵉ^2 / 70
    jy = 22 * EIIPᵉ * lᵉ^3 / 70
    ky = 8 * EIIPᵉ * lᵉ^3 / 70
    ly = 6 * EIIPᵉ * lᵉ^4 / 70
    my = EIIPᵉ * lᵉ^4 / 70
    aτ = 0.2 * (6 * GJᵉ * lᵉ^2 + 60 * Sᵉ)
    bτ = 0.1 * (GJᵉ * lᵉ^3 + 60 * Sᵉ * lᵉ)
    cτ = (GJᵉ * lᵉ^4 - 60 * Sᵉ * lᵉ^2) / 30
    dτ = (2 * GJᵉ * lᵉ^4 + 60 * Sᵉ * lᵉ^2) / 15
    aθ = BTᵉ * lᵉ^2
    bθ = 0.2 * 6 * BTᵉ * lᵉ^2
    cθ = 0.05 * BTᵉ * lᵉ^4
    dθ = abᵉ * BTᵉ * lᵉ^3
    eθ = 0.2 * 3 * BTᵉ * lᵉ^3
    fθ = 0.2 * 2 * BTᵉ * lᵉ^3
    gθ = 0.1 * BTᵉ * lᵉ^3



    # --- Constitutive law ---
    if constitutive == "isotropic"
        BTᵉ = 0.0
    end

    # ************************************************
    #     Diff elem type stiffness matrices
    # ************************************************
    if elemType == "bend"
        k11 = 12
        k12 = 6 * lᵉ
        k13 = -12
        k14 = 6 * lᵉ
        k22 = (4 + ϕ) * lᵉ^2
        k23 = -6 * lᵉ
        k24 = (2 - ϕ) * lᵉ^2
        k33 = 12
        k34 = -6 * lᵉ
        k44 = (4 + ϕ) * lᵉ^2
        Kᵉ = kb * [
            k11 k12 k13 k14
            k12 k22 k23 k24
            k13 k23 k33 k34
            k14 k24 k34 k44
        ]

    elseif elemType == "bend-twist"
        # 6x6 elem stiffness matrix
        # Beam bending element terms
        k11 = kb * 12
        k12 = kb * 6 * lᵉ
        k14 = kb * -12
        k15 = kb * 6 * lᵉ
        k22 = kb * 4 * lᵉ^2
        k24 = kb * -6 * lᵉ
        k25 = kb * 2 * lᵉ^2
        k44 = kb * 12
        k45 = kb * -6 * lᵉ
        k55 = kb * 4 * lᵉ^2
        # Torsion element terms
        k33 = kt
        k36 = -kt
        k66 = kt
        if constitutive == "isotropic"
            # bend-twist coupling
            k23 = 0.0
            k26 = 0.0
            k35 = 0.0
            k56 = 0.0
        elseif constitutive == "orthotropic"
            kBT = BTᵉ / lᵉ
            k23 = -kBT
            k26 = kBT
            k35 = kBT
            k56 = -kBT
        end

        Kᵉ = [
            k11 k12 0.0 k14 k15 0.0 # w₁
            k12 k22 k23 k24 k25 k26 # ϕ₁
            0.0 k23 k33 0.0 k35 k36 # ψ₁
            k14 k24 0.0 k44 k45 0.0
            k15 k25 0.0 k45 k55 k56
            0.0 k26 k36 0.0 k56 k66
        ]

    elseif elemType == "bend-twist-axial"
        println("Axial elements not implemented")
    elseif elemType == "BT2" # Higher order beam element
        # row 1
        k11_11::Float64 = 12 * EIᵉ
        k11_12::Float64 = 6 * EIᵉ * lᵉ
        k11_13::Float64 = -12 * abᵉ * EIᵉ
        k11_14::Float64 = -(6 * abᵉ * EIᵉ + BTᵉ * lᵉ) * lᵉ
        # row 2
        k11_22::Float64 = 4 * EIᵉ * lᵉ^2
        k11_23::Float64 = -(6 * abᵉ * EIᵉ - BTᵉ * lᵉ) * lᵉ
        k11_24::Float64 = -0.5 * BTᵉ * lᵉ^3 - 4 * abᵉ * EIᵉ * lᵉ^2
        # row 3
        k11_33::Float64 = 6 * GJᵉ * lᵉ^2 / 5 + 12 * Sᵉ
        k11_34::Float64 = GJᵉ * lᵉ^3 * 0.1 + 6 * Sᵉ * lᵉ
        # row 4
        k11_44 = (abᵉ * BTᵉ * lᵉ^3) + (2 * GJᵉ * lᵉ^4 / 15) + (4 * Sᵉ * lᵉ^2)
        # --- Block matrices ---
        K11 = coeff * [
            k11_11 k11_12 k11_13 k11_14
            k11_12 k11_22 k11_23 k11_24
            k11_13 k11_23 k11_33 k11_34
            k11_14 k11_24 k11_34 k11_44
        ]
        k12_14 = -(6 * abᵉ * EIᵉ - BTᵉ * lᵉ) * lᵉ
        k12_24 = 0.5 * BTᵉ * lᵉ^3 - 2 * abᵉ * EIᵉ * lᵉ^2
        k12_44 = 2 * Sᵉ * lᵉ^2 - GJᵉ * lᵉ^4 / 30
        k12_32 = -(6 * abᵉ * EIᵉ + BTᵉ * lᵉ) * lᵉ
        k12_42 = -0.5 * BTᵉ * lᵉ^3 - 2 * abᵉ * EIᵉ * lᵉ^2
        K12 = coeff * [
            -k11_11 k11_12 -k11_13 k12_14
            -k11_12 0.5*k11_22 -k11_23 k12_24
            -k11_13 k12_32 -k11_33 k11_34
            -k12_32 k12_42 -k11_34 k12_44
        ]
        k22_24 = 0.5 * BTᵉ * lᵉ^3 - 4 * abᵉ * EIᵉ * lᵉ^2
        k22_44 = -(abᵉ * BTᵉ * lᵉ^3) + (2 * GJᵉ * lᵉ^4 / 15) + (4 * Sᵉ * lᵉ^2)
        K22 = coeff * [
            k11_11 -k11_12 k11_13 -k11_23
            -k11_12 k11_22 -k11_14 k22_24
            k11_13 -k11_14 k11_33 -k11_34
            -k11_23 k22_24 -k11_34 k22_44
        ]
        Ktop = hcat(K11, K12)
        Kbot = hcat(K12', K22)
        Kᵉ = vcat(Ktop, Kbot)
    elseif elemType == "BEAM3D" # 1st order 12 DOF spatial beam element
        if constitutive == "isotropic"
            # 12x12 elem stiffness matrix
            EIᵉOOP = EIᵉ
            EIᵉIP = EIIPᵉ * 1e2
            EAᵉ *= 1e2
            k11_11 = EAᵉ * lᵉ^2
            k11_22 = 12 * EIᵉIP
            k11_26 = 6 * EIᵉIP * lᵉ
            k11_33 = 12 * EIᵉOOP
            k11_35 = -6 * EIᵉOOP * lᵉ
            k11_44 = GJᵉ * lᵉ^2
            k11_55 = 4 * EIᵉOOP * lᵉ^2
            k11_66 = 4 * EIᵉIP * lᵉ^2
            K11 = coeff * [
                k11_11 000000 000000 000000 000000 000000
                000000 k11_22 000000 000000 000000 k11_26
                000000 000000 k11_33 000000 k11_35 000000
                000000 000000 000000 k11_44 000000 000000
                000000 000000 k11_35 000000 k11_55 000000
                000000 k11_26 000000 000000 000000 k11_66
            ]
            k12_11 = -k11_11
            k12_22 = -k11_22
            k12_26 = k11_26
            k12_33 = -k11_33
            k12_35 = k11_35
            k12_44 = -k11_44
            k12_55 = 2 * EIᵉOOP * lᵉ^2
            k12_66 = 2 * EIᵉIP * lᵉ^2
            K12 = coeff * [
                k12_11 000000 000000 000000 000000 000000
                000000 k12_22 000000 000000 000000 k12_26
                000000 000000 k12_33 000000 k12_35 000000
                000000 000000 000000 k12_44 000000 000000
                000000 000000 -k12_35 000000 k12_55 000000
                000000 -k12_26 000000 000000 000000 k12_66
            ]
            K22 = coeff * [
                k11_11 000000 000000 000000 000000 000000
                000000 k11_22 000000 000000 000000 -k11_26
                000000 000000 k11_33 000000 -k11_35 000000
                000000 000000 000000 k11_44 000000 000000
                000000 000000 -k11_35 000000 k11_55 000000
                000000 -k11_26 000000 000000 000000 k11_66
            ]
            Ktop = hcat(K11, K12)
            Kbot = hcat(K12', K22)
            Kᵉ = vcat(Ktop, Kbot)
        elseif constitutive == "orthotropic"
            println("Orthotropic not implemented")
        end
    elseif elemType == "COMP2" # Higher order composite beam 18 DOF using a 4th order basis function in bending
        aa = -abᵉ*az
        at = -(abᵉ * bz + aθ)
        bb = -bz*abᵉ+bθ
        ff =-(abᵉ * ez + fθ)
        dd = dθ+dτ
        K11 = coeff * [
            ax 00 00 00 00 00 00 00 00
            00 cy 00 00 00 dy 00 00 iy
            00 00 cz aa dz 00 at iz 00
            00 00 aa aτ bb 00 bτ gθ 00
            00 00 dz bb gz 00 ff jz 00
            00 dy 00 00 00 gy 00 00 jy
            00 00 at bτ ff 00 dd cθ 00
            00 00 iz gθ jz 00 cθ lz 00
            00 iy 00 00 00 jy 00 00 ly
        ]
        an =-abᵉ*bz+aθ
        af = -abᵉ*fz+eθ
        ae = -(abᵉ*fz+eθ)
        bn = bz*abᵉ+bθ
        K12 = coeff * [
            -ax 000 000 000 000 00 00 000 00
            000 -cy 000 000 000 dy 00 000 -iy
            000 000 -cz -aa  dz 00 an -iz 00
            000 000 -aa -aτ -bn 00 bτ  gθ 00
            000 000 -dz -bb  hz 00 af -kz 00
            000 -dy 000 000 000 hy 00 000 -ky
            000 000 -at -bτ  ae 00 -cτ cθ 00
            000 000 -iz -gθ  kz 00  cθ mz 00
            000 -iy 000 000 000 ky 000 00 my
        ]
        fn =-abᵉ*ez+fθ
        dn = -dθ+dτ
        K22 = coeff * [
            ax 00  00  00  00  00  00  00 00
            00 cy  00  00  00 -dy  00  00 iy
            00 00  cz  aa -dz  00 -an  iz 00
            00 00  aa  aτ  bn  00 -bτ -gθ 00
            00 00 -dz  bn  gz  00  fn -jz 00
            00 -dy 00  00  00  gy  00  00 -jy
            00 00 -an -bτ  fn  00  dn  cθ 00
            00 00  iz -gθ -jz  00  cθ  lz 00
            00 iy  00  00  00 -jy  00  00 ly
        ]
        Ktop = hcat(K11, K12)
        Kbot = hcat(K12', K22)
        Kᵉ = vcat(Ktop, Kbot)
    end

    return Kᵉ
end

function compute_elem_mass(mᵉ, iᵉ, lᵉ, x_αbᵉ, elemType="bend-twist")
    """
    Outputs
    -------
    Mᵉ : Array, Float64
        element mass matrix
    Inputs
    ------
    mᵉ : Float64
        mass per unit span of the element [kg / m]
    iᵉ : Float64
        mass moment of inertia about EA per unit span  [kg - m]
    lᵉ : Float64
        length of the element [m]
    x_αbᵉ : Float64
        static imbalance (distance from EA to CG, +ve CG aft of EA) [m]
    elemType : String
        which element mass matrix to use


    The kinetic energy is
        T = 0.5∫₀ᴸ m (∂w/∂t)² dx = 0.5{q̇(t)}ᵀ[Mᵉ]{q̇(t)}

    Element mass matrix from the kinetic energies
    """

    # ************************************************
    #     Handy identities
    # ************************************************
    mb = mᵉ * lᵉ / 420
    mt = iᵉ * lᵉ / 6
    ax = 2 * mᵉ * lᵉ / 6
    bx = mᵉ * lᵉ^2 / 6
    az = 181 * mᵉ * lᵉ / 462
    bz = 8 * mᵉ * lᵉ / 21
    cz = 5 * mᵉ * lᵉ / 42
    dz = 25 * mᵉ * lᵉ / 231
    ez = 29 * mᵉ * lᵉ^2 / 840
    fz = 11 * mᵉ * lᵉ^2 / 168
    gz = 5 * mᵉ * lᵉ^2 / 168
    hz = 3 * mᵉ * lᵉ^2 / 56
    iz = 311 * mᵉ * lᵉ^2 / 4620
    jz = 151 * mᵉ * lᵉ^2 / 4620
    kz = 19 * mᵉ * lᵉ^3 / 1980
    lz = 52 * mᵉ * lᵉ^3 / 3465
    mz = 23 * mᵉ * lᵉ^4 / 18480
    nz = 13 * mᵉ * lᵉ^4 / 13860
    oz = 17 * mᵉ * lᵉ^3 / 5040
    pz = 5 * mᵉ * lᵉ^3 / 1008
    qz = 281 * mᵉ * lᵉ^3 / 55440
    rz = 181 * mᵉ * lᵉ^3 / 55440
    sz = mᵉ * lᵉ^3 / 84
    tz = mᵉ * lᵉ^5 / 9240
    uz = mᵉ * lᵉ^4 / 1008
    vz = mᵉ * lᵉ^3 / 120
    wz = mᵉ * lᵉ^4 / 1260
    xz = mᵉ * lᵉ^5 / 11088
    ay = 181 * mᵉ * lᵉ / 462
    by = 8 * mᵉ * lᵉ / 21
    cy = 5 * mᵉ * lᵉ / 42
    dy = 25 * mᵉ * lᵉ / 231
    ey = 29 * mᵉ * lᵉ^2 / 840
    fy = 11 * mᵉ * lᵉ^2 / 168
    gy = 5 * mᵉ * lᵉ^2 / 168
    hy = 3 * mᵉ * lᵉ^2 / 56
    iy = 311 * mᵉ * lᵉ^2 / 4620
    jy = 151 * mᵉ * lᵉ^2 / 4620
    ky = 19 * mᵉ * lᵉ^3 / 1980
    ly = 52 * mᵉ * lᵉ^3 / 3465
    my = 23 * mᵉ * lᵉ^4 / 18480
    ny = 13 * mᵉ * lᵉ^4 / 13860
    oy = 17 * mᵉ * lᵉ^3 / 5040
    py = 5 * mᵉ * lᵉ^3 / 1008
    qy = 281 * mᵉ * lᵉ^3 / 55440
    ry = 181 * mᵉ * lᵉ^3 / 55440
    sy = mᵉ * lᵉ^3 / 84
    ty = mᵉ * lᵉ^5 / 9240
    uy = mᵉ * lᵉ^4 / 1008
    vy = mᵉ * lᵉ^3 / 120
    wy = mᵉ * lᵉ^4 / 1260
    xy = mᵉ * lᵉ^5 / 11088
    aτ = 156 * iᵉ * lᵉ / 420
    bτ = 54 * iᵉ * lᵉ / 420
    cτ = 22 * iᵉ * lᵉ^2 / 420
    dτ = 13 * iᵉ * lᵉ^2 / 420
    eτ = 4 * iᵉ * lᵉ^3 / 420
    fτ = 3 * iᵉ * lᵉ^3 / 420

    if elemType == "bend"
        m11 = mb * 156
        m12 = mb * 22 * lᵉ
        m13 = mb * 54
        m14 = mb * -13 * lᵉ
        m22 = mb * 4 * lᵉ^2
        m23 = mb * 13 * lᵉ
        m24 = mb * -3 * lᵉ^2
        m33 = mb * 156
        m34 = mb * -22 * lᵉ
        m44 = mb * 4 * lᵉ^2
        Mᵉ = [
            m11 m12 m13 m14
            m12 m22 m23 m24
            m13 m23 m33 m34
            m14 m24 m34 m44
        ]
    elseif elemType == "bend-twist"
        m11 = mb * 156
        m12 = mb * 22 * lᵉ
        m14 = mb * 54
        m15 = mb * -13 * lᵉ
        m22 = mb * 4 * lᵉ^2
        m24 = mb * 13 * lᵉ
        m25 = mb * -3 * lᵉ^2
        m44 = mb * 156
        m45 = mb * -22 * lᵉ
        m55 = mb * 4 * lᵉ^2
        m33 = mt * 2
        m36 = mt
        m66 = mt * 2
        Mᵉ = [
            m11 m12 0.0 m14 m15 0.0
            m12 m22 0.0 m24 m25 0.0
            0.0 0.0 m33 0.0 0.0 m36
            m14 m24 0.0 m44 m45 0.0
            m15 m25 0.0 m45 m55 0.0
            0.0 0.0 m36 0.0 0.0 m66
        ]
    elseif elemType == "BEAM3D"
        m11_11 = 140 * mᵉ * lᵉ / 420
        m11_22 = 156 * mᵉ * lᵉ / 420
        m11_26 = 22 * mᵉ * lᵉ^2 / 420
        m11_33 = m11_22
        m11_35 = m11_26
        m11_44 = 2 * iᵉ * lᵉ / 6
        m11_55 = 4 * mᵉ * lᵉ^3 / 420
        m11_66 = m11_55
        M11 = [
            m11_11 000000 000000 000000 000000 000000
            000000 m11_22 000000 000000 000000 m11_26
            000000 000000 m11_33 000000 m11_35 000000
            000000 000000 000000 m11_44 000000 000000
            000000 000000 m11_35 000000 m11_55 000000
            000000 m11_26 000000 000000 000000 m11_66
        ]
        m12_11 = 0.5 * m11_11
        m12_22 = 54 * mᵉ * lᵉ / 420
        m12_26 = -13 * mᵉ * lᵉ^2 / 420
        m12_33 = m12_22
        m12_35 = m12_26
        m12_44 = 0.5 * m11_44
        m12_55 = -3 * mᵉ * lᵉ^2 / 420
        m12_66 = m12_55
        M12 = [
            m12_11 000000 000000 000000 000000 000000
            000000 m12_22 000000 000000 000000 m12_26
            000000 000000 m12_33 000000 m12_35 000000
            000000 000000 000000 m12_44 000000 000000
            000000 000000 -m12_35 000000 m12_55 000000
            000000 -m12_26 000000 000000 000000 m12_66
        ]
        M22 = [
            m11_11 000000 000000 000000 000000 000000
            000000 m11_22 000000 000000 000000 -m11_26
            000000 000000 m11_33 000000 -m11_35 000000
            000000 000000 000000 m11_44 000000 000000
            000000 000000 -m11_35 000000 m11_55 000000
            000000 -m11_26 000000 000000 000000 m11_66
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    elseif elemType == "BT2"
        # row 1
        m11_11 = 13 * mᵉ * lᵉ / 35
        m11_12 = 11 * mᵉ * lᵉ^2 / 210
        m11_13 = 13 * mᵉ * x_αbᵉ * lᵉ / 35
        m11_14 = 11 * mᵉ * x_αbᵉ * lᵉ^2 / 210
        # row 2
        m11_22 = mᵉ * lᵉ^3 / 105
        m11_24 = mᵉ * x_αbᵉ * lᵉ^3 / 105
        # row 3
        m11_33 = 13 * lᵉ * iᵉ / 35
        m11_34 = 11 * lᵉ^2 * iᵉ / 210
        # row 4
        m11_44 = iᵉ * lᵉ^3 / 105
        # --- Block matrices ---
        M11 = [
            m11_11 m11_12 m11_13 m11_14
            m11_12 m11_22 m11_14 m11_24
            m11_13 m11_14 m11_33 m11_34
            m11_14 m11_24 m11_34 m11_44
        ]
        # row 1
        m12_11 = 9 * mᵉ * lᵉ / 70
        m12_12 = -13 * mᵉ * lᵉ^2 / 420
        m12_13 = 9 * mᵉ * x_αbᵉ * lᵉ / 70
        m12_14 = -13 * mᵉ * x_αbᵉ * lᵉ^2 / 420
        # row 2
        m12_22 = -mᵉ * lᵉ^3 / 140
        m12_23 = 13 * mᵉ * x_αbᵉ * lᵉ^2 / 420
        m12_24 = -mᵉ * x_αbᵉ * lᵉ^3 / 140
        # row 3
        m12_33 = 9 * lᵉ * iᵉ / 70
        m12_34 = -13 * lᵉ^2 * iᵉ / 420
        # row 4
        m12_44 = -iᵉ * lᵉ^3 / 140
        M12 = [
            m12_11 m12_12 m12_13 m12_14
            -m12_12 m12_22 m12_23 m12_24
            m12_13 -m12_23 m12_33 m12_34
            -m12_14 m12_24 -m12_34 m12_44
        ]
        M22 = [
            m11_11 -m11_12 m11_13 -m11_14
            -m11_12 m11_22 -m11_14 m11_24
            m11_13 -m11_14 m11_33 -m11_34
            -m11_14 m11_24 -m11_34 m11_44
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    elseif elemType == "BT3" # higher order composite beam 10 DOF
        az = 181 * mᵉ * lᵉ / 462
        bz = 8 * mᵉ * lᵉ / 21
        cz = 5 * mᵉ * lᵉ / 42
        dz = 25 * mᵉ * lᵉ / 231
        ez = 29 * mᵉ * lᵉ^2 / 840
        fz = 11 * mᵉ * lᵉ^2 / 168
        gz = 5 * mᵉ * lᵉ^2 / 168
        hz = 3 * mᵉ * lᵉ^2 / 56
        iz = 311 * mᵉ * lᵉ^2 / 4620
        jz = 151 * mᵉ * lᵉ^2 / 4620
        kz = 19 * mᵉ * lᵉ^3 / 1980
        lz = 52 * mᵉ * lᵉ^3 / 3465
        mz = 23 * mᵉ * lᵉ^4 / 18480
        nz = 13 * mᵉ * lᵉ^4 / 13860
        oz = 17 * mᵉ * lᵉ^3 / 5040
        pz = 5 * mᵉ * lᵉ^3 / 1008
        qz = 281 * mᵉ * lᵉ^3 / 55440
        rz = 181 * mᵉ * lᵉ^3 / 55440
        sz = mᵉ * lᵉ^3 / 84
        tz = mᵉ * lᵉ^5 / 9240
        uz = mᵉ * lᵉ^4 / 1008
        vz = mᵉ * lᵉ^3 / 120
        wz = mᵉ * lᵉ^4 / 1260
        xz = mᵉ * lᵉ^5 / 11088
        aτ = 156 * iᵉ * lᵉ / 420
        bτ = 54 * iᵉ * lᵉ / 420
        cτ = 22 * iᵉ * lᵉ^2 / 420
        dτ = 13 * iᵉ * lᵉ^2 / 420
        eτ = 4 * iᵉ * lᵉ^3 / 420
        fτ = 3 * iᵉ * lᵉ^3 / 420
        M11 = [
            az iz qz x_αbᵉ*bz x_αbᵉ*hz
            iz lz mz x_αbᵉ*fz x_αbᵉ*sz
            qz mz tz x_αbᵉ*pz x_αbᵉ*uz
            x_αbᵉ*bz x_αbᵉ*fz x_αbᵉ*pz aτ cτ
            x_αbᵉ*hz x_αbᵉ*sz x_αbᵉ*uz cτ eτ
        ]
        M12 = [
            dz -jz rz x_αbᵉ*cz -x_αbᵉ*gz
            jz -kz nz x_αbᵉ*ez -x_αbᵉ*vz
            rz -nz xz x_αbᵉ*oz -x_αbᵉ*wz
            x_αbᵉ*cz -x_αbᵉ*ez x_αbᵉ*oz bτ -dτ
            x_αbᵉ*gz -x_αbᵉ*vz x_αbᵉ*wz dτ -fτ
        ]
        M22 = [
            az -iz qz x_αbᵉ*bz -x_αbᵉ*hz
            -iz lz -mz -x_αbᵉ*fz x_αbᵉ*sz
            qz -mz tz x_αbᵉ*pz -x_αbᵉ*uz
            x_αbᵉ*bz -x_αbᵉ*fz x_αbᵉ*pz aτ -cτ
            -x_αbᵉ*hz x_αbᵉ*sz -x_αbᵉ*uz -cτ eτ
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    elseif elemType == "COMP2"
        xb = x_αbᵉ*bz
        xf = x_αbᵉ*fz
        xh = x_αbᵉ*hz
        xc = x_αbᵉ*cz
        xp = x_αbᵉ*pz
        xs = x_αbᵉ*sz
        xu = x_αbᵉ*uz
        M11 = [
            ax 00 00 00 00 00 00 00 00
            00 ay 00 00 00 iy 00 00 qy
            00 00 az xb iz 00 xh qz 00
            00 00 xb aτ xf 00 cτ xp 00
            00 00 iz xf lz 00 xs mz 00
            00 iy 00 00 00 ly 00 00 my
            00 00 xh cτ xs 00 eτ xu 00
            00 00 qz xp mz 00 xu tz 00
            00 qy 00 00 00 my 00 00 ty
        ]
        xg = -x_αbᵉ*gz
        xe = -x_αbᵉ*ez
        xo = x_αbᵉ*oz
        xv = -x_αbᵉ*vz
        xw = -x_αbᵉ*wz
        M12 = [
            bx 00  00  00  00  00  00  00 00
            00 dy  00  00  00 -jy  00  00 ry
            00 00  dz  xc -jz  00  xg  rz 00
            00 00  xc  bτ  xe  00 -dτ  xo 00
            00 00  jz -xe -kz  00  xv  nz 00
            00 jy  00  00  00 -ky  00  00 ny
            00 00 -xg  dτ  xv  00 -fτ -xw 00
            00 00  rz  xo -nz  00  xw  xz 00
            00 ry  00  00  00 -ny  00  00 xy
        ]
        M22 = [
            ax  00  00  00  00  00  00  00  00
            00  ay  00  00  00 -iy  00  00  qy
            00  00  az  xb -iz  00 -xh  qz  00
            00  00  xb  aτ -xf  00 -cτ  xp  00
            00  00 -iz -xf  lz  00  xs -mz  00
            00 -iy  00  00  00  ly  00  00 -my
            00  00 -xh -cτ  xs  00  eτ -xu  00
            00  00  qz  xp -mz  00 -xu  tz  00
            00  qy  00  00  00 -my  00  00  ty
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    end

    return Mᵉ
end

end # end module

# ==============================================================================
#                         GENERIC FEM METHODS
# ==============================================================================
module FEMMethods
"""
Module with generic FEM methods
"""

# --- Libraries ---
using Zygote, ChainRulesCore
using DelimitedFiles
using LinearAlgebra
using ..LinearBeamElem
include("../solvers/SolverRoutines.jl")
include("../constants/SolutionConstants.jl")
using .SolverRoutines
using .SolutionConstants

# --- Globals ---
global XDIM = 1
global YDIM = 2
global ZDIM = 3

function make_mesh(nElem::Int64, span::Float64; config="wing", rotation=0.000, nElStrut=0, spanStrut=0.0)
    """
    Makes a mesh and element connectivity
    First element is always origin (x,y,z) = (0,0,0)
    You do not necessarily have to make this mesh yourself every run
    The default is to mesh y as the span

    Inputs
    ------
    nElem:
        number of elements
    config:
        "wing" or "t-foil"
    rotation:
        rotation of the foil in degrees where 0.0 is lifting up in 'z' and y is the spanwise direction
    Outputs
    -------
    mesh
        (nNodes, nDim) array
    elemConn
        (nElem, nNodesPerElem) array saying which elements hold which nodes
    """
    mesh = Array{Float64}(undef, nElem + 1, 3)
    elemConn = Array{Int64}(undef, nElem, 2)
    mesh_z = Zygote.Buffer(mesh)
    elemConn_z = Zygote.Buffer(elemConn)
    rot = deg2rad(rotation)
    if config == "wing"
        # Set up a line mesh
        dl = span / (nElem) # dist btwn nodes
        # println("span",span)
        # println("dl",dl)
        mesh_z[:, YDIM] = collect((0.0:dl:span))
        for nodeIdx in 1:nElem+1 # loop nodes and rotate
            mesh_z[nodeIdx, :] = rotate3d(mesh_z[nodeIdx, :], rot; axis="x")
        end
        for ee in 1:nElem
            elemConn_z[ee, 1] = ee
            elemConn_z[ee, 2] = ee + 1
        end
    elseif config == "t-foil"
        mesh = Array{Float64}(undef, nElem + nElStrut + 1, 3)
        elemConn = Array{Int64}(undef, nElem + nElStrut, 2)
        # Simple meshes starting from junction at zero
        # Mesh foil wing
        dl = span / (nElem) # dist btwn nodes
        foilwingMesh = collect(0:dl:span)
        # Mesh strut
        dlStrut = spanStrut / (nElStrut - 1)
        strutMesh = collect(dlStrut:dlStrut:spanStrut) # don't start at zero since it already exists
        # This is basically avoiding double counting the nodes
        if abs(rot) < SolutionConstants.mepsLarge # no rotation, just a straight wing
            println("Default rotation of zero")
            nodeCtr = 1
            # Add foil wing first
            for nodeIdx in 1:nElem+1
                mesh[nodeCtr, :] = [0.0, foilwingMesh[nodeIdx], 0.0]
                elemConn[nodeCtr, 1] = nodeCtr
                elemConn[nodeCtr, 2] = nodeCtr + 1
                nodeCtr += 1
            end
            for nodeIdx in 1:nElStrut # loop elem, not nodes
                if nodeIdx <= nElStrut - 1
                    mesh[nodeCtr, 1:3] = [0.0, 0.0, strutMesh[nodeIdx]]
                    elemConn[nodeCtr, 1] = nodeCtr
                    elemConn[nodeCtr, 2] = nodeCtr + 1
                end
                nodeCtr += 1
            end
        else

        end

        return mesh, elemConn

    end

    return copy(mesh_z), copy(elemConn_z)

end

function rotate3d(dataVec, rot; axis="x")
    """
    Rotates a 3D vector about axis by rot radians (RH rule!)
    """
    rotMat = Array{Float64}(undef, 3, 3)
    c = cos(rot)
    s = sin(rot)
    if axis == "x"
        rotMat = [
            1 0 0
            0 c -s
            0 s c
        ]
    elseif axis == "y"
        rotMat = [
            c 0 s
            0 1 0
            -s 0 c
        ]
    elseif axis == "z"
        rotMat = [
            c -s 0
            s c 0
            0 0 1
        ]
    else
        println("Only axis rotation implemented")
    end
    transformedVec = rotMat * dataVec
    return transformedVec
end

function assemble(coordMat, elemConn, abVec, x_αbVec, FOIL, elemType="bend-twist", constitutive="isotropic")
    """
    Generic function to assemble the global mass and stiffness matrices

    Inputs
    ------
        coordMat: 2D array of coordinates of nodes
        elemConn: 2D array of element connectivity (nElem x 2)
    Outputs
    -------
        globalK: global stiffness matrix
        globalM: global mass matrix
        globalF: global force vector
    """

    # --- Local nodal DOF vector ---
    # Determine the number of dofs per node
    if elemType == "bend"
        nnd = 2
    elseif elemType == "bend-twist"
        nnd = 3
    elseif elemType == "BT2"
        nnd = 4
    elseif elemType == "BT3"
        nnd = 5
    elseif elemType == "BEAM3D"
        nnd = 6
    elseif elemType == "COMP2"
        nnd = 9
    else
        error(elemType, " element type not implemented")
    end
    qLocal = zeros(nnd * 2)

    # --- Initialize matrices ---
    nElem::Int64 = size(elemConn)[1]
    nNodes = nElem + 1
    ndim = ndims(coordMat[1, :])
    globalK::Matrix{Float64} = zeros(nnd * (nNodes), nnd * (nNodes))
    globalM::Matrix{Float64} = zeros(nnd * (nNodes), nnd * (nNodes))
    globalF::Vector{Float64} = zeros(nnd * (nNodes))


    # --- Debug printout for initialization ---
    ChainRulesCore.ignore_derivatives() do
        println("+------------------------------+")
        println("|   Assembling beam matrices   |")
        println("+------------------------------+")
        # println("Default 2 nodes per elem, nothing else will work")
        println("Constitutive relations: ", constitutive)
        println("Element: ", elemType)
        println("No. of elems: ",nElem)
        println("Beam nodes: ", nNodes," (", nnd * nNodes, " DOFs)")
    end

    # ************************************************
    #     Element loop
    # ************************************************
    # --- Zygote buffer initializations ---
    globalK_z = Zygote.Buffer(globalK)
    globalM_z = Zygote.Buffer(globalM)
    globalF_z = Zygote.Buffer(globalF)
    globalK_z[:,:] = globalK
    globalM_z[:,:] = globalM
    globalF_z[:] = globalF
    for elemIdx ∈ 1:nElem
        # ---------------------------
        #   Extract element info
        # ---------------------------
        dR::Vector{Float64} = (coordMat[elemIdx+1, :] - coordMat[elemIdx, :])
        lᵉ::Float64 = norm(dR, 2) # length of elem
        nVec = dR / lᵉ # normalize
        EIₛ::Float64 = FOIL.EIₛ[elemIdx]
        EIIPₛ::Float64 = FOIL.EIIPₛ[elemIdx]
        EAₛ::Float64 = FOIL.EAₛ[elemIdx]
        GJₛ::Float64 = FOIL.GJₛ[elemIdx]
        Kₛ::Float64 = FOIL.Kₛ[elemIdx]
        Sₛ::Float64 = FOIL.Sₛ[elemIdx]
        mₛ::Float64 = FOIL.mₛ[elemIdx]
        iₛ::Float64 = FOIL.Iₛ[elemIdx]
        # These are currently DVs
        ab::Float64 = abVec[elemIdx]
        x_αb::Float64 = x_αbVec[elemIdx]

        # ---------------------------
        #   Local stiffness matrix
        # ---------------------------
        kLocal::Matrix{Float64} = LinearBeamElem.compute_elem_stiff(EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, lᵉ, ab, elemType, constitutive, false)

        # ---------------------------
        #   Local mass matrix
        # ---------------------------
        mLocal::Matrix{Float64} = LinearBeamElem.compute_elem_mass(mₛ, iₛ, lᵉ, x_αb, elemType)

        # ---------------------------
        #   Local force vector
        # ---------------------------
        fLocal::Vector{Float64} = zeros(nnd * 2)

        # ---------------------------
        #   Transform from local to global coordinates
        # ---------------------------
        #  AEROSP510 notes and python code, Engineering Vibration Chapter 8 (Inman 2014)
        # The local coordinate system is {u} while the global is {U}
        # {u} = [Γ] * {U}
        # where [Γ] is the transformation matrix
        Γ = SolverRoutines.get_transMat(dR, lᵉ, elemType)
        kElem = Γ' * kLocal * Γ
        mElem = Γ' * mLocal * Γ
        fElem = Γ' * fLocal
        # kElem = kLocal
        # mElem = mLocal
        # fElem = fLocal
        # println("T:")
        # show(stdout, "text/plain", Γ[1:3, 1:3])
        # println()
        # println("kLocal: ")
        # show(stdout, "text/plain", kLocal)
        # println()
        # println("mLocal: ")
        # show(stdout, "text/plain", mLocal)
        # println()
        # println("kElem: ")
        # show(stdout, "text/plain", kElem)
        # println()
        # println("mElem: ")
        # show(stdout, "text/plain", mElem)
        # println()
        # writedlm("DebugKLocal.csv", kLocal, ',')
        # writedlm("DebugMLocal.csv", mLocal, ',')
        # writedlm("DebugKElem.csv", kElem, ',')
        # writedlm("DebugMElem.csv", mElem, ',')

        # ---------------------------
        #   Assemble into global matrices
        # ---------------------------
        # The following procedure generally follows:
        for nodeIdx ∈ 1:2 # loop over nodes in element
            for dofIdx ∈ 1:nnd # loop over DOFs in node
                idxRow = ((elemConn[elemIdx, nodeIdx] - 1) * nnd + dofIdx) # idx of global dof (row of global matrix)
                idxRowₑ = (nodeIdx - 1) * nnd + dofIdx # idx of dof within this element

                # --- Assemble RHS ---
                globalF_z[idxRow] = fElem[idxRowₑ]

                # --- Assemble LHS ---
                for nodeColIdx ∈ 1:2 # loop over nodes in element
                    for dofColIdx ∈ 1:nnd # loop over DOFs in node
                        idxCol = (elemConn[elemIdx, nodeColIdx] - 1) * nnd + dofColIdx # idx of global dof (col of global matrix)
                        idxColₑ = (nodeColIdx - 1) * nnd + dofColIdx # idx of dof within this element (column)

                        globalK_z[idxRow, idxCol] += kElem[idxRowₑ, idxColₑ]
                        globalM_z[idxRow, idxCol] += mElem[idxRowₑ, idxColₑ]
                    end
                end
            end
        end
    end
    globalK = copy(globalK_z)
    globalM = copy(globalM_z)
    globalF = copy(globalF_z)

    return globalK, globalM, globalF
end

function get_fixed_nodes(elemType::String, BCCond="clamped")
    """
    Depending on the elemType, return the indices of fixed nodes
    """
    if BCCond == "clamped"
        if elemType == "bend"
            fixedNodes = [1, 2]
        elseif elemType == "bend-twist"
            fixedNodes = [1, 2, 3]
        elseif elemType == "BT2"
            fixedNodes = [1, 2, 3, 4]
        elseif elemType == "BT3"
            fixedNodes = [1, 2, 3, 4, 5]
        elseif elemType == "BEAM3D"
            fixedNodes = [1, 2, 3, 4, 5, 6]
        elseif elemType == "COMP2"
            fixedNodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else
            error("elemType not recognized")

        end

    else
        error("BCCond not recognized")
    end

    ChainRulesCore.ignore_derivatives() do
        println("BCType: ", BCCond)
    end

    return fixedNodes
end

function apply_tip_load!(globalF, elemType, transMat, loadType="force")
    """
    Routine for applying unit tip load to the end node
        globalF
            vector
        elemType: str
        transMat: 2d array
            Transformation matrix from local into global coordinates
    """

    MAG = 1.0
    m, n = size(transMat)

    if loadType == "force"
        if elemType == "bend"
            FLocalVec = [1.0, 0.0]
        elseif elemType == "bend-twist"
            FLocalVec = [1.0, 0.0, 0.0]
        elseif elemType == "BT2"
            FLocalVec = [1.0, 0.0, 0.0, 0.0]
        elseif elemType == "BEAM3D"
            FLocalVec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        elseif elemType == "COMP2"
            FLocalVec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            error("element not defined")
        end
    elseif loadType == "torque"
        if elemType == "bend-twist"
            FLocalVec = [0.0, 0.0, 1.0]
        elseif elemType == "BT2"
            FLocalVec = [0.0, 0.0, 1.0, 0.0]
        elseif elemType == "BEAM3D"
            FLocalVec = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elseif elemType == "COMP2"
            FLocalVec = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else
            error("element not defined")
        end
    end

    # --- Transform to global then add into vector ---
    if elemType == "COMP2"
        FLocalVec = transMat[1:m÷2, 1:n÷2]' * FLocalVec
    end
    globalF[end-length(FLocalVec)+1:end] += FLocalVec * MAG

end

function apply_tip_mass(globalM, mass, inertia, elemLength, x_αbBulb, transMat, elemType="BT2")
    """
    Apply a tip mass to the global mass matrix

    mass: mass of the tip [kg]
    inertia: moment of inertia of the tip about C.G. [kg-m^2]
    """

    globalM_z = Zygote.Buffer(globalM)
    globalM_z[:, :] = globalM
    # TODO: unit test for the tip mass, verification case from Eirikur's thesis?
    if elemType == "bend-twist"
        println("Does not work")
    elseif elemType == "BT2"
        nDOF = 8
        # --- Get sectional properties ---
        ms = mass / elemLength
        # Parallel axis theorem
        Iea = inertia + mass * (x_αbBulb)^2
        is = Iea / elemLength
        tipMassMat = LinearBeamElem.compute_elem_mass(ms, is, elemLength, x_αbBulb, elemType)
        tipMassMat = transMat' * tipMassMat * transMat

        # --- Assemble into global matrix ---
        globalM_z[end-nDOF+1:end, end-nDOF+1:end] += tipMassMat
    elseif elemType == "COMP2"
        nDOF = 18
        # --- Get sectional properties ---
        ms = mass / elemLength
        # Parallel axis theorem
        Iea = inertia + mass * (x_αbBulb)^2
        is = Iea / elemLength
        tipMassMat = LinearBeamElem.compute_elem_mass(ms, is, elemLength, x_αbBulb, elemType)
        tipMassMat = transMat' * tipMassMat * transMat

        # --- Assemble into global matrix ---
        globalM_z[end-nDOF+1:end, end-nDOF+1:end] += tipMassMat
    else
        error("Not implemented")
    end

    ChainRulesCore.ignore_derivatives() do
        println("+------------------------------------+")
        println("|    Tip mass added!                 |")
        println("+------------------------------------+")
        println("Dist. CG is aft of EA: ", x_αbBulb, " [m]")
    end
    # NOTE: After adding the tip mass, the matrix is no longer positive definite... bug?
    # println("after tip mass:",eigvals(copy(globalM_z)))
    return copy(globalM_z)
end

function apply_inertialLoad!(globalF; gravityVector=[0.0, 0.0, -9.81])
    """
    Applies inertial load and modifies globalF
    """

    println("Adding inertial loads to FEM with gravity vector of", gravityVector)

    # TODO: add gravity vector
end

function apply_BCs(K, M, F, globalDOFBlankingList)
    """
    Applies BCs for nodal displacements and blanks them
    """

    # newK = K[
    #     setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    # ]
    # newM = M[
    #     setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    # ]
    # newF = F[setdiff(1:end, (globalDOFBlankingList))]

    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newF = F[1:end.∉[globalDOFBlankingList]]

    return newK, newM, newF
end

function put_BC_back(q, elemType::String, BCType="clamped")
    """
    appends the BCs back into the solution
    """

    if BCType == "clamped"
        if elemType == "BT2"
            uSol = vcat([0, 0, 0, 0], q)
        elseif elemType == "COMP2"
            uSol = vcat(zeros(9), q)
        else
            println("Not working")
            exit()
        end
    else
        println("Not working")
    end

    return uSol, length(uSol)
end

function solve_structure(K, M, F)
    """
    Solve the structural system
    """

    q = (K) \ F # TODO: should probably replace this with an iterative solver

    return q
end

function compute_modal(K, M, nEig::Int64)
    """
    Compute the eigenvalues (natural frequencies) and eigenvectors (mode shapes) of the in-vacuum system.
    i.e., this is structural dynamics, not hydroelastics.

    The eigenvalues should be all positive and real
    """

    # use krylov method to get first few smallest eigenvalues
    # Solve [K]{x} = λ[M]{x} where λ = ω²
    eVals, eVecs = SolverRoutines.compute_eigsolve(K, M, nEig)

    naturalFreqs = sqrt.(eVals) / (2π)

    return naturalFreqs, eVecs
end

function compute_proportional_damping(K, M, ζ, nMode)
    """
    TODO: MAKE SURE THIS CONSTANT STAYS THE SAME THROUGHOUT OPTIMIZATION
    Compute the proportional (Rayleigh) damping matrix

    C = alpha [M] + beta [K]

    where alpha and beta are the mass- and stiffness-
    proportional damping coefficients, respectively.

    
    Inputs
    ------
        K: stiffness matrix after BCs applied
        M: mass matrix after BCs applied
        Zeta: damping ratio at the first min and max natural frequencies [-]
        OR
        damping ratio at the first natural frequency (vacuum)
    """

    # --- Compute undamped natural frequencies ---
    fns, _ = compute_modal(K, M, nMode)
    ω₁ = fns[1] * 2π
    ω₂ = fns[nMode] * 2π

    # --- Compute proportional coefficients ---
    # # Mass and stiffness proportional damping
    # All damping in between the first two natural frequencies is not overdamped artificially
    # massPropConst = 2 * ζ / (ω₁ + ω₂)
    # stiffPropConst = ω₁ * ω₂ * massPropConst

    # Stiffness proportional damping (grows with increasing response freq)
    massPropConst = 0.0
    stiffPropConst = 2 * ζ / ω₂

    # # Mass proportional damping
    # massPropConst = 2 * ζ * ω₁
    # stiffPropConst = 0.0

    return massPropConst, stiffPropConst
end

end # end module

# module BrickElem
# function compute_shapeFuncs(coordMat, ξ, η, ζ, order=1)
#     # --- Lagrange poly shape funcs ---
#     Nᵢ = 0.125 * [
#         (1 - ξ) * (1 - η) * (1 - ζ) # node 1 (-1, -1, -1)
#         (1 + ξ) * (1 - η) * (1 - ζ) # node 2 (1, -1, -1)
#         (1 + ξ) * (1 + η) * (1 - ζ) # node 3 (1, 1, -1)
#         (1 - ξ) * (1 + η) * (1 - ζ) # node 4 (-1, 1, -1)
#         (1 - ξ) * (1 - η) * (1 + ζ) # node 5 (-1, -1, 1)
#         (1 + ξ) * (1 - η) * (1 + ζ) # node 6 (1,-1, 1)
#         (1 + ξ) * (1 + η) * (1 + ζ) # node 7 (1, 1, 1)
#         (1 - ξ) * (1 - η) * (1 + ζ) # node 8 (-1, 1, 1)
#     ]
#     return nothing

# end

# end # end module