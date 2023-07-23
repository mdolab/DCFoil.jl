# --- Julia 1.7---
"""
@File    :   FiniteElements.jl
@Time    :   2022/08/04
@Author  :   Galen Ng
@Desc    :   Finite element library
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

function compute_elem_stiff(EIᵉ, EIIPᵉ, GJᵉ, BTᵉ, Sᵉ, EAᵉ, lᵉ, abᵉ, elemType="bend-twist", constitutive="isotropic", useTimoshenko=false, dim=1)
    """
    The internal strain energy of a beam is
        U = some-integral-function-derived-from-energy-principles = 0.5{q(t)}ᵀ[Kᵉ]{q(t)}

    Element stiffness matrix from the strain energies
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
        this is Kₛ from the paper (material bend-twist coupling) [N-m²]
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
    atau = 0.2 * (6 * GJᵉ * lᵉ^2 + 60 * Sᵉ)
    btau = 0.1 * (GJᵉ * lᵉ^3 + 60 * Sᵉ * lᵉ)
    ctau = (GJᵉ * lᵉ^4 - 60 * Sᵉ * lᵉ^2) / 30
    dtau = (2 * GJᵉ * lᵉ^4 + 60 * Sᵉ * lᵉ^2) / 15
    atheta = BTᵉ * lᵉ^2
    btheta = 0.2 * 6 * BTᵉ * lᵉ^2
    ctheta = 0.05 * BTᵉ * lᵉ^4
    dtheta = abᵉ * BTᵉ * lᵉ^3
    etheta = 0.2 * 3 * BTᵉ * lᵉ^3
    ftheta = 0.2 * 2 * BTᵉ * lᵉ^3
    gtheta = 0.1 * BTᵉ * lᵉ^3


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
        if dim == 3
            # NOTE: This test failed and gives a singular matrix. You have to be more careful
            inf = 1e4 * k11
            Kᵉ = kb * [
                # u  v  w   θx  θy  θz  u   v   w   θx  θy  θz
                inf 000 000 000 000 000 -inf 00 000 000 000 000
                000 inf 000 000 000 inf 000 -inf 00 000 000 inf
                000 000 k11 000 k12 000 000 000 k13 000 k14 000
                000 000 000 inf 000 000 000 000 000 -inf 00 000
                000 000 k12 000 k22 000 000 000 k23 000 k24 000
                000 inf 000 000 000 inf 000 -inf 00 000 000 inf
                -inf 00 000 000 000 000 inf 000 000 000 000 000
                000 -inf 00 000 000 -inf 00 inf 000 000 000 -inf
                000 000 k13 000 k23 000 000 000 k33 000 k34 000
                000 000 000 -inf 00 000 000 000 000 inf 000 000
                000 000 k14 000 k24 000 000 000 k34 000 k44 000
                000 inf 000 000 000 inf 000 -inf 00 000 000 inf
            ]
        end

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
    elseif elemType == "BT3" # higher order composite beam 10 DOF
        K11 = coeff * [
            cz dz iz -abᵉ*az -(abᵉ * bz + atheta)
            dz gz jz -(abᵉ * bz - btheta) (-ftheta-abᵉ*ez)
            iz jz lz gtheta ctheta
            -abᵉ*az -(abᵉ * bz - btheta) gtheta atau btau
            -(abᵉ * bz + atheta) (-ftheta-abᵉ*ez) ctheta btau dtau+dtheta
        ]
        K12 = coeff * [
            -cz dz -iz abᵉ*az -(abᵉ * bz - atheta)
            -dz hz -kz (abᵉ*bz-btheta) etheta-abᵉ*fz
            -iz kz mz -gtheta ctheta
            abᵉ*az -(abᵉ * bz + btheta) gtheta -atau btau
            (abᵉ*bz+atheta) (-etheta-abᵉ*fz) ctheta -btau -ctau
        ]
        K22 = coeff * [
            cz -dz iz -abᵉ*az (abᵉ*bz-atheta)
            -dz gz -jz (abᵉ*bz+btheta) (ftheta-abᵉ*ez)
            iz -jz lz -gtheta ctheta
            -abᵉ*az (abᵉ*bz+btheta) -gtheta atau -btau
            (abᵉ*bz-atheta) (ftheta-abᵉ*ez) ctheta -btau dtau-dtheta
        ]
        Ktop = hcat(K11, K12)
        Kbot = hcat(K12', K22)
        Kᵉ = vcat(Ktop, Kbot)
    elseif elemType == "COMP2" # Higher order composite beam 18 DOF using a 4th order basis function in bending
        K11 = coeff * [
            ax 00 0000000 0000000 00 00 0000000000000000 00 0
            00 cy 0000000 0000000 00 dy 0000000000000000 00 iy
            00 00 cz -abᵉ*az dz 00 -(abᵉ * bz + atheta) iz 0
            00 00 -abᵉ*az atau -bz*abᵉ+btheta 00 btau gtheta 0
            00 00 dz -bz*abᵉ+btheta gz 00 -(abᵉ * ez + ftheta) jz 00
            00 dy 00 00 00 gy 0000000000000000 00 jy
            00 00 -(abᵉ * bz + atheta) btau -(abᵉ * ez + ftheta) 00 dtheta+dtau ctheta 00
            00 00 iz gtheta jz 00 ctheta lz 00
            00 iy 00 00 00 jy 0000000000000000 00 ly
        ]
        K12 = coeff * [
            -ax 00 0000000 0000000 00 00 0000000000000000 00 0
            00 -cy 0000000 0000000 00 dy 0000000000000000 00 -iy
            00 00 -cz abᵉ*az dz 00 -abᵉ*bz+atheta -iz 0
            00 00 abᵉ*az -atau -bz*abᵉ-btheta 00 btau gtheta 0
            00 00 -dz bz*abᵉ-btheta hz 00 -abᵉ*fz+etheta -kz 00
            00 -dy 00 00 00 hy 0000000000000000 00 -ky
            00 00 (abᵉ*bz+atheta) -btau -(abᵉ * fz + etheta) 00 -ctau ctheta 00
            00 00 -iz -gtheta kz 00 ctheta mz 00
            00 -iy 00 00 00 ky 0000000000000000 00 my
        ]
        K22 = coeff * [
            ax 00 0000000 0000000 00 00 0000000000000000 00 0
            00 cy 0000000 0000000 00 -dy 0000000000000000 00 iy
            00 00 cz -abᵉ*az -dz 00 (abᵉ*bz-atheta) iz 0
            00 00 -abᵉ*az atau bz*abᵉ+btheta 00 -btau -gtheta 0
            00 00 -dz bz*abᵉ+btheta gz 00 -abᵉ*ez+ftheta -jz 00
            00 -dy 00 00 00 gy 0000000000000000 00 -jy
            00 00 (abᵉ*bz-atheta) -btau -(abᵉ * ez + ftheta) 00 -dtheta+dtau ctheta 00
            00 00 iz -gtheta -jz 00 ctheta lz 00
            00 iy 00 00 00 -jy 0000000000000000 00 ly
        ]
        Ktop = hcat(K11, K12)
        Kbot = hcat(K12', K22)
        Kᵉ = vcat(Ktop, Kbot)
    end

    return Kᵉ
end

function compute_elem_mass(mᵉ, iᵉ, lᵉ, x_αbᵉ, elemType="bend-twist", dim=1)
    """
    The kinetic energy is
        T = 0.5∫₀ᴸ m (∂w/∂t)² dx = 0.5{q̇(t)}ᵀ[Mᵉ]{q̇(t)}

    Element mass matrix from the kinetic energies
    """

    # ************************************************
    #     Handy identities
    # ************************************************
    mb = mᵉ * lᵉ / 420
    mt = iᵉ * lᵉ / 6
    ax= 2*mᵉ*lᵉ/6
    bx= mᵉ*lᵉ^2/6
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
    atau = 156 * iᵉ * lᵉ / 420
    btau = 54 * iᵉ * lᵉ / 420
    ctau = 22 * iᵉ * lᵉ^2 / 420
    dtau = 13 * iᵉ * lᵉ^2 / 420
    etau = 4 * iᵉ * lᵉ^3 / 420
    ftau = 3 * iᵉ * lᵉ^3 / 420

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
        if dim == 3
            sml = 1e-3 * m11
            Mᵉ = [
                sml 000 000 000 000 000 000 000 000 000 000 000
                000 sml 000 000 000 000 000 000 000 000 000 000
                000 000 m11 000 m12 000 000 m13 000 m14 000 000
                000 000 000 sml 000 000 000 000 000 000 000 000
                000 000 m12 000 m22 000 000 m23 000 m24 000 000
                000 000 000 000 000 sml 000 000 000 000 000 000
                000 000 000 000 000 000 sml 000 000 000 000 000
                000 000 m13 000 m23 000 000 m33 000 m34 000 000
                000 000 000 000 000 000 000 000 sml 000 000 000
                000 000 m14 000 m24 000 000 m34 000 m44 000 000
                000 000 000 000 000 000 000 000 000 000 sml 000
                000 000 000 000 000 000 000 000 000 000 000 sml
            ]
        end
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
        atau = 156 * iᵉ * lᵉ / 420
        btau = 54 * iᵉ * lᵉ / 420
        ctau = 22 * iᵉ * lᵉ^2 / 420
        dtau = 13 * iᵉ * lᵉ^2 / 420
        etau = 4 * iᵉ * lᵉ^3 / 420
        ftau = 3 * iᵉ * lᵉ^3 / 420
        M11 = [
            az iz qz x_αbᵉ*bz x_αbᵉ*hz
            iz lz mz x_αbᵉ*fz x_αbᵉ*sz
            qz mz tz x_αbᵉ*pz x_αbᵉ*uz
            x_αbᵉ*bz x_αbᵉ*fz x_αbᵉ*pz atau ctau
            x_αbᵉ*hz x_αbᵉ*sz x_αbᵉ*uz ctau etau
        ]
        M12 = [
            dz -jz rz x_αbᵉ*cz -x_αbᵉ*gz
            jz -kz nz x_αbᵉ*ez -x_αbᵉ*vz
            rz -nz xz x_αbᵉ*oz -x_αbᵉ*wz
            x_αbᵉ*cz -x_αbᵉ*ez x_αbᵉ*oz btau -dtau
            x_αbᵉ*gz -x_αbᵉ*vz x_αbᵉ*wz dtau -ftau
        ]
        M22 = [
            az -iz qz x_αbᵉ*bz -x_αbᵉ*hz
            -iz lz -mz -x_αbᵉ*fz x_αbᵉ*sz
            qz -mz tz x_αbᵉ*pz -x_αbᵉ*uz
            x_αbᵉ*bz -x_αbᵉ*fz x_αbᵉ*pz atau -ctau
            -x_αbᵉ*hz x_αbᵉ*sz -x_αbᵉ*uz -ctau etau
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    elseif elemType == "COMP2"
        M11 = [
            ax 00000     0000 00000000     0000 0000     0000      0000 0000
            00    ay     0000 00000000     0000   iy     0000      0000   qy
            00  0000       az x_αbᵉ*bz       iz 0000 x_αbᵉ*hz        qz 0000
            00  0000 x_αbᵉ*bz     atau x_αbᵉ*fz 0000     ctau  x_αbᵉ*pz 0000
            00  0000       iz x_αbᵉ*fz       lz 0000 x_αbᵉ*sz        mz 0000
            00   iy      0000     0000     0000   ly     0000      0000   my
            00  0000 x_αbᵉ*hz     ctau x_αbᵉ*sz 0000     etau  x_αbᵉ*uz 0000
            00  0000       qz x_αbᵉ*pz       mz 0000 x_αbᵉ*uz        tz 0000
            00    qy    0000     0000     0000   my     0000      0000    ty
        ]
        M12 = [
            bx 00000     0000 00000000     0000 0000     0000      0000 0000
            00    dy     0000 00000000     0000  -jy     0000      0000   ry
            00  0000       dz x_αbᵉ*cz      -jz 0000 -x_αbᵉ*gz       rz 0000
            00  0000 x_αbᵉ*cz btau    -x_αbᵉ*ez 0000    -dtau  x_αbᵉ*oz 0000
            00  0000       jz x_αbᵉ*ez      -kz 0000 -x_αbᵉ*vz       nz 0000
            00  jy       0000     0000     0000  -ky     0000      0000   ny
            00  0000 x_αbᵉ*gz dtau    -x_αbᵉ*vz 0000    -ftau -x_αbᵉ*wz 0000
            00 0000       rz x_αbᵉ*oz      -nz  0000 -x_αbᵉ*wz       xz 0000
            00    ry     0000     0000     0000  -ny     0000      0000   xy
        ]
        M22 = [
            ax 00000     0000 00000000     0000 0000     0000      0000 0000
            00    ay     0000 00000000     0000  -iy     0000      0000   qy
            00  0000       az x_αbᵉ*bz      -iz 0000 -x_αbᵉ*hz       qz 0000
            00  0000 x_αbᵉ*bz     atau -x_αbᵉ*fz 0000    -ctau  x_αbᵉ*pz 0000
            00  0000      -iz -x_αbᵉ*fz       lz 0000 x_αbᵉ*sz       -mz 0000
            00  -iy      0000     0000     0000   ly     0000      0000  -my
            00  0000 -x_αbᵉ*hz   -ctau x_αbᵉ*sz 0000     etau  -x_αbᵉ*uz 0000
            00  0000       qz x_αbᵉ*pz     -mz 0000 -x_αbᵉ*uz        tz 0000
            00    qy    0000     0000     0000  -my     0000      0000    ty
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

function make_mesh(nElem::Int64, span; config="wing", rotation=0.000, nElStrut=0, spanStrut=0.0)
    """
    Makes a mesh and element connectivity
    First element is always origin (x,y,z) = (0,0,0)
    You do not necessarily have to make this mesh yourself every run

    Inputs
    ------
    nElem: 
        number of elements
    config: 
        "wing" or "t-foil"
    rotation: 
        rotation of the foil in degrees where 0 is lifting up in 'z'
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
        mesh_z[:, YDIM] = collect((0:dl:span))
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

function get_transMat(dR, l, elemType="BT2", dim=3)
    """
    Returns the transformation matrix for a given element type into 3D space
    """

    rxy_div = 1 / sqrt(dR[XDIM]^2 + dR[YDIM]^2) # length of projection onto xy plane
    calpha = dR[1] * rxy_div
    salpha = dR[2] * rxy_div
    cbeta = 1 / rxy_div / l
    sbeta = dR[3] / l

    # Direction cosine matrix
    T = [
        calpha*cbeta salpha calpha*sbeta
        -salpha*cbeta calpha -salpha*sbeta
        -sbeta 0 cbeta
    ]
    writedlm("DebugT.csv", T, ',')

    if elemType == "BT2"
        # Because BT2 had reduced DOFs, we need to transform the reduced DOFs into 3D space which results in storing more numbers
        Γ = Matrix(I, 8, 8)
    elseif elemType == "bend-twist"
        Γ = Matrix(I, 6, 6)
    elseif elemType == "BT3"
        Γ = Matrix(I, 10, 10)
    elseif elemType == "bend"
        if dim == 3
            # 4x12
            Γ = [
                T zeros(3, 3) zeros(3, 3) zeros(3, 3)
                zeros(3, 3) T zeros(3, 3) zeros(3, 3)
                zeros(3, 3) zeros(3, 3) T zeros(3, 3)
                zeros(3, 3) zeros(3, 3) zeros(3, 3) T
            ]
            # Γ = Matrix(I, 4, 4)
            writedlm("DebugGamma.csv", Γ, ',')
        else
            error("Only 3D bend implemented")
        end
    elseif elemType == "BEAM3D"
        if dim == 3
            # 12x12
            Γ = [
                T zeros(3, 3) zeros(3, 3) zeros(3, 3)
                zeros(3, 3) T zeros(3, 3) zeros(3, 3)
                zeros(3, 3) zeros(3, 3) T zeros(3, 3)
                zeros(3, 3) zeros(3, 3) zeros(3, 3) T
            ]
        else
            error("Unsupported dimension")
        end
    else
        error("Unsupported element type")
    end

    for ii in eachindex(Γ[:, 1])
        for jj in eachindex(Γ[1, :])
            if abs(Γ[ii, jj]) < 1e-16
                Γ[ii, jj] = 0.0
            end
        end
    end
    # show(stdout, "text/plain", Γ)
    return Γ
end

function assemble(coordMat, elemConn, abVec, x_αbVec, FOIL, elemType="bend-twist", constitutive="isotropic", dim=3)
    """
    Generic function to assemble the global mass and stiffness matrices

    Inputs
    ------
        coordMat: 2D array of coordinates of nodes
        elemConn: 2D array of element connectivity (nElem x 2)
    """

    # --- Local nodal DOF vector ---
    # Determine the number of dofs per node
    if elemType == "bend"
        nnd = 2
        # This is a reduced dim element so we need to store more numbers
        nndG = dim * nnd
        # nndG = 2
    elseif elemType == "bend-twist"
        nnd = 3
        nndG = nnd
    elseif elemType == "BT2"
        nnd = 4
        nndG = nnd
    elseif elemType == "BT3"
        nnd = 5
        nndG = nnd
    elseif elemType == "BEAM3D"
        nnd = 6
        nndG = nnd

    else
        error(elemType, " element type not implemented")
    end
    qLocal = zeros(nnd * 2)

    # --- Initialize matrices ---
    nElem::Int64 = size(elemConn)[1]
    nNodes = nElem + 1
    ndim = ndims(coordMat[1, :])
    globalK::Matrix{Float64} = zeros(nndG * (nNodes), nndG * (nNodes))
    globalM::Matrix{Float64} = zeros(nndG * (nNodes), nndG * (nNodes))
    globalF::Vector{Float64} = zeros(nndG * (nNodes))


    # --- Debug printout for initialization ---
    ChainRulesCore.ignore_derivatives() do
        println("+", "-"^50, "+")
        println("|   Assembling global stiffness and mass matrices  |")
        println("+", "-"^50, "+")
        println("Default 2 nodes per elem, nothing else will work")
        println("Using ", elemType, " elements")
        println("Using ", constitutive, " constitutive relations...")
        println(nElem, " elements")
        println(nNodes, " nodes")
        println(nnd * nNodes, " total DOFs")
    end

    # ************************************************
    #     Element loop
    # ************************************************
    # --- Zygote buffer initializations ---
    globalK_z = Zygote.Buffer(globalK)
    globalM_z = Zygote.Buffer(globalM)
    globalF_z = Zygote.Buffer(globalF)
    for jj in 1:nndG*nNodes
        globalF_z[jj] = 0.0
        for ii in 1:nndG*nNodes
            globalK_z[jj, ii] = 0.0
            globalM_z[jj, ii] = 0.0
        end
    end
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
        kLocal::Matrix{Float64} = LinearBeamElem.compute_elem_stiff(EIₛ, EIIPₛ, GJₛ, Kₛ, Sₛ, EAₛ, lᵉ, ab, elemType, constitutive, false, dim)

        # ---------------------------
        #   Local mass matrix
        # ---------------------------
        mLocal::Matrix{Float64} = LinearBeamElem.compute_elem_mass(mₛ, iₛ, lᵉ, x_αb, elemType, dim)

        # ---------------------------
        #   Local force vector
        # ---------------------------
        fLocal::Vector{Float64} = zeros(nndG * 2)

        # ---------------------------
        #   Transform from local to global coordinates
        # ---------------------------
        #  AEROSP510 notes and python code, Engineering Vibration Chapter 8 (Inman 2014)
        # The local coordinate system is {u} while the global is {U}
        # {u} = [Γ] * {U}
        # where [Γ] is the transformation matrix
        Γ = get_transMat(dR, lᵉ, elemType, dim)
        kElem = Γ' * kLocal * Γ
        mElem = Γ' * mLocal * Γ
        fElem = Γ' * fLocal
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
            for dofIdx ∈ 1:nndG # loop over DOFs in node
                idxRow = ((elemConn[elemIdx, nodeIdx] - 1) * nndG + dofIdx) # idx of global dof (row of global matrix)
                idxRowₑ = (nodeIdx - 1) * nndG + dofIdx # idx of dof within this element

                # --- Assemble RHS ---
                globalF_z[idxRow] = fElem[idxRowₑ]

                # --- Assemble LHS ---
                for nodeColIdx ∈ 1:2 # loop over nodes in element
                    for dofColIdx ∈ 1:nndG # loop over DOFs in node
                        idxCol = (elemConn[elemIdx, nodeColIdx] - 1) * nndG + dofColIdx # idx of global dof (col of global matrix)
                        idxColₑ = (nodeColIdx - 1) * nndG + dofColIdx # idx of dof within this element (column)

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

function get_fixed_nodes(elemType::String, BCCond="clamped", dim=3)
    """
    Depending on the elemType, return the indices of fixed nodes
    """
    if BCCond == "clamped"
        if elemType == "bend"
            fixedNodes = [1, 2]
            if dim == 2
                fixedNodes = [1, 2, 3, 4]
            elseif dim == 3
                fixedNodes = [1, 2, 3, 4, 5, 6]
            end

        elseif elemType == "bend-twist"
            fixedNodes = [1, 2, 3]

        elseif elemType == "BT2"
            fixedNodes = [1, 2, 3, 4]
            if dim == 2
                # now the fixed nodes are [wx, wy, ∂wx, ∂wy, θx, ∂θx, θy, ∂θy]
                fixedNodes = [1, 2, 3, 4, 5, 6, 7, 8]
            elseif dim == 3
                # now the fixed nodes are [wx, wy, wz, ∂wx, ∂wy, ∂wz, θx, θy, θz, ∂θx, ∂θy, ∂θz]
                fixedNodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            end
        elseif elemType == "BT3"
            fixedNodes = [1, 2, 3, 4, 5]
        elseif elemType == "BEAM3D"
            fixedNodes = [1, 2, 3, 4, 5, 6]
        else
            error("elemType not recognized")

        end

    else
        error("BCCond not recognized")
    end

    return fixedNodes
end

function apply_tip_load!(globalF, elemType, transMat, loadType="force")
    """
    Routine for applying unit tip load to the end node
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
        else
            error("element not defined")
        end
    end

    # --- Transform to global then add into vector ---
    FLocalVec = transMat[1:m÷2, 1:n÷2]' * FLocalVec
    globalF[end-n÷2+1:end] += FLocalVec * MAG

end

function apply_tip_mass(globalM, mass, inertia, elemLength, x_αbVec, elemType="BT2")
    """
    Apply a tip mass to the global mass matrix

    mass: mass of the tip [kg]
    inertia: moment of inertia of the tip about C.G. [kg-m^2]
    """

    globalM_z = Zygote.Buffer(globalM)
    for jj in eachindex(globalM[:, 1])
        for ii in eachindex(globalM[1, :])
            globalM_z[jj, ii] = globalM[jj, ii]
        end
    end
    if elemType == "bend-twist"
        println("Does not work")
    elseif elemType == "BT2"
        nDOF = 8
        # --- Get sectional properties ---
        ms = mass / elemLength
        # Parallel axis theorem
        Iea = inertia + mass * (x_αbVec[end])^2
        is = Iea / elemLength
        tipMassMat = LinearBeamElem.compute_elem_mass(ms, is, elemLength, x_αbVec[end], elemType)

        # --- Assemble into global matrix ---
        globalM_z[end-nDOF+1:end, end-nDOF+1:end] += tipMassMat
    end

    ChainRulesCore.ignore_derivatives() do
        println("+------------------------------------+")
        println("|    Tip mass added!                 |")
        println("+------------------------------------+")
        println("Dist. CG is aft of EA: ", x_αbVec[end], " [m]")
    end
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
            uSol = vcat([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], q) # global now
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

    q = K \ F # TODO: should probably replace this with an iterative solver

    return q
end

function compute_modal(K, M, nEig::Int64)
    """
    Compute the eigenvalues (natural frequencies) and eigenvectors (mode shapes) of the in-vacuum system.
    i.e., this is structural dynamics, not hydroelastics.
    """

    # use krylov method to get first few smallest eigenvalues
    # Solve [K]{x} = λ[M]{x} where λ = ω²
    eVals, eVecs = SolverRoutines.compute_eigsolve(K, M, nEig)

    naturalFreqs = sqrt.(eVals) / (2π)

    return naturalFreqs, eVecs
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