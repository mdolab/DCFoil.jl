"""
unit test to verify the beam bend and bend-twist element
"""

include("../src/InitModel.jl")
include("../src/struct/FiniteElements.jl")

using LinearAlgebra
using .FEMMethods, .InitModel
using .LinearBeamElem


function test_FiniteElement()
    """
    Test the finite elements with unit loads, thickness, length, and structural moduli
    """
    neval = 30
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 5.0, # free stream velocity [m/s]
        "Λ" => 30.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000, # fluid density [kg/m³]
        "material" => "test", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 1 * ones(neval), # chord length [m]
        "s" => 1, # semispan [m]
        "ab" => zeros(neval), # dist from midchord to EA [m]
        "toc" => 1, # thickness-to-chord ratio
        "x_αb" => zeros(neval), # static imbalance [m]
        "θ" => 0 * π / 180, # fiber angle global [rad]
    )
    foil = InitModel.init_steady(neval, DVDict)

    nElem = neval - 1
    constitutive = "isotropic"
    # ************************************************
    #     bend element
    # ************************************************
    elemType = "bend"
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType)

    structMesh, elemConn = FEMMethods.make_mesh(nElem, foil)
    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, foil, elemType, constitutive)
    globalF[end-1] = 1.0 # 1 Newton tip force NOTE: FIX LATER bend
    u = copy(globalF)

    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    q1 = FEMMethods.solve_structure(K, M, F)

    # ************************************************
    #     bend-twist 
    # ************************************************
    # ---------------------------
    #   Tip force only
    # ---------------------------
    elemType = "bend-twist"
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType)
    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, foil, elemType, constitutive)
    globalF[end-2] = 1.0 # 0 Newton tip force
    u = copy(globalF)


    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    q2 = FEMMethods.solve_structure(K, M, F)

    # ---------------------------
    #   Tip torque only
    # ---------------------------
    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, foil, elemType, constitutive)
    globalF[end] = 1.0 # 0 Newton tip force
    u = copy(globalF)


    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    q3 = FEMMethods.solve_structure(K, M, F)

    # --- Reference value ---
    # the tip deformations should be 4m for pure bending with tip force and 3 radians for tip torque
    ref_sol = [4, 4, 3]

    # --- Relative error ---
    answers = [q1[end-1], q2[end-2], q3[end]] # put computed solutions here
    rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

    return rel_err
end

function test_BT2_stiff()
    """
    Test the second order beam matrix with unit values
    """
    constitutive = "orthotropic"
    # ************************************************
    #     BT2 element stiff
    # ************************************************
    elemType = "BT2"

    Ktest = LinearBeamElem.compute_elem_stiff(2, 8, 4, 8 / 3, 2, 1, elemType, constitutive)
    # show(stdout, "text/plain", Ktest[5:end, 5:end])

    # --- Reference value ---
    # These were obtained from the matlab symbolic script plugging 1 for flexural stiffnesses and 2 for the chord
    ref_sol = vec([
        3.0000 3.0000 -3.0000 -5.0000 -3.0000 3.0000 3.0000 -1.0000
        3.0000 4.0000 -1.0000 -6.0000 -3.0000 2.0000 1.0000 0
        -3.0000 -1.0000 8.8000 4.8000 3.0000 -5.0000 -8.8000 4.8000
        -5.0000 -6.0000 4.8000 11.4667 5.0000 -4.0000 -4.8000 2.1333
        -3.0000 -3.0000 3.0000 5.0000 3.0000 -3.0000 -3.0000 1.0000
        3.0000 2.0000 -5.0000 -4.0000 -3.0000 4.0000 5.0000 -2.0000
        3.0000 1.0000 -8.8000 -4.8000 -3.0000 5.0000 8.8000 -4.8000
        -1.0000 0 4.8000 2.1333 1.0000 -2.0000 -4.8000 3.4667
    ])

    # # --- Relative error ---
    answers = vec(Ktest) # put computed solutions here
    rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

    if det(Ktest) != 0
        print("Your stiffness matrix is not singular...it's wrong")
        rel_err += 1 # make test fail
    end

    if Ktest' != Ktest
        print("Your stiffness matrix is not symmetric...it's wrong")
        rel_err += 1 # make test fail
    end

    return rel_err
end

function test_BT2_mass()
    """
    Test the second order beam matrix with unit values
    """
    # ************************************************
    #     BT2 element mass
    # ************************************************
    elemType = "BT2"

    Mtest = LinearBeamElem.compute_elem_mass(4, 16 / 3, 2, -1, elemType)
    # show(stdout, "text/plain", Mtest[5:end, 5:end])

    # --- Reference value ---
    # These were obtained from the matlab symbolic script plugging 2 for rho, 2 for the chord, and 1 for everything else
    ref_sol = vec([
        2.9714 0.8381 -2.9714 -0.8381 1.0286 -0.4952 -1.0286 0.4952
        0.8381 0.3048 -0.8381 -0.3048 0.4952 -0.2286 -0.4952 0.2286
        -2.9714 -0.8381 3.9619 1.1175 -1.0286 0.4952 1.3714 -0.6603
        -0.8381 -0.3048 1.1175 0.4063 -0.4952 0.2286 0.6603 -0.3048
        1.0286 0.4952 -1.0286 -0.4952 2.9714 -0.8381 -2.9714 0.8381
        -0.4952 -0.2286 0.4952 0.2286 -0.8381 0.3048 0.8381 -0.3048
        -1.0286 -0.4952 1.3714 0.6603 -2.9714 0.8381 3.9619 -1.1175
        0.4952 0.2286 -0.6603 -0.3048 0.8381 -0.3048 -1.1175 0.4063
    ])

    # # --- Relative error ---
    answers = vec(Mtest) # put computed solutions here
    rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

    if minimum(eigvals(Mtest)) < 0
        print("Your stiffness matrix is not positive definite...it's wrong")
        rel_err += 1 # make test fail
    end

    return rel_err
end