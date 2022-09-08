"""
unit test to verify the beam bend and bend-twist element
"""
include("../src/InitModel.jl")
include("../src/struct/FiniteElements.jl")

using LinearAlgebra


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
        "x_α" => zeros(neval), # static imbalance [m]
        "θ" => 0 * π / 180, # fiber angle global [rad]
    )
    foil = InitModel.init_steady(neval, DVDict)

    nElem = neval - 1
    constitutive = "isotropic"
    # ************************************************
    #     bend element
    # ************************************************
    elemType = "bend"
    globalDOFBlankingList = [1, 2]

    structMesh, elemConn = FEMMethods.make_mesh(nElem, foil)
    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, foil, elemType, constitutive)
    globalF[end-1] = 1.0 # 1 Newton tip force NOTE: FIX LATER bend
    u = copy(globalF)

    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    q1 = FEMMethods.solve_structure(K, M, F)

    # ************************************************
    #     Solve with bend-twist element
    # ************************************************
    # ---------------------------
    #   Tip force only
    # ---------------------------
    elemType = "bend-twist"
    globalDOFBlankingList = [1, 2, 3]
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
    answers = [q1[end-1], q2[end-3], q3[end]] # put computed solutions here
    rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

    return rel_err
end