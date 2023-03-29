"""
Run tests on hydro module file
"""

using LinearAlgebra

include("../src/hydro/Hydro.jl")
using .Hydro # Using the Hydro module

include("../src/struct/FiniteElements.jl")
include("../src/InitModel.jl")
using .FEMMethods # Using the FEMMethods module just for some mesh gen methods
using .InitModel # Using the InitModel module

# ==============================================================================
#                         Nodal hydrodynamic forces
# ==============================================================================
function test_stiffness()
    """
    Compare strip forces to a hand calculated reference solution
    TODO: use a non-zero 'a'
    """

    # --- Reference value ---
    # These were obtained from hand calcs
    ref_sol = vec([
        0.0 -6250*π
        0.0 -6250/4*π
    ])
    sweepref_sol = vec([
        3125*2π -3125*π/2
        3125*π/2 3125*π/8
    ])

    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0
    U = 5
    Λ = 45 * π / 180 # 45 deg
    ω = 1e10 # infinite frequency limit
    ρ = 1000.0
    k = ω * b / (U * cos(Λ))
    CKVec = Hydro.compute_theodorsen(k)
    Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2]
    Matrix, SweepMatrix = Hydro.compute_node_stiff(clα, b, eb, ab, U, Λ, ρ, Ck)

    # show(stdout, "text/plain", real(Matrix))
    # show(stdout, "text/plain", imag(Matrix))
    # show(stdout, "text/plain", real(SweepMatrix))
    # show(stdout, "text/plain", imag(SweepMatrix))

    # --- Relative error ---
    answers = vec(real(Matrix)) # put computed solutions here
    rel_err1 = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)
    # For sweep
    sweepAnswers = vec(real(SweepMatrix)) # put computed solutions here
    rel_err2 = LinearAlgebra.norm(sweepAnswers - sweepref_sol, 2) / LinearAlgebra.norm(sweepref_sol, 2)

    # Just take the max error
    rel_err = max(abs(rel_err1), abs(rel_err2))

    return rel_err
end

function test_damping()
    """
    Compare strip forces to a hand calculated reference solution
    TODO: use a non-zero 'a'
    """
    # --- Reference value ---
    # These were obtained from hand calcs
    ref_sol = 625 * sqrt(2) * vec([
                  2π -1.5*π
                  0.5*π 0.125*π
              ])
    sweepref_sol = 625 * sqrt(2) * vec([
                       π 0.0
                       0.0 π/32
                   ])

    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0
    U = 5
    Λ = 45 * π / 180 # 45 deg
    ω = 1e10 # infinite frequency limit
    ρ = 1000.0
    k = ω * b / (U * cos(Λ))
    CKVec = Hydro.compute_theodorsen(k)
    Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2] # TODO: for now, put it back together so solve is easy to debug
    Matrix, SweepMatrix = Hydro.compute_node_damp(clα, b, eb, ab, U, Λ, ρ, Ck)

    # --- Relative error ---
    answers = vec(real(Matrix)) # put computed solutions here
    rel_err1 = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)
    # For sweep
    sweepAnswers = vec(real(SweepMatrix)) # put computed solutions here
    rel_err2 = LinearAlgebra.norm(sweepAnswers - sweepref_sol, 2) / LinearAlgebra.norm(sweepref_sol, 2)

    # Just take the max error and normalize it by the norm of the sweep reference solution
    rel_err = max(abs(rel_err1), abs(rel_err2))

    # println(ref_sol)
    # println(answers)
    # println(sweepref_sol)
    # println(sweepAnswers)

    return rel_err
end

function test_mass()
    # --- Reference value ---
    # These were obtained from hand calcs
    ref_sol = 250 * π * vec([
                  1.0 0.25
                  0.25 3/32
              ])

    clα = 2 * π
    b = 0.5
    eb = 0.25
    ab = 0.25
    U = 5.0
    Λ = 45 * π / 180 # 45 deg
    ω = 1e10 # infinite frequency limit
    ρ = 1000.0
    Matrix = Hydro.compute_node_mass(b, ab, ρ)

    # --- Relative error ---
    answers = vec(real(Matrix)) # put computed solutions here
    rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)


    # println(ref_sol)
    # println(answers)

    return rel_err
end

# ==============================================================================
#                         Larger scale tests
# ==============================================================================
function test_hydroLoads()
    """
    Need to test mesh independance of hydro loads on a rigid hydrofoil
    """

    # --- Reference value ---
    # These were obtained from hand calcs
    ref_sol = vec([])

    elemType = "BT2"

    nNodess = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    liftData = zeros(length(nNodess))
    momData = zeros(length(nNodess))
    meshlvl = 1

    for nNodes in nNodess

        DVDict = Dict(
            "nNodes" => nNodes,
            "α₀" => 6.0, # initial angle of attack [deg]
            "U∞" => 5.0, # free stream velocity [m/s]
            "Λ" => 30.0 * π / 180, # sweep angle [rad]
            "ρ_f" => 1000.0, # fluid density [kg/m³]
            "material" => "test-comp", # preselect from material library
            "g" => 0.04, # structural damping percentage
            "c" => 1 * ones(nNodes), # chord length [m]
            "s" => 1.0, # semispan [m]
            "ab" => zeros(nNodes), # dist from midchord to EA [m]
            "toc" => 1, # thickness-to-chord ratio
            "x_αb" => zeros(nNodes), # static imbalance [m]
            "θ" => 0 * π / 180, # fiber angle global [rad]
        )
        FOIL = InitModel.init_static(nNodes, DVDict)

        nElem = nNodes - 1
        constitutive = FOIL.constitutive
        structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)

        _, _, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, constitutive)

        # --- Initialize states ---
        u = copy(globalF)
        u .= 0.0 # NOTE: Because we don't actually do the FEMSolve, this is a rigid hydrofoil

        # ---------------------------
        #   Get initial fluid tracts
        # ---------------------------
        fTractions, AIC, planformArea = Hydro.compute_steady_hydroLoads(u, structMesh, FOIL, elemType)
        forces = fTractions

        # ---------------------------
        #   Force integration
        # ---------------------------
        if elemType == "BT2"
            nDOF = 4
            Moments = forces[3:nDOF:end]
            Lift = forces[1:nDOF:end]
        else
            println("Invalid element type")
        end
        TotalLift = sum(Lift)
        TotalMoment = sum(Moments)

        liftData[meshlvl] = TotalLift
        momData[meshlvl] = TotalMoment

        meshlvl += 1
    end

    # # --- Print hydro loads ---
    # println("Lift: ", liftData)
    # println("Moment: ", momData)
end




