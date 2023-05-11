"""
Run tests on hydro module file
"""

using LinearAlgebra
using Printf
include("../src/hydro/Hydro.jl")
using .Hydro # Using the Hydro module
include("../src/solvers/DCFoilSolution.jl")
include("../src/constants/SolutionConstants.jl")
include("../src/struct/FiniteElements.jl")
include("../src/InitModel.jl")
include("../src/solvers/SolveStatic.jl")
using .FEMMethods # Using the FEMMethods module just for some mesh gen methods
using .InitModel # Using the InitModel module
using .SolutionConstants, .SolveStatic

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
    Need to test mesh independence of hydro loads on a rigid hydrofoil
    """

    # --- Reference value ---
    # These were obtained from hand calcs
    ref_sol = vec([])

    elemType = "BT2"
    evalFuncs = ["lift", "moment", "cl", "cmy"]

    nNodess = [10, 20, 40, 80, 160, 320]
    liftData = zeros(length(nNodess))
    momData = zeros(length(nNodess))
    clData = zeros(length(nNodess))
    cmyData = zeros(length(nNodess))
    meshlvl = 1

    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 30.0 * π / 180, # sweep angle [rad]
        "g" => 0.04, # structural damping percentage
        "c" => 1 * ones(nNodess[1]), # chord length [m]
        "s" => 1.0, # semispan [m]
        "ab" => zeros(nNodess[1]), # dist from midchord to EA [m]
        "toc" => 1, # thickness-to-chord ratio
        "x_αb" => zeros(nNodess[1]), # static imbalance [m]
        "θ" => 0 * π / 180, # fiber angle global [rad]
    )
    solverOptions = Dict(
        "outputDir" => "test_out/",
        "nNodes" => nNodess[1],
        "U∞" => 5.0, # free stream velocity [m/s]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "material" => "test-comp", # preselect from material library
    )
    mkpath(solverOptions["outputDir"])
    for nNodes in nNodess
        solverOptions["nNodes"] = nNodes
        DVDict["c"] = 1 * ones(nNodes)
        DVDict["ab"] = zeros(nNodes)
        DVDict["x_αb"] = zeros(nNodes)

        FOIL = InitModel.init_static(DVDict, solverOptions)
        nElem = nNodes - 1
        constitutive = FOIL.constitutive
        structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)

        # _, _, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, constitutive)
        _, _, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, constitutive)


        # --- Initialize states ---
        u = copy(globalF)
        u .= 0.0 # NOTE: Because we don't actually do the FEMSolve, this is a rigid hydrofoil

        # ---------------------------
        #   Get initial fluid tracts
        # ---------------------------
        fTractions, AIC, planformArea = Hydro.compute_steady_hydroLoads(u, structMesh, FOIL, elemType)
        forces = fTractions
        CONSTANTS = SolutionConstants.DCFoilConstants(zeros(2, 2), zeros(2, 2), elemType, structMesh, AIC, "RAD", planformArea)

        costFuncs = SolveStatic.evalFuncs(forces, forces, evalFuncs; constants=CONSTANTS, foil=FOIL)

        liftData[meshlvl] = costFuncs["lift"]
        momData[meshlvl] = costFuncs["moment"]
        clData[meshlvl] = costFuncs["cl"]
        cmyData[meshlvl] = costFuncs["cmy"]

        meshlvl += 1
    end

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = solverOptions["outputDir"] * "test_hydroLoads.out"
    meshlvl = 1
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_hydroLoads\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | lift [N] | mom. [N-m] |  cl   |  cmy   |\n")
        for lift in liftData
            mom = momData[meshlvl]
            cl = clData[meshlvl]
            cmy = cmyData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f   %.16f    %.16f\n", meshlvl, nNodess[meshlvl], lift, mom, cl, cmy)
            write(io, line)
            meshlvl += 1
        end
    end

    return 0
end




