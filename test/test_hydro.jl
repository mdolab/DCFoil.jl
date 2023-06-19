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


function test_AICs()
    AIC = zeros(24, 24)
    aeroMesh = [
        0.0 0.0 0.0
        0.0 1.0 0.0
    ]
    chordVec = [1.0, 1.0]
    abVec = [0.0, 0.0]
    ebVec = [0.5, 0.5]
    Λ = 0.0
    nNodes = 2
    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => deg2rad(-15.0), # sweep angle [rad]
        "g" => 0.04, # structural damping percentage
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        "strut" => 0.4, # from Yingqian
    )

    solverOptions = Dict(
        # ---------------------------
        #   I/O
        # ---------------------------
        # "name" => "akcabay-swept",
        "name" => "t-foil",
        "debug" => false,
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "config" => "wing",
        # "config" => "t-foil",
        "nNodes" => nNodes, # number of nodes on foil half wing
        "nNodeStrut" => 10, # nodes on strut
        "rotation" => 45.0, # deg
        "gravityVector" => [0.0, 0.0, -9.81],
        "tipMass" => false,
        # ---------------------------
        #   Flow
        # ---------------------------
        "U∞" => 5.0, # free stream velocity [m/s]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "use_freeSurface" => false,
        "use_cavitation" => false,
        "use_ventilation" => false,
        # ---------------------------
        #   Structure
        # ---------------------------
        "material" => "cfrp", # preselect from material library
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static" => true,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 1:2,
        "tipForceMag" => 1,
        # --- p-k (Eigen) solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 1,
        "uRange" => [1,2],
        "maxQIter" => 100, # that didn't fix the slow run time...
        "rhoKS" => 80.0,
    )

    FOIL = InitModel.init_model_wrapper(DVDict,solverOptions)

    AIC, planformArea = Hydro.compute_steady_AICs!(AIC, aeroMesh, chordVec, abVec, ebVec, Λ, FOIL, "BT2")

    return AIC
end

AIC = test_AICs()