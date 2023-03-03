"""
Test to verify that the static solve works without failing
"""

using Printf
using LinearAlgebra

include("../src/solvers/SolveStatic.jl")
using .SolveStatic
include("../src/solvers/SolveForced.jl")
using .SolveForced
include("../src/solvers/SolveFlutter.jl")
using .SolveFlutter

# ==============================================================================
#                         Test Static Solver
# ==============================================================================
function test_SolveStaticRigid()
    """
    Very simple mesh convergence test with hydro and structural solvers over different numbers of nodes
    Rigid beam
    """

    # --- Reference value ---
    # It should be zero
    refBendSol = [0.0000002080248011, 0.0000002068145797, 0.0000002072123383]
    refTwistSol = [0.0000000159776292, 0.0000000160026983, 0.0000000160364539]

    nevals = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    neval = nevals[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 6.0, # free stream velocity [m/s]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "material" => "rigid", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 0.1 * ones(neval), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(neval), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(neval), # static imbalance [m]
        "θ" => 15 * π / 180, # fiber angle global [rad]
    )

    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nevals))
    tipTwistData = zeros(length(nevals))
    meshlvl = 1
    for neval in nevals
        # --- Resize some stuff ---
        DVDict["neval"] = neval
        DVDict["c"] = 0.1 * ones(neval)
        DVDict["ab"] = 0 * ones(neval)
        DVDict["x_αb"] = 0 * ones(neval)
        # --- Solve ---
        costFuncs = SolveStatic.solve(DVDict, evalFuncs, "../OUTPUT")
        tipBendData[meshlvl] = costFuncs["w_tip"]
        tipTwistData[meshlvl] = costFuncs["psi_tip"]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticRigid\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

    # --- Relative error ---
    answers = vec(tipBendData) # put computed solutions here
    rel_err1 = LinearAlgebra.norm(answers - refBendSol, 2) / LinearAlgebra.norm(refBendSol, 2)
    answers = vec(tipTwistData) # put computed solutions here
    rel_err2 = LinearAlgebra.norm(answers - refTwistSol, 2) / LinearAlgebra.norm(refTwistSol, 2)
    rel_err = max(rel_err1, rel_err2)

    # println("Relative error: ", rel_err1)
    # println("Relative error: ", rel_err2)

    return rel_err

end # test_SolveStaticRigid

function test_SolveStaticIso()
    """
    Very simple mesh convergence test with hydro and structural solvers over different numbers of nodes
    Stainless steel beam
    """
    # --- Reference value ---
    #  Obtained by running the code
    refBendSol = [0.0002080524592060, 0.0002068421525681, 0.0002072400251625]
    refTwistSol = [0.0000159796182303, 0.0000160046939335, 0.0000160384580202]

    nevals = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    neval = nevals[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 6.0, # free stream velocity [m/s]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "material" => "ss", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 0.1 * ones(neval), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(neval), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(neval), # static imbalance [m]
        "θ" => 15 * π / 180, # fiber angle global [rad]
    )

    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nevals))
    tipTwistData = zeros(length(nevals))
    meshlvl = 1
    for neval in nevals
        # --- Resize some stuff ---
        DVDict["neval"] = neval
        DVDict["c"] = 0.1 * ones(neval)
        DVDict["ab"] = 0 * ones(neval)
        DVDict["x_αb"] = 0 * ones(neval)
        # --- Solve ---
        costFuncs = SolveStatic.solve(DVDict, evalFuncs, "../OUTPUT")
        tipBendData[meshlvl] = costFuncs["w_tip"]
        tipTwistData[meshlvl] = costFuncs["psi_tip"]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticIso\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

    # --- Relative error ---
    rel_err1 = LinearAlgebra.norm(tipBendData - refBendSol, 2) / LinearAlgebra.norm(refBendSol, 2)
    rel_err2 = LinearAlgebra.norm(tipTwistData - refTwistSol, 2) / LinearAlgebra.norm(refTwistSol, 2)
    rel_err = max(rel_err1, rel_err2)

    return rel_err
end # test_SolveStaticIso

function test_SolveStaticComp()
    """
    Very simple mesh convergence test with hydro and structural solvers over different numbers of nodes
    Composite beam
    """

    # --- Reference value ---
    #  Obtained by running the code
    refBendSol = [0.0004975455285840, 0.0004938633758853, 0.0004949057635612]
    refTwistSol = [-0.0008526586938542, -0.0008359901040122, -0.0008387865527998]

    nevals = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    neval = nevals[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 6.0, # free stream velocity [m/s]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "material" => "cfrp", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 0.1 * ones(neval), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(neval), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(neval), # static imbalance [m]
        "θ" => 15 * π / 180, # fiber angle global [rad]
    )

    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nevals))
    tipTwistData = zeros(length(nevals))
    meshlvl = 1
    for neval in nevals
        # --- Resize some stuff ---
        DVDict["neval"] = neval
        DVDict["c"] = 0.1 * ones(neval)
        DVDict["ab"] = 0 * ones(neval)
        DVDict["x_αb"] = 0 * ones(neval)
        # --- Solve ---
        costFuncs = SolveStatic.solve(DVDict, evalFuncs, "../OUTPUT")
        tipBendData[meshlvl] = costFuncs["w_tip"]
        tipTwistData[meshlvl] = costFuncs["psi_tip"]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticComp\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

    # --- Relative error ---
    rel_err1 = LinearAlgebra.norm(tipBendData - refBendSol, 2) / LinearAlgebra.norm(refBendSol, 2)
    rel_err2 = LinearAlgebra.norm(tipTwistData - refTwistSol, 2) / LinearAlgebra.norm(refTwistSol, 2)
    rel_err = max(rel_err1, rel_err2)

    # println("Relative error: ", rel_err1)
    # println("Relative error: ", rel_err2)

    return rel_err

end # test_SolveStaticComp

# ==============================================================================
#                         Test Forced Response
# ==============================================================================
function test_SolveForcedComp()
    """
    Very simple mesh convergence test with hydro and structural solvers over different numbers of nodes
    Composite beam
    """
    nevals = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    neval = nevals[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    fSweep = 0.01:0.1:10
    tipForceMag = 1.0
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 6.0, # free stream velocity [m/s]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "material" => "cfrp", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 0.1 * ones(neval), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(neval), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(neval), # static imbalance [m]
        "θ" => 15 * π / 180, # fiber angle global [rad]
    )


    # ==============================================================================
    #                         Call Forced Vibration Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nevals))
    tipTwistData = zeros(length(nevals))
    meshlvl = 1
    for neval in nevals
        # --- Resize some stuff ---
        DVDict["neval"] = neval
        DVDict["c"] = 0.1 * ones(neval)
        DVDict["ab"] = 0 * ones(neval)
        DVDict["x_αb"] = 0 * ones(neval)
        # --- Solve ---
        TipBendDyn, TipTwistDyn, LiftDyn, MomDyn = SolveForced.solve(DVDict, "../OUTPUT", fSweep, tipForceMag)
        tipBendData[meshlvl] = TipBendDyn[1]
        tipTwistData[meshlvl] = TipTwistDyn[1]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveForcedComp\n")
        write(io, "+---------------------------------------+\n")
        write(io, "f = 0 [Hz]")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

    # --- Relative error ---

end # end test_SolveForcedComp

# ==============================================================================
#                         Test Flutter Solver
# ==============================================================================

function test_correlationMatrix()
    """
    Test the correlation method of van Zyl between k increments
    """

    # --- k^(n) data ---
    R_old = [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
    # --- k^(n+1) data ---
    R_new = []

    old_r = real(R_old)
    old_i = imag(R_old)
    new_r = real(R_new)
    new_i = imag(R_new)
    SolverFlutter.compute_correlationMatrix(old_r, old_i, new_r, new_i)
end

function test_correlationMetrics()
    """
    Test the correlation method of van Zyl when new modes are found between dynP increments
    It uses the eigenvalues to help decide if a new mode is found
    """

    # --- q^(n) data ---
    p_old = []
    R_old = []
    # --- q^(n+1) data ---
    p_new = []
    R_new = []

    old_r = real(R_old)
    old_i = imag(R_old)
    new_r = real(R_new)
    new_i = imag(R_new)
    p_old_i = imag(p_old)
    p_new_i = imag(p_new)
    SolverFlutter.compute_correlationMetrics(old_r, old_i, new_r, new_i, p_old_i, p_new_i)
end