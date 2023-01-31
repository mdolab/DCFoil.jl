"""
Test to verify that the static solve works without failing
"""

using Printf

include("../src/solvers/SolveStatic.jl")
using .SolveStatic


function test_SolveStaticRigid()
    """
    Very simple test with hydro and structural solvers over different numbers of nodes
    Rigid beam
    """

    nevals = [10, 20, 40, 80] # list of number of nodes to test
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
            line = @sprintf("%i (%i nodes)   %.6f    %.6f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

end

function test_SolveStaticIso()
    """
    Very simple test with hydro and structural solvers over different numbers of nodes
    Stainless steel beam
    """

    nevals = [10, 20, 40, 80] # list of number of nodes to test
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
            line = @sprintf("%i (%i nodes)   %.6f    %.6f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

end

function test_SolveStaticComp()
    """
    Very simple test with hydro and structural solvers over different numbers of nodes
    Composite beam
    """

    nevals = [10, 20, 40, 80] # list of number of nodes to test
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
            line = @sprintf("%i (%i nodes)   %.6f    %.6f\n", meshlvl, nevals[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

end