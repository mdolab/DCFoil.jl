"""
Test to verify that the static solve works without failing
"""

using Printf
using LinearAlgebra
using Dates

include("../src/solvers/SolveStatic.jl")
using .SolveStatic
include("../src/solvers/SolveForced.jl")
using .SolveForced
include("../src/solvers/SolveFlutter.jl")
using .SolveFlutter
include("../src/InitModel.jl")
using .InitModel
include("../src/struct/FiniteElements.jl")
using .FEMMethods
include("../src/solvers/SolverRoutines.jl")
using .SolverRoutines
include("../src/DCFoil.jl")
using .DCFoil

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
    refBendSol = [0.0000002020248011, 0.0000002058145797, 0.0000002072123383]
    refTwistSol = [0.0000000156876292, 0.0000000159326983, 0.0000000160164539]

    nNodess = [10, 20, 30] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    nNodes = nNodess[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => deg2rad(0.0), # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        "strut" => 0.4, # from Yingqian
    )
    solverOptions = Dict(
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "U∞" => 6.0, # free stream velocity [m/s]
        # --- I/O ---
        "name" => "akcabay",
        "debug" => false,
        "outputDir" => "test_out/",
        # --- General solver options ---
        "config" => "wing",
        # "config" => "t-foil",
        "nNodes" => nNodes, # number of nodes on foil half wing
        "nNodeStrut" => 10, # nodes on strut
        "use_tipMass" => false,
        "material" => "rigid", # preselect from material library
        "rotation" => 0.0, # deg
        # --------------------------------
        #   Flow
        # --------------------------------
        "use_cavitation" => false,
        "use_freesurface" => false,
        # --- Static solve ---
        "run_static" => true,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 0:0.1:10,
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 5,
        "uRange" => nothing,
        "config" => "wing",
    )
    mkpath(solverOptions["outputDir"])


    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nNodess))
    tipTwistData = zeros(length(nNodess))
    meshlvl = 1
    for nNodes in nNodess
        # --- Resize some stuff ---
        solverOptions["nNodes"] = nNodes
        DVDict["c"] = 0.1 * ones(nNodes)
        DVDict["ab"] = 0 * ones(nNodes)
        DVDict["x_αb"] = 0 * ones(nNodes)

        DCFoil.run_model(DVDict, evalFuncs; solverOptions)
        costFuncs = DCFoil.evalFuncs(evalFuncs, solverOptions)
        tipBendData[meshlvl] = costFuncs["wtip"]
        tipTwistData[meshlvl] = costFuncs["psitip"]
        println("cl:",costFuncs["cl"])
        println("cmy:",costFuncs["cmy"])
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = solverOptions["outputDir"] * "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticRigid\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nNodess[meshlvl], tip, tipTwist)
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

    return rel_err

end # test_SolveStaticRigid

# test_SolveStaticRigid()

function test_SolveStaticIso()
    """
    Very simple mesh convergence test with hydro and structural solvers over different numbers of nodes
    Stainless steel beam
    """
    # --- Reference value ---
    #  Obtained by running the code
    refBendSol = [0.00020270524592060, 0.0002058421525681, 0.0002069400251625]
    refTwistSol = [0.0000156996182303, 0.000015946939335, 0.00001602174580202]

    nNodess = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    nNodes = nNodess[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => 15 * π / 180, # fiber angle global [rad]
    )
    solverOptions = Dict(
        # --- I/O ---
        "name" => "akcabay",
        "nNodes" => nNodes,
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "U∞" => 6.0, # free stream velocity [m/s]
        "material" => "ss", # preselect from material library
        "debug" => false,
        "outputDir" => "test_out/",
        # --- General solver options ---
        "use_tipMass" => false,
        "use_cavitation" => false,
        "use_freesurface" => false,
        # --- Static solve ---
        "run_static" => true,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 0:0.1:10,
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 5,
        "uRange" => nothing,
        "config" => "wing",
    )
    mkpath(solverOptions["outputDir"])


    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nNodess))
    tipTwistData = zeros(length(nNodess))
    meshlvl = 1
    for nNodes in nNodess
        # --- Resize some stuff ---
        solverOptions["nNodes"] = nNodes
        DVDict["c"] = 0.1 * ones(nNodes)
        DVDict["ab"] = 0 * ones(nNodes)
        DVDict["x_αb"] = 0 * ones(nNodes)

        DCFoil.run_model(DVDict, evalFuncs; solverOptions)
        costFuncs = DCFoil.evalFuncs(evalFuncs, solverOptions)

        tipBendData[meshlvl] = costFuncs["wtip"]
        tipTwistData[meshlvl] = costFuncs["psitip"]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = solverOptions["outputDir"] * "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticIso\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nNodess[meshlvl], tip, tipTwist)
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
    refBendSol = [0.0004825455285840, 0.0004908633758853, 0.0004909057635612]
    refTwistSol = [-0.00079926586938542, -0.0008235901040122, -0.0008357865527998]

    nNodess = [10, 20, 40] # list of number of nodes to test
    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    nNodes = nNodess[1] # spatial nodes
    # --- Foil from Deniz Akcabay's 2020 paper ---
    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        "strut" => 0.4, # from Yingqian
    )
    solverOptions = Dict(
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "U∞" => 6.0, # free stream velocity [m/s]
        # --- I/O ---
        "name" => "akcabay",
        "debug" => false,
        "outputDir" => "test_out/",
        # --- General solver options ---
        "config" => "wing",
        # "config" => "t-foil",
        "nNodes" => nNodes, # number of nodes on foil half wing
        "nNodeStrut" => 10, # nodes on strut
        "use_tipMass" => false,
        "material" => "cfrp", # preselect from material library
        "rotation" => 0.0, # deg
        # --------------------------------
        #   Flow
        # --------------------------------
        "use_cavitation" => false,
        "use_freesurface" => false,
        # --- Static solve ---
        "run_static" => true,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 0:0.1:10,
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 5,
        "uRange" => nothing,
        "config" => "wing",
    )
    mkpath(solverOptions["outputDir"])

    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment"]

    # ==============================================================================
    #                         Call Static Solver
    # ==============================================================================
    # Call it for different mesh levels
    tipBendData = zeros(length(nNodess))
    tipTwistData = zeros(length(nNodess))
    meshlvl = 1
    for nNodes in nNodess
        # --- Resize some stuff ---
        solverOptions["nNodes"] = nNodes
        DVDict["c"] = 0.1 * ones(nNodes)
        DVDict["ab"] = 0 * ones(nNodes)
        DVDict["x_αb"] = 0 * ones(nNodes)

        DCFoil.run_model(DVDict, evalFuncs; solverOptions=solverOptions)
        costFuncs = DCFoil.evalFuncs(evalFuncs, solverOptions)

        tipBendData[meshlvl] = costFuncs["wtip"]
        tipTwistData[meshlvl] = costFuncs["psitip"]
        meshlvl += 1
    end

    meshlvl = 1

    # ************************************************
    #     Write results to test output file
    # ************************************************
    # NOTE: this assumes you're running the test from the run_tests.jl file
    fname = solverOptions["outputDir"] * "test.out"
    open(fname, "a") do io
        write(io, "+---------------------------------------+\n")
        write(io, "|    test_SolveStaticComp\n")
        write(io, "+---------------------------------------+\n")
        write(io, "  meshlvl   | tip bend [m] | tip twist [rad] |\n")
        for tip in tipBendData
            tipTwist = tipTwistData[meshlvl]
            line = @sprintf("%i (%i nodes)   %.16f    %.16f\n", meshlvl, nNodess[meshlvl], tip, tipTwist)
            write(io, line)
            meshlvl += 1
        end
    end

    # --- Relative error ---
    rel_err1 = LinearAlgebra.norm(tipBendData - refBendSol, 2) / LinearAlgebra.norm(refBendSol, 2)
    rel_err2 = LinearAlgebra.norm(tipTwistData - refTwistSol, 2) / LinearAlgebra.norm(refTwistSol, 2)
    rel_err = max(rel_err1, rel_err2)

    return rel_err

end # test_SolveStaticComp

# rel_err = test_SolveStaticComp()

# ==============================================================================
#                         Test Forced Response
# ==============================================================================
function test_SolveForcedComp()
    """
    Composite beam
    """

end # end test_SolveForcedComp

# test_SolveForcedComp()

# ==============================================================================
#                         Test Flutter Solver
# ==============================================================================
function test_modal()
    """
    Test modal solver
    NOTE: only testing the frequencies bc it is assumed the eigenvector coming out is right
    """
    # ************************************************
    #     Reference solution
    # ************************************************
    refDryFreqs = [74.848, 157.680, 469.063, 613.743, 1313.393]
    refWetFreqs = [19.142, 62.571, 119.878, 243.393, 335.450]

    # ************************************************
    #     Computed solution
    # ************************************************
    nNodes = 40 # spatial nodes

    # --- Yingqian's Viscous FSI Paper (2019) ---
    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "c" => 0.0925 * ones(nNodes), # chord length [m]
        "s" => 0.2438, # semispan [m]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.03459, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(0), # fiber angle global [rad]
    )
    solverOptions = Dict(
        # --- I/O ---
        "debug" => false,
        "outputDir" => "",
        # --- General solver options ---
        "nNodes" => nNodes,
        "use_tipMass" => false,
        "use_cavitation" => false,
        "use_freesurface" => false,
        "material" => "cfrp", # preselect from material library
        "U∞" => 5.0, # free stream velocity [m/s]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        # --- Static solve ---
        "run_static" => false,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 0:0.1:10,
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => true,
        "run_flutter" => false,
        "nModes" => 5,
        "uRange" => nothing,
    )
    # --- Mesh ---
    FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)
    nElem = nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])
    structNatFreqs, _, wetNatFreqs, _ = SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)

    # ************************************************
    #     Relative error
    # ************************************************
    rel_err1 = LinearAlgebra.norm(structNatFreqs - refDryFreqs, 2) / LinearAlgebra.norm(refDryFreqs, 2)
    rel_err2 = LinearAlgebra.norm(wetNatFreqs - refWetFreqs, 2) / LinearAlgebra.norm(refWetFreqs, 2)
    rel_err = max(rel_err1, rel_err2)

    return rel_err
end

function test_eigensolve()
    """
    Test a simple eigenvalue solver
    """
    # --- A test matrix ---
    A_r = [2.0 7.0; 1.0 8.0]
    A_i = [1.0 1.0; 1.0 1.0]
    n = 2
    y = SolverRoutines.cmplxStdEigValProb2(A_r, A_i, n)
    w_r1 = y[1:n]
    w_i1 = y[n+1:2*n]
    VR_r1 = reshape(y[2*n+1:2*n+n^2], n, n)
    VR_i1 = reshape(y[2*n+n^2+1:end], n, n)
    w_r, w_i, _, _, VR_r, VR_i = SolverRoutines.cmplxStdEigValProb(A_r, A_i, 2)

    err = LinearAlgebra.norm(w_r1 - w_r, 2) + LinearAlgebra.norm(w_i1 - w_i, 2)
end

function test_pk_staticDiv()
    """
    Test flutter solver for the static divergence case
    """

    ref_val = 0.2346166

    nNodes = 10
    df = 1
    fSweep = 0.1:df:1000.0 # forcing and search frequency sweep [Hz]
    uRange = [20.0, 40.0] # flow speed [m/s] sweep for flutter
    tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => deg2rad(0.0), # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(-15), # fiber angle global [rad]
        "strut" => 0.4, # from Yingqian
    )

    solverOptions = Dict(
        # --- I/O ---
        "name" => "akcabay",
        "debug" => true,
        # --- General solver options ---
        "config" => "wing",
        "nNodes" => nNodes,
        "nNodeStrut" => 10,
        "U∞" => 5.0, # free stream velocity [m/s]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "rotation" => 0.0, # deg
        "material" => "cfrp", # preselect from material library
        "gravityVector" => [0.0, 0.0, -9.81],
        "use_tipMass" => false,
        "use_freeSurface" => false,
        "use_cavitation" => false,
        "use_ventilation" => false,
        # --- Static solve ---
        "run_static" => false,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => fSweep,
        "tipForceMag" => tipForceMag,
        # --- Eigen solve ---
        "run_modal" => true,
        "run_flutter" => true,
        "nModes" => 4,
        "uRange" => uRange,
        "maxQIter" => 4000,
        "rhoKS" => 80.0,
    )
    evalFuncs = ["ksflutter"]

    outputDir = @sprintf("./OUTPUT/%s_%s_%s_f%.1f_w%.1f/",
        string(Dates.today()),
        solverOptions["name"],
        solverOptions["material"],
        rad2deg(DVDict["θ"]),
        rad2deg(DVDict["Λ"]))
    mkpath(outputDir)

    solverOptions["outputDir"] = outputDir

    DCFoil.run_model(DVDict, evalFuncs; solverOptions=solverOptions)
    costFuncs = DCFoil.evalFuncs(evalFuncs, solverOptions)

    err = (ref_val - costFuncs["ksflutter"]) / ref_val
    return err
end

# test_pk_staticDiv()

function test_pk_flutter()
    """
    Test flutter solver for the fluttering case
    """
    ref_val = 0.05347
    nNodes = 10
    df = 1
    fSweep = 0.1:df:1000.0 # forcing and search frequency sweep [Hz]
    uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
    tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

    DVDict = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => deg2rad(-15.0), # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        "strut" => 0.4, # from Yingqian
    )

    solverOptions = Dict(
        # --- I/O ---
        "name" => "akcabay",
        "debug" => true,
        # --- General solver options ---
        "config" => "wing",
        "nNodes" => nNodes,
        "nNodeStrut" => 10,
        "U∞" => 5.0, # free stream velocity [m/s]
        "ρ_f" => 1000.0, # fluid density [kg/m³]
        "rotation" => 0.0, # deg
        "material" => "cfrp", # preselect from material library
        "gravityVector" => [0.0, 0.0, -9.81],
        "use_tipMass" => false,
        "use_freeSurface" => false,
        "use_cavitation" => false,
        "use_ventilation" => false,
        # --- Static solve ---
        "run_static" => false,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => fSweep,
        "tipForceMag" => tipForceMag,
        # --- Eigen solve ---
        "run_modal" => true,
        "run_flutter" => true,
        "nModes" => 4,
        "uRange" => uRange,
        "maxQIter" => 4000,
        "rhoKS" => 80.0,
    )
    evalFuncs = ["ksflutter"]

    outputDir = @sprintf("./OUTPUT/%s_%s_%s_f%.1f_w%.1f/",
        string(Dates.today()),
        solverOptions["name"],
        solverOptions["material"],
        rad2deg(DVDict["θ"]),
        rad2deg(DVDict["Λ"]))
    mkpath(outputDir)

    solverOptions["outputDir"] = outputDir

    DCFoil.run_model(DVDict, evalFuncs; solverOptions=solverOptions)
    costFuncs = DCFoil.evalFuncs(evalFuncs, solverOptions)
    err = (ref_val - costFuncs["ksflutter"]) / ref_val

    return err
end

# test_pk_flutter()