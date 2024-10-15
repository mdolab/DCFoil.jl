"""
Test derivative routines with super basic tests
"""

using FiniteDifferences, Zygote, Enzyme
using Plots
using Printf
using LinearAlgebra
using JLD2

include("../src/DCFoil.jl")
using .DCFoil: DCFoil, SolverRoutines, HydroStrip, SolveFlutter, SolveStatic, FEMMethods
# ==============================================================================
#                         Aero-node tests
# ==============================================================================
function test_hydromass()

    # Test values
    rho_f = 1000 # kg/m^3 FW
    b = 2.0
    ab = 3.0

    derivs = Zygote.jacobian((x1, x2) -> HydroStrip.compute_node_mass(x1, x2, rho_f),
        b, ab)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> HydroStrip.compute_node_mass(x, ab, rho_f),
        b)

    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> HydroStrip.compute_node_mass(b, x, rho_f),
        ab)

    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2

    # Get norms
    test1 = norm(test1, 2)
    test2 = norm(test2, 2)
    return min(test1, test2)
end

function test_hydrodamp()
    # Test values
    rho_f = 1000 # kg/m^3 FW
    clalfa = 6.0
    b = 2.0
    eb = 0.5
    ab = 3.0
    k = 0.1
    Cklist = HydroStrip.compute_theodorsen(k)
    Ck = Cklist[1] + Cklist[2] * im
    U∞ = 10.0
    Λ = 0.0

    function my_compute_node_damp(clα, b, eb, ab, U∞, Λ, rho_f, Ck)
        Cf, Cfhat = HydroStrip.compute_node_damp(clα, b, eb, ab, U∞, Λ, rho_f, Ck)

        return imag(Cf)
    end

    derivs = Zygote.jacobian((x1, x2, x3, x4) -> my_compute_node_damp(x1, b, x2, x3, U∞, Λ, rho_f, x4),
        clalfa, eb, ab, Ck)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_node_damp(x, b, eb, ab, U∞, Λ, rho_f, Ck),
        clalfa)

    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_node_damp(clalfa, b, x, ab, U∞, Λ, rho_f, Ck),
        eb)

    fdderivs3, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_node_damp(clalfa, b, eb, x, U∞, Λ, rho_f, Ck),
        ab)

    fdderivs4, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_node_damp(clalfa, b, eb, ab, U∞, Λ, rho_f, x),
        Ck)

    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2
    test3 = derivs[3] - fdderivs3

    return max(norm(test1, 2), norm(test2, 2), norm(test3, 2))
end

function test_interp()
    """Test the my linear interpolation"""
    mesh = collect(0:0.1:2)
    yVec, _, _ = HydroStrip.compute_glauert_circ(mesh[end], ones(length(mesh)), deg2rad(1), 1.0)
    xq = 0.5

    derivs = Zygote.jacobian((x1, x2, x3) -> SolverRoutines.do_linear_interp(x1, x2, x3),
        mesh, yVec, xq)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(x, yVec, xq),
        mesh)
    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(mesh, x, xq),
        yVec)
    fdderivs3, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(mesh, yVec, x),
        xq)

    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2
    test3 = derivs[3] - fdderivs3

    return max(norm(test1, 2), norm(test2, 2), norm(test3, 2))
end

function test_hydroderiv(DVDict, solverOptions)
    """
    Test the assembly of hydro matrices
    """

    nElem = solverOptions["nNodes"] - 1
    mesh, elemConn = FEMMethods.make_componentMesh(nElem, DVDict["s"])
    mesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, _, DOFBlankingList, _, nModes, _, _ = SolveFlutter.setup_solver(
        DVDict["alfa0"], DVDict["sweep"], DVDict["s"], DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_ab"], DVDict["zeta"], DVDict["theta_f"], solverOptions
    )
    FEMESH = FEMMethods.StructMesh(structMesh, elemConn, chordVec, toc, abVec, x_αbVec, theta_f, zeros(2, 2))
    globalKs, _, _ = FEMMethods.assemble(FEMESH, abVec, x_αbVec, FOIL, "BT2", "orthotropic")

    dim = size(globalKs, 1) # big problem
    ω = 0.1
    b = 1.0
    U∞ = 1.0

    function my_compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω)
        """Simple wrapper"""

        Mf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω, "BT2")

        # Select the fluid matrix you want to verify derivatives for
        return Mf
        # return globalCf_r
    end

    function new_compute_AICs(Mf, dim, x1, x2, x3, x4, x5, FOIL, U∞, ω)
        Mf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω, "BT2")
    end


    # --- AD ---
    derivs = Zygote.jacobian((x1, x2, x3, x4, x5) -> my_compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω),
        mesh, Λ, chordVec, abVec, ebVec)

    Enzyme.jacobian(Reverse, my_compute_AICs, dim, mesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω)

    # --- FD ---
    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_AICs(dim, x, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω),
        mesh) # good
    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_AICs(dim, mesh, x, chordVec, abVec, ebVec, FOIL, U∞, ω),
        Λ) # good
    fdderivs3, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_AICs(dim, mesh, Λ, x, abVec, ebVec, FOIL, U∞, ω),
        chordVec) # not good
    fdderivs4, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_AICs(dim, mesh, Λ, chordVec, x, ebVec, FOIL, U∞, ω),
        abVec) # not good
    fdderivs5, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> my_compute_AICs(dim, mesh, Λ, chordVec, abVec, x, FOIL, U∞, ω),
        ebVec)

    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2
    test3 = derivs[3] - fdderivs3
    test4 = derivs[4] - fdderivs4
    test5 = derivs[5] - fdderivs5

    # Stack derivs
    return max(norm(test1, 2), norm(test2, 2), norm(test3, 2), norm(test4, 2), norm(test5, 2))
    # return derivs, fdderivs1, fdderivs2, fdderivs3, fdderivs4, fdderivs5
end

function test_eigenvalueAD()
    """
    Dot product test!
    """

    # --- A test matrix ---
    A_r = [2.0 7.0; 1.0 8.0]
    A_i = [1.0 1.0; 0.0 1.0]

    # ---------------------------
    #   forward AD
    # ---------------------------
    dim = 2
    A_rd = zeros(Float64, dim, dim)
    A_id = zeros(Float64, dim, dim)
    # A_rd[2, 1] = 1.0
    A_rd .= 1.0 # poke all entries in matrix in forward
    # TODO: this is not working for imaginary seed
    # A_id .= 1.0
    # A_id[1, 1] = 1.0
    w_r, w_rd, w_i, w_id, VR_r, VR_rd, VR_i, VR_id = SolverRoutines.cmplxStdEigValProb_d(A_r, A_rd, A_i, A_id, dim)
    println("Primal forward values:")
    println("----------------------")
    println("w_r = ", w_r)
    println("w_i = ", w_i)
    println("VR_r")
    show(stdout, "text/plain", VR_r)
    println("")
    println("VR_i")
    show(stdout, "text/plain", VR_i)
    println("")
    println("Dual forward values:")
    println("--------------------")
    println("w_rd = ", w_rd)
    println("w_id = ", w_id)
    println("VR_rd")
    show(stdout, "text/plain", VR_rd)
    println("")
    println("VR_id")
    show(stdout, "text/plain", VR_id)
    println("")
    # ---------------------------
    #   backward AD
    # ---------------------------
    w_rb = zeros(Float64, dim)
    w_ib = zeros(Float64, dim)
    w_rb = [1, 1] # poke both eigenvalues in reverse
    w_rb = w_rd
    w_ib = [1, 1]
    Vrb_r = zeros(Float64, dim, dim)
    Vrb_i = zeros(Float64, dim, dim)
    A_rb, A_ib, w_r, w_rbz, w_i, w_ibz, _, _, _, _ = SolverRoutines.cmplxStdEigValProb_b(A_r, A_i, dim, w_rb, w_ib, Vrb_r, Vrb_i)
    println("Primal reverse values:")
    println("w_r = ", w_r)
    println("w_i = ", w_i)
    # println("VR_r", VR_r)
    # println("VR_i", VR_i)
    println("Dual reverse values:")
    # println("wb_r = ", w_rb)
    # println("wb_i = ", w_ib)
    println("A_rb", A_rb)
    println("A_ib", A_ib)

    # ---------------------------
    #   Dot product test
    # ---------------------------
    # --- Outputs ---
    ḟ = w_rd
    f̄ = w_rb
    # --- Inputs ---
    # The inputs were matrices so we just unroll them
    ẋ = vec(A_rd)
    x̄ = vec(A_rb)
    # --- Dot product ---
    lhs = (transpose(ẋ) * x̄)
    rhs = (transpose(ḟ) * f̄)
    # These should be equal if you did it right
    return lhs - rhs
end

function test_theodorsenDeriv()
    """
    Test the theodorsen functions
    """
    nNodes = 3 # Number of spatial nodes
    chordVec = vcat(LinRange(0.81, 0.405, nNodes))
    # ---------------------------
    #   Test glauert lift distribution
    # ---------------------------
    cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=1.0)
    pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
    plot!(title="lift slope")

    # ---------------------------
    #   Test 𝙲(k)
    # ---------------------------
    k0 = 1e-12
    kSweep = k0:0.001:2

    datar = []
    datai = []
    datar2 = []
    datai2 = []
    datar3 = []
    datai3 = []
    dADr = []
    dADi = []
    dFDr = []
    dFDi = []
    dRD_r = []
    dRD_i = []
    for k ∈ kSweep
        datum = HydroStrip.compute_theodorsen(k)
        datum2 = HydroStrip.compute_pade(k)
        datum3 = HydroStrip.compute_fraccalc(k)
        push!(datar, datum[1])
        push!(datai, datum[2])
        push!(datar2, datum2[1])
        push!(datai2, datum2[2])
        push!(datar3, datum3[1])
        push!(datai3, datum3[2])
        # derivAD = ForwardDiff.derivative(HydroStrip.compute_theodorsen, k)
        # derivFD = FiniteDifferences.forward_fdm(2, 1)(HydroStrip.compute_theodorsen, k)
        derivAD, = Zygote.jacobian(HydroStrip.compute_pade, k)
        derivFD, = Zygote.jacobian(HydroStrip.compute_fraccalc, k)
        derivRD, = Zygote.jacobian(HydroStrip.compute_theodorsen, k)
        push!(dADr, derivAD[1])
        push!(dADi, derivAD[2])
        push!(dFDr, derivFD[1])
        push!(dFDi, derivFD[2])
        push!(dRD_r, derivRD[1])
        push!(dRD_i, derivRD[2])
    end

    # # --- Derivatives ---
    # dADr
    # println("Forward AD:", ForwardDiff.derivative(HydroStrip.compute_theodorsen, 0.1))
    # println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(HydroStrip.compute_theodorsen, 0.1))
    # println("Reverse AD:", Zygote.jacobian(HydroStrip.compute_theodorsen, 0.1))

    # ************************************************
    #     Plot data
    # ************************************************
    lw = 2.0
    la = 0.5
    # --- Plot primal ---
    # Solid lines
    p1 = plot(kSweep, datar, label="Re", tick_dir=:out, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, datai, label="Im", color=:blue, linewidth=lw, linealpha=la)
    # Dashed lines
    plot!(kSweep, datar2, label="Re Pade-3", line=:dash, color=:red, linewidth=lw, linealpha=la, tick_dir=:out)
    plot!(kSweep, datai2, label="Im Pade-3", line=:dash, color=:blue, linewidth=lw, linealpha=la)
    # Dotted lines
    plot!(kSweep, datar3, label="Re FracCalc", line=:dot, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, datai3, label="Im FracCalc", line=:dot, color=:blue, linewidth=lw, linealpha=la)

    plot!(title="Theodorsen function")
    plot!(xlabel="k", ylabel="C(k)")

    # --- Plot derivatives ---
    p2 = plot(kSweep, dRD_r, label="Real RD", color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, dRD_i, label="Imag RD", color=:blue, linewidth=lw, linealpha=la)

    # p2 = plot(kSweep, dADr, label="Re FAD", tick_dir=:out, color=:red, linewidth=lw, linealpha=la)
    # plot!(kSweep, dADi, label="Im FAD", color=:blue, linewidth=lw, linealpha=la)
    plot!(kSweep, dADr, label="Re Pade-3", tick_dir=:out, color=:red, linewidth=lw, linealpha=la, line=:dash)
    plot!(kSweep, dADi, label="Im Pade-3", color=:blue, linewidth=lw, linealpha=la, line=:dash)

    plot!(kSweep, dFDr, label="Re FracCalc", line=:dot, color=:red, linewidth=lw, linealpha=la)
    plot!(kSweep, dFDi, label="Im FracCalc", line=:dot, color=:blue, linewidth=lw, linealpha=la)


    ylims!(-10, 2.0)
    xlims!(-0.1, 1.1)
    plot!(title="RAD wrt k")
    xlabel!("k")
    ylabel!("partial C(k)/ partial k")
    plot(p1, p2)

    savefig("theodorsen.png")
end

function test_staticDeriv(DVDict, solverOptions, wingOptions)
    """
    Unit test for derivative components
    """
    # ************************************************
    #     Task type
    # ************************************************
    # Set task you want to true
    # Defaults
    run = true # run the solver for a single point
    run_static = false
    run_forced = false
    run_modal = false
    run_flutter = false
    debug = false
    tipMass = false

    # Uncomment here
    run_static = true
    debug = false



    # ************************************************
    #     I/O
    # ************************************************
    # The file directory has the convention:
    # <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
    # But we write the DVDict to a human readable file in the directory anyway so you can double check
    outputDir = @sprintf("./test_out/%s_%s_f%.1f_w%.1f/",
        solverOptions["name"],
        wingOptions["material"],
        rad2deg(DVDict["theta_f"]),
        rad2deg(DVDict["sweep"]))
    mkpath(outputDir)
    # ************************************************
    #     Cost functions
    # ************************************************
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]
    evalFuncsSensList = ["lift"]

    solverOptions["outputDir"] = outputDir

    # ==============================================================================
    #                         Call DCFoil
    # ==============================================================================
    steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16] # step sizes
    # dvKey = "theta_f" # dv to test deriv
    # dvKey = "sweep" # dv to test deriv
    dvKey = "rake" # dv to test deriv
    # dvKey = "alfa0" # dv to test deriv
    # dvKey = "toc" # dv to test deriv
    # evalFunc = "ksflutter"
    evalFunc = "lift"
    # evalFunc = "moment"

    # ==============================================================================
    #                         STATIC DERIV TESTS
    # ==============================================================================
    derivs = zeros(length(steps))
    funcVal = 0.0

    # ************************************************
    #     FULL TEST
    # ************************************************
    DCFoil.init_model([DVDict], evalFuncs; solverOptions=solverOptions)
    SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
    costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
    funcsSensAdjoint = DCFoil.evalFuncsSens(SOL, [DVDict], evalFuncsSensList, solverOptions; mode="Adjoint")
    adjointDerivative = funcsSensAdjoint[1][dvKey]
    for (ii, dh) in enumerate(steps)
        DCFoil.init_model([DVDict], evalFuncs; solverOptions=solverOptions)
        SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
        costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
        stat_i = costFuncs[evalFuncsSensList[1]*"-"*wingOptions["compName"]]
        funcVal = stat_i
        DVDict[dvKey] += dh
        SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
        costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
        stat_f = costFuncs[evalFuncsSensList[1]*"-"*wingOptions["compName"]]

        derivs[ii] = (stat_f - stat_i) / dh

        @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

        # --- Reset DV ---
        DVDict[dvKey] -= dh
    end

    diff = zeros(length(steps))
    for (ii, dfdx) in enumerate(derivs)
        diff[ii] = only(adjointDerivative) - dfdx
    end

    # find out how many are below a tolerance
    tol = 7e-2
    numBelowTol = sum(abs.(diff) .< tol)

    return numBelowTol
end

function test_staticdrdu(DVDict, solverOptions, wingOptions)
    # ==============================================================================
    #                         Also test dr du
    # ==============================================================================
    # this should actually be a separate test function but I'm lazy
    # ************************************************
    #     Complex step derivative
    # ************************************************
    DVDictList = [DVDict, DVDict]
    _, _, solverParams = DCFoil.SolveStatic.setup_problem(DVDictList, wingOptions, solverOptions)

    mode = "CS"
    structStates = zeros(size(solverParams.Kmat, 1) - length(solverParams.dofBlank))
    @time jacobianCS = DCFoil.SolveStatic.compute_∂r∂u(structStates, mode; DVDictList=DVDictList, solverParams=solverParams, appendageOptions=wingOptions, solverOptions=solverOptions)

    mode = "RAD"
    @time jacobianAD = real(DCFoil.SolveStatic.compute_∂r∂u(structStates, mode; DVDictList=DVDictList, solverParams=solverParams, appendageOptions=wingOptions, solverOptions=solverOptions))

    # --- Compare ---
    difference = jacobianCS .- jacobianAD

    return maximum(abs.(difference))
end

function test_pkflutterderiv(DVDict, solverOptions)
    """
    Test AD derivative of the pk flutter analysis with 
    KS aggregation against finite differences

    Also some profiling
    """

    @time SolveFlutter.get_sol(DVDict, solverOptions)
    @time funcsSensAD = SolveFlutter.evalFuncsSens(["ksflutter"], DVDict, solverOptions; mode="RAD")
    @time funcsSensFD = SolveFlutter.evalFuncsSens(["ksflutter"], DVDict, solverOptions; mode="FiDi")


    # Print it out
    for (key, val) in funcsSensAD
        save("RAD" * key * ".jld2", "derivs", funcsSensAD)
        println("AD: ", key, " = ", val)
        save("FWDDiff" * key * ".jld2", "derivs", funcsSensFD)
        println("FD: ", key, " = ", funcsSensFD[key])

        if val !== nothing
            if maximum(funcsSensFD[key] - val) > 1e-3
                println("Possibly bad derivative: ", key)
                println("Abs difference btwn FD and AD: ", funcsSensFD[key] - val)
                println("Check FD step size")
            end
        end
    end

    return 0.0

end

function test_pkprofile(DVDict, solverOptions)
    """
    Profile pk flutter deriv
    """
    # SolveFlutter.compute_costFuncs(DVDict, solverOptions)
    funcsSensAD = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode="RAD")
end
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
nNodes = 4
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "sweep" => deg2rad(-15.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
    # --- Strut vars ---
    "rake" => 0.0,
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.14 * ones(nNodes), # chord length [m]
    "toc_strut" => 0.095, # thickness-to-chord ratio (mean)
    "ab_strut" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "x_ab_strut" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)

appendageDict = Dict(
    "nNodes" => nNodes,
    "config" => "wing",
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_tipMass" => false,
    "material" => "cfrp", # preselect from material library
    "config" => "wing",
)
solverOptions = Dict(
    # --- I/O ---
    "name" => "test",
    "debug" => false,
    "outputDir" => "./test_out/",
    # --- General solver options ---
    "Uinf" => 5.0, # free stream velocity [m/s]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    "appendageList" => [appendageDict],
    # --- Static solve ---
    "run_static" => false,
    # --- Forced solve ---
    "run_forced" => false,
    "fRange" => [0.0, 1.0],
    "tipForceMag" => 0.5 * 0.5 * 1000 * 100 * 0.03,
    # --- Eigen solve ---
    "run_modal" => false,
    "run_flutter" => true,
    "nModes" => 4,
    "uRange" => [185, 190.0],
    "maxQIter" => 100,
    "rhoKS" => 100.0,
)

# Profiling code
using BenchmarkTools
using TimerOutputs
using Profile

# test_pkflutterderiv(DVDict, solverOptions) # primal
# @profview test_pkprofile(DVDict, solverOptions) #deriv
# @benchmark test_pkprofile(DVDict, solverOptions) #benchmark
# Zygote.@profile test_pkprofile(DVDict, solverOptions) # this doesn't work
# Profile.clear() # run to clear compilation stuff

# answer = test_staticDeriv(nothing, nothing)