"""
Test derivative routines with super basic tests
"""

include("../src/solvers/SolverRoutines.jl")
using .SolverRoutines
include("../src/hydro/Hydro.jl")
include("../src/InitModel.jl")
include("../src/struct/FiniteElements.jl")
include("../src/solvers/SolveFlutter.jl")
using .Hydro, .InitModel, .FEMMethods, .SolveFlutter
using FiniteDifferences, ForwardDiff, Zygote
using Plots, LaTeXStrings, Printf, LinearAlgebra

# ==============================================================================
#                         Aero-node tests
# ==============================================================================
function test_hydromass()

    # Test values
    rho_f = 1000 # kg/m^3 FW
    b = 2.0
    ab = 3.0

    derivs = Zygote.jacobian((x1, x2) -> Hydro.compute_node_mass(x1, x2, rho_f),
        b, ab)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> Hydro.compute_node_mass(x, ab, rho_f),
        b)

    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> Hydro.compute_node_mass(b, x, rho_f),
        ab)

    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2

    # Get norms
    test1 = norm(test1, 2)
    test2 = norm(test2, 2)
    return min(test1, test2)
end # end function

function test_hydrodamp()
    # Test values
    rho_f = 1000 # kg/m^3 FW
    clalfa = 6.0
    b = 2.0
    eb = 0.5
    ab = 3.0
    k = 0.1
    Cklist = Hydro.compute_theodorsen(k)
    Ck = Cklist[1] + Cklist[2] * im
    U∞ = 10.0
    Λ = 0.0

    function my_compute_node_damp(clα, b, eb, ab, U∞, Λ, rho_f, Ck)
        Cf, Cfhat = Hydro.compute_node_damp(clα, b, eb, ab, U∞, Λ, rho_f, Ck)

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
end # end function

function test_interp()
    """Test the my linear interpolation"""
    mesh = collect(0:0.1:2)
    yVec = Hydro.compute_glauert_circ(mesh[end], ones(length(mesh)), deg2rad(1), 1.0, length(mesh))
    xq = 0.5

    derivs = Zygote.jacobian((x1, x2, x3) -> SolverRoutines.do_linear_interp(x1, x2, x3),
        mesh, yVec, xq)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(x, yVec, xq),
        mesh)
    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(mesh, x, xq),
        yVec)
    fdderivs3, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolverRoutines.do_linear_interp(mesh, yVec, x),
        xq)

    return max(norm(test1, 2), norm(test2, 2), norm(test3, 2))
end

function test_hydroderiv(DVDict, solverOptions)
    """
    Test the assembly of hydro matrices
    """

    nElem = solverOptions["nNodes"] - 1
    mesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])
    _, _, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, _, DOFBlankingList, _, nModes, _, _ = SolveFlutter.setup_solver(mesh, elemConn, DVDict, solverOptions)
    globalKs, _, _ = FEMMethods.assemble(mesh, elemConn, abVec, x_αbVec, FOIL, "BT2", FOIL.constitutive)

    dim = size(globalKs, 1) # big problem
    ω = 0.1
    b = 1.0
    U∞ = 1.0

    function my_compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω)
        """Simple wrapper"""

        Mf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω, "BT2")

        # Select the fluid matrix you want to verify derivatives for
        return Mf
        # return globalCf_r
    end


    # --- AD ---
    derivs = Zygote.jacobian((x1, x2, x3, x4, x5) -> my_compute_AICs(dim, x1, x2, x3, x4, x5, FOIL, U∞, ω),
        mesh, Λ, chordVec, abVec, ebVec)

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
    cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=1.0, nNodes=nNodes)
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
        datum = Hydro.compute_theodorsen(k)
        datum2 = Hydro.compute_pade(k)
        datum3 = Hydro.compute_fraccalc(k)
        push!(datar, datum[1])
        push!(datai, datum[2])
        push!(datar2, datum2[1])
        push!(datai2, datum2[2])
        push!(datar3, datum3[1])
        push!(datai3, datum3[2])
        # derivAD = ForwardDiff.derivative(Hydro.compute_theodorsen, k)
        # derivFD = FiniteDifferences.forward_fdm(2, 1)(Hydro.compute_theodorsen, k)
        derivAD, = Zygote.jacobian(Hydro.compute_pade, k)
        derivFD, = Zygote.jacobian(Hydro.compute_fraccalc, k)
        derivRD, = Zygote.jacobian(Hydro.compute_theodorsen, k)
        push!(dADr, derivAD[1])
        push!(dADi, derivAD[2])
        push!(dFDr, derivFD[1])
        push!(dFDi, derivFD[2])
        push!(dRD_r, derivRD[1])
        push!(dRD_i, derivRD[2])
    end

    # # --- Derivatives ---
    # dADr
    # println("Forward AD:", ForwardDiff.derivative(Hydro.compute_theodorsen, 0.1))
    # println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(Hydro.compute_theodorsen, 0.1))
    # println("Reverse AD:", Zygote.jacobian(Hydro.compute_theodorsen, 0.1))

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
    plot!(xlabel=L"k", ylabel=L"C(k)")

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
    xlabel!(L"k")
    ylabel!(L"\partial C(k)/ \partial k")
    plot(p1, p2)

    savefig("theodorsen.png")
end

function test_pkflutterderiv(DVDict, solverOptions)
    """
    # TODO: have common inputs and outputs to this test function
    Test AD derivative of the pk flutter analysis with 
    KS aggregation against finite differences

    The only DV tested is chord ref right now
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
    # run_forced = true
    run_modal = true
    run_flutter = true
    debug = false
    # tipMass = true

    # ************************************************
    #     DV Dictionaries (see INPUT directory)
    # ************************************************
    nNodes = 5 # spatial nodes
    nModes = 4 # number of modes to solve for;
    # NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
    # This is because poles bifurcate
    # nModes is really the starting number of structural modes you want to solve for
    fSweep = range(0.1, 1000.0, 1000) # forcing and search frequency sweep [Hz]
    # uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
    uRange = [187.0, 190.0] # flow speed [m/s] sweep for flutter
    tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing


    FOIL = InitModel.init_dynamic(DVDict, solverOptions; uRange=solverOptions["uRange"], fSweep=solverOptions["fSweep"])
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"]; config=solverOptions["config"])
    outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
        solverOptions["name"],
        solverOptions["material"],
        rad2deg(DVDict["θ"]),
        rad2deg(DVDict["Λ"]))
    solverOptions["outputDir"] = outputDir



    uRange, b_ref, chordVec, abVec, _, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug = SolveFlutter.setup_solver(structMesh, elemConn, DVDict, solverOptions)


    derivs = Zygote.jacobian((x1, x2, x3, x4, x5, x6) -> SolveFlutter.solve(
            x1, solverOptions, uRange, x2, x3, x4, x5, x6, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        structMesh, b_ref, chordVec, abVec, ebVec, Λ)

    fdderivs1, = FiniteDifferences.jacobian(central_fdm(3, 1), (x1) -> SolveFlutter.solve(
            x1, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        structMesh) # good

    fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x2) -> SolveFlutter.solve(
            structMesh, solverOptions, uRange, x2, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        b_ref)  # good

    fdderivs3, = FiniteDifferences.jacobian(central_fdm(3, 1), (x3) -> SolveFlutter.solve(
            structMesh, solverOptions, uRange, b_ref, x3, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        chordVec) # good


    fdderivs4, = FiniteDifferences.jacobian(central_fdm(3, 1), (x4) -> SolveFlutter.solve(
            structMesh, solverOptions, uRange, b_ref, chordVec, x4, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        abVec) # good


    fdderivs5, = FiniteDifferences.jacobian(central_fdm(3, 1), (x5) -> SolveFlutter.solve(
            structMesh, solverOptions, uRange, b_ref, chordVec, abVec, x5, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        ebVec) # good


    fdderivs6, = FiniteDifferences.jacobian(central_fdm(3, 1), (x6) -> SolveFlutter.solve(
            structMesh, solverOptions, uRange, b_ref, chordVec, abVec, ebVec, x6, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
        Λ) # good

    # println("struct mesh dv")
    # println("AD derivs: ", derivs[1])
    # println("FD derivs: ", fdderivs1)
    # println("semichord dv")
    # println("AD derivs: ", derivs[2])
    # println("FD derivs: ", fdderivs2)
    # println("chord vec dv")
    # println("AD derivs: ", derivs[3])
    # println("FD derivs: ", fdderivs3)
    # println("ab vec dv")
    # println("AD derivs: ", derivs[4])
    # println("FD derivs: ", fdderivs4)
    # println("eb vec dv")
    # println("AD derivs: ", derivs[5])
    # println("FD derivs: ", fdderivs5)
    # println("sweep dv")
    # println("AD derivs: ", derivs[6])
    # println("FD derivs: ", fdderivs6)
    test1 = derivs[1] - fdderivs1
    test2 = derivs[2] - fdderivs2
    test3 = derivs[3] - fdderivs3
    test4 = derivs[4] - fdderivs4
    test5 = derivs[5] - fdderivs5
    test6 = derivs[6] - fdderivs6

    return max(norm(test1, 2), norm(test2, 2), norm(test3, 2), norm(test4, 2), norm(test5, 2), norm(test6, 2))
end


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
nNodes = 4
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
)

solverOptions = Dict(
    # --- I/O ---
    "name" => "test",
    "debug" => false,
    "outputDir" => "./test_out/",
    # --- General solver options ---
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "nNodes" => nNodes,
    "config" => "wing",
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "tipMass" => false,
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => false,
    # --- Forced solve ---
    "run_forced" => false,
    "fSweep" => range(0.1, 1000.0, 1000),
    "tipForceMag" => 0.5 * 0.5 * 1000 * 100 * 0.03,
    # --- Eigen solve ---
    "run_modal" => false,
    "run_flutter" => true,
    "nModes" => 4,
    "uRange" => [187.0, 190.0],
    "maxQIter" => 100,
    "rhoKS" => 80.0,
)

derivs = test_pkflutterderiv(DVDict, solverOptions)


# derivs, fdderivs1, fdderivs2, fdderivs3, fdderivs4, fdderivs5 = test_hydroderiv(DVDict, solverOptions)
# test = test_hydroderiv(DVDict, solverOptions)