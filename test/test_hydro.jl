"""
Run tests on hydro module file
"""

using LinearAlgebra
using Printf


include("../src/DCFoil.jl")
using .DCFoil: SolveStatic, SolutionConstants, InitModel, FEMMethods, HydroStrip
using Plots, Printf
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
    ab = 0.0
    U = 5.0
    Λ = 45 * π / 180 # 45 deg
    clambda = cos(Λ)
    slambda = sin(Λ)
    ω = 1e10 # infinite frequency limit
    ρ = 1000.0
    k = ω * b / (U * cos(Λ))
    CKVec = HydroStrip.compute_theodorsen(k)
    Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2]
    Matrix, SweepMatrix = HydroStrip.compute_node_stiff_faster(clα, b, eb, ab, U, clambda, slambda, ρ, Ck)

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
    ab = 0.0
    U = 5.0
    Λ = 45 * π / 180 # 45 deg
    clambda = cos(Λ)
    slambda = sin(Λ)
    ω = 1e10 # infinite frequency limit
    ρ = 1000.0
    k = ω * b / (U * cos(Λ))
    CKVec = HydroStrip.compute_theodorsen(k)
    Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2] # TODO: for now, put it back together so solve is easy to debug
    Matrix, SweepMatrix = HydroStrip.compute_node_damp_faster(clα, b, eb, ab, U, clambda, slambda, ρ, Ck)

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
    Matrix = HydroStrip.compute_node_mass(b, ab, ρ)

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

function test_dwWave()
    FALTINSENDATACLvsFnc = [
        0.5094812005034943 0.3426978668091788
        0.5433651233527405 0.3499158955949865
        0.5758285113651205 0.34131039462543256
        0.5972677381400785 0.3597312329104295
        0.6284165013752634 0.37750318109997694
        0.6643105956038546 0.31019176904096146
        0.6468506238891395 0.35364149432649095
        0.6588488341736092 0.33030751747904974
        0.6975881065957248 0.28309854810000235
        0.7177389272903784 0.3128806628622546
        0.7230589987837779 0.33311098940145367
        0.7269223105419024 0.3484113211732518
        0.747578059268363 0.3697831358555631
        0.7527163614042112 0.3977708394159597
        0.7602982124406361 0.41819166180511114
        0.767277895259778 0.43583771741082633
        0.7709574751020014 0.45570589437165493
        0.7737246439854444 0.4703588968647249
        0.7807950102422779 0.4850295448211071
        0.8117787006400436 0.4999526137303307
        0.8434800804509636 0.44248826212708625
        0.83172748513974 0.4681376575844303
        0.8481549601173562 0.4216433968266685
        0.8558979238148724 0.39847452896502555
        0.862639559748705 0.3769313818815687
        0.888871249852291 0.2575665850922475
        0.8792665682956471 0.31711349385708254
        0.8864701699869266 0.2889837818710749
        0.9018663519768098 0.23392132359641382
        0.9079696697416079 0.2101699683073135
        0.9128085559262755 0.19992927144564887
        0.9222366199126888 0.15765838823947986
        0.9161946843918923 0.1798920973657624
        0.9344087296147973 0.12852017595227128
        0.9418438080985021 0.10735031091359803
        0.9485197599584166 0.09236664400152772
        0.9839094276395153 0.06857495388234536
        1.0147478320461998 0.08098950597435772
        1.02791912744865 0.0980766591046025
        1.037556252589681 0.11629333137032816
        1.0476137199218107 0.13995180640347227
        1.057817829844837 0.16372331674000107
        1.0694851443280249 0.18946444885221514
        1.0786686049446788 0.21092089691405358
        1.084759557560752 0.23339407804250367
        1.0901989759456832 0.2531812117586576
        1.095650319895372 0.2698613383366598
        1.1078445244814372 0.29518117249704656
        1.0978145403574262 0.2807845420642734
        1.1197362040305139 0.3172081808502193
        1.1331722072750483 0.3930578452505979
        1.124057084998392 0.3410242089019534
        1.1283670737219271 0.3676780296798253
        1.16526832207864 0.44273328020471625
        1.1786002488339549 0.5000812936459995
        1.1720375724176564 0.4855529316480076
        1.18292845966172 0.5219876488685008
        1.1912396776348624 0.5407824704099318
        1.1993273602977048 0.5603382139840479
        1.2091909002404162 0.5770437607294862
        1.2183809942059345 0.5967720003004784
        1.2306391007727577 0.62186537482953
        1.2398220814741816 0.6434468568059688
        1.2551770350011995 0.6663930546569535
        1.2683528118038574 0.6823126536531445
        1.2797647620783037 0.6979480376462506
        1.2947008547963945 0.7150664832947949
        1.3167003793153365 0.7312047763742255
        1.347521140213331 0.7482160505968596
        1.3960410959051135 0.7521953060761464
        1.4402160437360798 0.738644393343945
        1.4733785477119756 0.7203283124455093
        1.4999249929199572 0.7013917515566851
        1.5242581414029768 0.6843179893031484
        1.5463899121516107 0.6660017313400982
        1.586234416562865 0.63112612800729
        1.6010130659677104 0.6126271682299798
        1.6282934418116266 0.5940688039172956
        1.641577939136433 0.58166308287645
        1.6592890299111627 0.5655317645650293
        1.679214811794694 0.5471743641486395
        1.6984013125353243 0.5298327525711248
        1.7234877044073484 0.5081055709605178
        1.7411828495054966 0.4961286329211957
        1.7588956660028003 0.4795477063552188
        1.7810263322889872 0.46151919767508454
        1.8031553418815032 0.44392231291932427
        1.8252799336242302 0.4274764252952275
        1.8496111492979672 0.41090622428679374
        1.8761546032534875 0.39274898437253336
        1.9026920747040856 0.37615038640740095
        1.9314367617620043 0.35927332365636944
        1.9675235601632606 0.3454474348371098
        1.9955443476901102 0.32557873832234363
        2.0309088901056054 0.3083330174016772
        2.075100503485998 0.29044017352547646
        2.119287975132214 0.2736263894602107
        2.1700885690787266 0.2581981330819749
        2.21867387213197 0.24515222265540282
        2.271668741505742 0.23281684973654282
        2.3268640989080396 0.22195577056065618
        2.3776345502335965 0.21438067169533426
        2.4328158717589607 0.20717640632317103
        2.483579292362921 0.20143304479122592
        2.5343321924507465 0.19843062839631576
        2.585083584129211 0.19582120292020289
        2.6380428652685723 0.19275775133974482
        2.693204973749282 0.19055912453497303
        2.746155664625165 0.18973372293275015
        2.799103939489445 0.18953777288690565
        2.8520508337756665 0.1897015094447061
        2.902790183745505 0.19022935045594036
        2.95573219670216 0.1916648360766282
        3.0064659356083157 0.19365454621267641
        3.059406755351077 0.19540090382651454
        3.114557665809752 0.19611973502908497
        3.167497312061163 0.19817182625602114
        3.218234322718179 0.1993091362345426
        3.273379979310352 0.20139677478987283
        3.3263183983812663 0.20376858744227122
        3.3770542969059574 0.20519564496261766
        3.4299884208451323 0.2086864826041337
        3.4829291025300875 0.2104688088783362
        3.535867023058925 0.21297050835982856
        3.5888113287612855 0.2138086572994633
        3.648367781078508 0.21628653250460095
        3.6968939399945895 0.21864964127994513
        3.7454209437032637 0.22079265342509657
        3.7939499999578756 0.22240090893527842
        3.8424741330369185 0.2252918157870264
        3.8910028426528833 0.22699038221933543
        3.939901627384061 0.22806784109738043
        3.983652185519682 0.2292887945298282
    ]
    semispan = 1.0
    nNodes = 10
    chordVec = 0.1 * ones(nNodes)
    alfa0 = deg2rad(4.0)
    Uinf = 2.0
    GRAV = 9.81
    depth = 0.1
    # ξRange = LinRange(0, 10, 100)
    pcRatio = 6
    URange = LinRange(0.5, 10, 1000)
    chordRefM = sum(chordVec) / nNodes

    clalfa, _, _ = DCFoil.HydroStrip.compute_glauert_circ(semispan, chordVec, alfa0, Uinf, nNodes)

    cl = clalfa * alfa0
    ξ = pcRatio * chordRefM
    Fnc = Uinf / sqrt(GRAV * chordRefM)
    Fnh = Uinf / sqrt(GRAV * depth)

    alphavsXi = []
    # for ξ in ξRange
    FncRange = []
    for U in URange

        Fnc = U / sqrt(GRAV * chordRefM)
        Fnh = U / sqrt(GRAV * depth)

        alpha = DCFoil.HydroStrip.compute_wavePatternDWAng(cl, chordVec, Fnc, Fnh, ξ)

        # FALTINSEN TEST CODE
        cltest = 0.35 * ones(nNodes)
        cl = cltest[1]
        alpha = DCFoil.HydroStrip.compute_wavePatternDWAng(cl, chordRefM, Fnc, Fnh, ξ)

        # Just grab root value
        push!(alphavsXi, alpha[1])
        push!(FncRange, Fnc)
    end

    # plot(ξRange, alphavsXi, xlabel="ξ [m]", ylabel="α_dw [rad]", title=@sprintf("α vs downstream distance ξ \n Fnc = %.1f, Fnh = %.1f", Fnc, Fnh))
    plot(FncRange, alphavsXi, xlabel="Fn_c", ylabel="α_dw [rad]", title=@sprintf("α_w vs Fnc\n ξ/c=%.1f", pcRatio), label=:false)

    # FALTINSEN TEST CODE
    cl2 = 0.35 .+ 2π * alphavsXi
    plot(FncRange, cl2, xlabel="Fn_c", ylabel="CL2", title=@sprintf("CL2 vs. Fn_c (U) \n ξ/c=%.1f", pcRatio), label="Eqn")
    scatter!(FALTINSENDATACLvsFnc[:, 1], FALTINSENDATACLvsFnc[:, 2], label="Faltinsen 2005")
    xlims!(0.5, 4)
    plot!(tick_direction=:out)
    savefig("test_dwWave.pdf")
    println("Making plot test_dwWave.pdf to check")
    return 1e-6
end

function test_dwWake()
    semispan = 1.0
    nNodes = 10
    chordVec = 0.1 * ones(nNodes)
    alfa0 = deg2rad(4.0)
    Uinf = 2.0
    GRAV = 9.81
    depth = 0.4
    chordRefM = sum(chordVec) / nNodes

    GLAURTVALUES = [
        1/3 3.23
        2/3 2.43
        1.0 2.22
        2.0 2.06
    ]

    clalfa, _, _ = DCFoil.HydroStrip.compute_glauert_circ(semispan, chordVec, alfa0, Uinf, nNodes)

    cl = clalfa * alfa0
    CL = 0.35

    xM = 0.0
    xRRange = LinRange(0, 10, 100)
    epsvsXr = []
    # αi on the wing
    ϵ0 = CL / (π * (2 * semispan) / chordRefM)
    for xR in xRRange
        ϵ = DCFoil.HydroStrip.compute_wakeDWAng(semispan * 2, chordRefM, CL, xR)
        push!(epsvsXr, ϵ)
    end

    plot(xRRange ./ (semispan * 2), epsvsXr ./ ϵ0, xlabel="x/s ", ylabel="ε/αi", title=@sprintf("ε vs downstream distance \n CL_M = %.2f", CL), label="Eqn")
    # Plot glauert values as scatter
    scatter!(GLAURTVALUES[:, 1], GLAURTVALUES[:, 2], label="Glauert 1983")
    xlims!(0.2, 3)
    ylims!(1.5, 6.0)
    # set tick direction out for plot
    plot!(tick_direction=:out)

    savefig("test_dwWake.pdf")
    println("Making plot test_dwWake.pdf to check")
    return 1e-6
end

function test_AICs()
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
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        "s_strut" => 0.4, # from Yingqian
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
        "use_tipMass" => false,
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
        "fRange" => [1, 2],
        "tipForceMag" => 1,
        # --- p-k (Eigen) solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 1,
        "uRange" => [1, 2],
        "maxQIter" => 100, # that didn't fix the slow run time...
        "rhoKS" => 80.0,
    )

    FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

    elemType = "BT2"
    # elemType = "COMP2"
    alfaRad = deg2rad(4.0)
    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad]
    elseif elemType == "BT2"
        nDOF = 4
        # nGDOF = nDOF * 3 # number of DOFs on node in global coordinates
        staticOffset = [0, 0, alfaRad, 0] #TODO: pretwist will change this
    # staticOffset = [0, 0, 0, 0, 0, 0, alfaRad, 0, 0, 0, 0, 0]
    elseif elemType == "COMP2"
        nDOF = 9
        staticOffset = [0, 0, 0, alfaRad, 0, 0, 0, 0, 0]
    end
    AIC = zeros(nDOF * nNodes, nDOF * nNodes)

    AIC, planformArea = HydroStrip.compute_steady_AICs!(AIC, aeroMesh, chordVec, abVec, ebVec, Λ, FOIL, elemType)
    dummy = AIC[:, 1]
    w = dummy[1:nDOF:end]
    structStates = (repeat(staticOffset, outer=[length(w)]))
    fHydro = -1 * AIC * structStates
    return AIC, fHydro
end

# function run_hydrofoil()
#     chordVec = ones(10)
#     alpha = 1.0
#     Uinf = 5.0
#     clalpha, cl, gamma = HydroStrip.compute_spanwise_vortex(semispan, chordVec, alpha, Uinf, nNodes, "elliptical")
#     clalpha, cl, gamma = HydroStrip.compute_spanwise_vortex(semispan, chordVec, alpha, Uinf, nNodes, "chord")

# end

# AIC, fHydro = test_AICs()