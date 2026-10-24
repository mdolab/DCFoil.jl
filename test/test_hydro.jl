"""
Run tests on hydro module file
"""



for headerName in [
    "../src/struct/FEMMethods",
    "../src/hydro/LiftingLine",
    "../src/hydro/HydroStrip",
    "../src/hydro/Unsteady",
    "../src/hydro/OceanWaves",
    "../src/ComputeHydroFunctions",
    "../src/constants/SolutionConstants",
]
    include(headerName * ".jl")
end

using LinearAlgebra
using JLD2
using Plots, Printf
using DelimitedFiles
using .HydroStrip
using .LiftingLine

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
    ω = 1e6 # infinite frequency limit
    ρ = 1000.0
    k = ω * b / (U * cos(Λ))
    println("ω = ", ω)
    println("k = ", k)
    CKVec = HydroStrip.compute_theodorsen(k)
    Ck::ComplexF64 = CKVec[1] + 1im * CKVec[2]
    Matrix, SweepMatrix = HydroStrip.compute_node_stiff_faster(clα, b, eb, ab, U, clambda, slambda, ρ, Ck)

    Sk, S0k = HydroStrip.compute_sears(k)


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
    ref_sol = 625 * √(2) * vec([
        2π -1.5*π
        0.5*π 0.125*π
    ])
    sweepref_sol = 625 * √(2) * vec([
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
    ω = 1e6 # infinite frequency limit
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

function test_pFactor()

    p_i = []
    depth = 1.0
    chordLocal = 0.1
    for hLocal in LinRange(0, 1, 100)
        push!(p_i, HydroStrip.compute_pFactor(depth, chordLocal, hLocal))
    end

    # plot(LinRange(0, 1, 100), p_i, xlabel="h/c", ylabel="p_i", title="p_i vs h/c")
    # savefig("test_pFactor.pdf")
    println(vec(p_i))
end

# test_pFactor()
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

    solverOptions = Dict(
        "Uinf" => Uinf,
    )

    clalfa, _, _ = DCFoil.HydroStrip.compute_hydroLLProperties(semispan * 2, chordVec, alfa0, 0.0, 0.0, 0.0; solverOptions)
    # clalfa, _, _ = DCFoil.HydroStrip.compute_glauert_circ(semispan, chordVec, alfa0, Uinf)

    cl = clalfa * alfa0
    ξ = pcRatio * chordRefM
    Fnc = Uinf / √(GRAV * chordRefM)
    Fnh = Uinf / √(GRAV * depth)

    alphavsXi = []
    # for ξ in ξRange
    FncRange = []
    for U in URange

        Fnc = U / √(GRAV * chordRefM)
        Fnh = U / √(GRAV * depth)

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

    solverOptions = Dict(
        "Uinf" => Uinf,
    )

    clalfa, _, _ = DCFoil.HydroStrip.compute_hydroLLProperties(semispan * 2, chordVec, alfa0, 0.0, 0.0, 0.0; solverOptions)
    # clalfa, _, _ = DCFoil.HydroStrip.compute_glauert_circ(semispan, chordVec, alfa0, Uinf)

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
        "alfa0" => 6.0, # initial angle of attack [deg]
        "sweep" => deg2rad(-15.0), # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12, # thickness-to-chord ratio
        "x_ab" => 0 * ones(nNodes), # static imbalance [m]
        "theta_f" => deg2rad(15), # fiber angle global [rad]
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
        "Uinf" => 5.0, # free stream velocity [m/s]
        "rhof" => 1000.0, # fluid density [kg/m³]
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
        "rhoKS" => 100.0,
    )

    FOIL = InitModel.init_modelFromDVDict(DVDict, solverOptions)

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


# AIC, fHydro = test_AICs()
function test_biplanefs()
    λRange = 0.0:0.05:0.8
    σRng = []
    γRng = []
    for λ in λRange
        σ, γ = HydroStrip.compute_biplanefreesurface(λ)
        push!(σRng, σ)
        push!(γRng, γ)
    end
    p1 = plot(λRange, σRng, xlabel="λ", ylabel="σ", title="σ vs λ")
    p2 = plot(λRange, γRng, xlabel="λ", ylabel="γ", title="γ vs λ")
    ylims!(p1, 0, 0.9)
    ylims!(p2, 0, 0.4)
    xlims!(p1, 0, 0.7)
    xlims!(p2, 0, 0.4)
    # make taller plot
    plot(p1, p2, layout=(2, 1), size=(400, 900))


    savefig("test_biplanefs.pdf")
    return collect(λRange), σRng, γRng
end

function test_wavedrag()
    CL = 1.0
    chord = 8.0
    FncRange = 0.6:0.1:10.0
    solverOptions = Dict(
        "Uinf" => 5.0,
    )
    appendageParams = Dict(
        "depth0" => 6.72,
        "s" => 80.0,
    )
    areaRef = 1.0 # doesn't matter for this test
    qdyn = 1.0 # doesn't matter for this test
    AR = appendageParams["s"] / chord
    vals = []
    aeroSpan = appendageParams["s"]
    for Fnc in FncRange

        solverOptions["Uinf"] = Fnc * sqrt(9.81 * chord)

        Cdw, _ = compute_wavedrag(CL, chord, qdyn, areaRef, aeroSpan, appendageParams, solverOptions)
        CDi = Cdw + 1 / (π * AR)
        push!(vals, CDi / CL^2)
    end
    return vals, FncRange
end

function test_hydrofoilWaveLoads()
    # should be higher k
    # chord = 0.1 # m
    chord = 0.3 # m
    # depth = 0.5 # m
    depth = 1.0 # m
    Uinf = 10.0 # m/s
    rho = 1000.0 # kg/m^3
    # Hsig = 0.8 # m
    # T_m = 6.1 # s , modal period
    T_m = 5.0 # s , modal period Sea state 3
    ω_m = 2 * π / T_m # rad/s
    # --- Great Lakes typical seas ---
    T_z = 3.0 # s, significant period
    ω_z = 2π / T_z # rad/s significant frequency
    Hsig = 1.5 # m
    Hsig = 0.8 # m
    omegaSweep::Vector = (0.05:0.05:(3.0*2π)) # rad/s
    β = deg2rad(180)
    we = compute_encounterFreq(β, omegaSweep, Uinf)

    kRedSweep = we * chord * 0.5 / Uinf

    hcRatio = depth / chord
    corrFactor = (1.0 + 16.0 * hcRatio^2) / (2.0 + 16.0 * hcRatio^2)
    claVec = corrFactor * (2π) * ones(2) # lift curve slope [1/rad]

    Swave_pm, wm, lm, var, Vwind = compute_PMWaveSpectrum(Hsig, omegaSweep)
    println("Hsig = ", Hsig)
    println("T_m = ", T_m)
    println("ω_m PM = ", wm)
    println("ω_m = ", ω_m)
    println("Vwind = ", Vwind)
    Swave_bs = compute_BSWaveSpectrum(Hsig, ω_z, omegaSweep)

    chordVec = chord * ones(2)
    stripWidths = ones(2)
    fAeySk, mAey, ampDist = compute_waveloads(chordVec, Uinf, rho, we, omegaSweep, 1.0, depth, stripWidths, claVec; method="Sears")
    fAeyCk, mAey, ampDist = compute_waveloads(chordVec, Uinf, rho, we, omegaSweep, 1.0, depth, stripWidths, claVec; method="Theodorsen")
    fAeyQS, mAey, ampDist = compute_waveloads(chordVec, Uinf, rho, we, omegaSweep, 1.0, depth, stripWidths, claVec; method="Quasisteady")

    HwSk = abs.(fAeySk[:, 1])
    HwCk = abs.(fAeyCk[:, 1])
    HwQS = abs.(fAeyQS[:, 1])
    # SliftSk = compute_responseSpectralDensityFunc(HwSk, Swave_bs)
    # SliftCk = compute_responseSpectralDensityFunc(HwCk, Swave_bs)
    # SliftQS = compute_responseSpectralDensityFunc(HwQS, Swave_bs)
    ZwSk = fAeySk[:, 1]
    ZwCk = fAeyCk[:, 1]
    ZwQS = fAeyQS[:, 1]
    SliftSk = compute_responseSpectralDensityFunc(ZwSk, Swave_bs)
    SliftCk = compute_responseSpectralDensityFunc(ZwCk, Swave_bs)
    SliftQS = compute_responseSpectralDensityFunc(ZwQS, Swave_bs)

    percDiff = abs.(SliftCk - SliftSk) ./ (abs.(SliftSk) .+ 1)
    println("="^100)
    println(percDiff[1:30])
    println("="^100)
    # println(kRedSweep)
    # println("="^100)
    println("Max perc diff Sears vs Theodorsen = ", maximum(percDiff), "%")
    percDiff = abs.(SliftQS - SliftSk) ./ (abs.(SliftSk) .+ 1)
    println("Max perc diff Sears vs Quasisteady = ", maximum(percDiff), "%")

    # println("Swave_bs")
    # println(Swave_bs)
    # println("Swave_pm")
    # println(Swave_pm)
    p1 = plot(omegaSweep/2π, Swave_bs, xlabel="ω [rad/s]", label="P-M")
    p1 = plot!(omegaSweep/2π, Swave_pm, xlabel="ω [rad/s]", ylabel="S(ω)", title="Wave Spectrum", label="Bretschneider")
    p2 = plot(omegaSweep/2π, HwSk, label="(Sears)")
    p2 = plot!(omegaSweep/2π, HwCk, label="(Theodorsen)")
    p2 = plot!(omegaSweep/2π, HwQS, xlabel="ω [1/s]", ylabel="H(ω) = Lift / ζ_wave", title="Wave-induced load transfer function", label="(Quasisteady)")
    xlims!(p2, 0, 1)
    # p1 = plot(kRedSweep, Swave_bs, label="Bretschneider")
    # p1 = plot!(kRedSweep, Swave_pm, xlabel="Reduced frequency", ylabel="S(ω)", title="Wave Spectrum", label="P-M", xlims=(0, 0.3))
    # p2 = plot(kRedSweep, HwSk, label="Sears")
    # p2 = plot!(kRedSweep, HwCk, label="Theodorsen")
    # p2 = plot!(kRedSweep, HwQS, xlabel="Reduced frequency", ylabel="H(ω) = Lift / ζ_wave", title="Wave-induced load transfer function", label="Quasisteady", xlims=(0, 1))
    savefig("test_waveSpectrum.pdf")

    p3 = plot(omegaSweep, SliftSk, xlabel="ω [rad/s]", ylabel="S_lift(ω)", title="Lift response spectral density", label="(Sears)")
    p3 = plot!(omegaSweep, SliftCk, xlabel="ω [rad/s]", ylabel="S_lift(ω)", title="Lift response spectral density", label="(Theodorsen)")
    p3 = plot!(omegaSweep, SliftQS, xlabel="ω [rad/s]", ylabel="S_lift(ω) [N^2-s]", title="Lift response spectral density", label="(Quasisteady)", xlims=(0, 1))

    p3 = plot(kRedSweep, SliftSk, title="Lift response spectral density", label="(Sears)")
    p3 = plot!(kRedSweep, SliftCk, title="Lift response spectral density", label="(Theodorsen)")
    p3 = plot!(kRedSweep, SliftQS, xlabel="Reduced frequency", ylabel="S_lift(ω) [N^2-s]", title="Lift response spectral density", label="(Quasisteady)", xlims=(0, 0.3))
    plot(p1, p3, layout=(2, 1), size=(600, 900))
    savefig("test_liftResponseSpectrum.pdf")


    # --- Save data to JLD for python pretty plotting ---
    save("test_waveSpectrum.jld2", "omegaSweep", omegaSweep, "encounter", we, "k", kRedSweep, "HwSk", HwSk, "HwCk", HwCk, "HwQS", HwQS, "Swave_bs", Swave_bs, "Swave_pm", Swave_pm, "SliftSk", SliftSk, "SliftCk", SliftCk, "SliftQS", SliftQS)

end
# test_hydrofoilWaveLoads()
# ==============================================================================
#                         Test lifting line code
# ==============================================================================
function test_VPM()
    airfoilX = [1.00000000e+00, 9.98983961e-01, 9.95940183e-01,
        9.90881655e-01, 9.83829939e-01, 9.74815051e-01,
        9.63875290e-01, 9.51057023e-01, 9.36414438e-01,
        9.20009251e-01, 9.01910391e-01, 8.82193653e-01,
        8.60941320e-01, 8.38241778e-01, 8.14189094e-01,
        7.88882584e-01, 7.62426369e-01, 7.34928905e-01,
        7.06502508e-01, 6.77262862e-01, 6.47328517e-01,
        6.16820381e-01, 5.85861206e-01, 5.54575073e-01,
        5.23086880e-01, 4.91521839e-01, 4.60004979e-01,
        4.28660672e-01, 3.97631556e-01, 3.67253779e-01,
        3.37414109e-01, 3.08224091e-01, 2.79792467e-01,
        2.52224996e-01, 2.25624298e-01, 2.00089704e-01,
        1.75717105e-01, 1.52598783e-01, 1.30823213e-01,
        1.10474834e-01, 9.16337794e-02, 7.43755645e-02,
        5.87707351e-02, 4.48844859e-02, 3.27762555e-02,
        2.24993132e-02, 1.41003523e-02, 7.61910468e-03,
        3.08799469e-03, 5.31844686e-04, -2.83870691e-05,
        1.44008274e-03, 4.94200664e-03, 1.04698608e-02,
        1.80077131e-02, 2.75311237e-02, 3.90070567e-02,
        5.23938162e-02, 6.76410223e-02, 8.46896392e-02,
        1.03472072e-01, 1.23912338e-01, 1.45926329e-01,
        1.69422161e-01, 1.94300609e-01, 2.20455639e-01,
        2.47775004e-01, 2.76140921e-01, 3.05430784e-01,
        3.35517928e-01, 3.66272407e-01, 3.97561776e-01,
        4.29024490e-01, 4.60745064e-01, 4.92612197e-01,
        5.24495035e-01, 5.56263127e-01, 5.87786972e-01,
        6.18938555e-01, 6.49591858e-01, 6.79623359e-01,
        7.08912505e-01, 7.37342170e-01, 7.64799099e-01,
        7.91174326e-01, 8.16363574e-01, 8.40267633e-01,
        8.62792718e-01, 8.83850790e-01, 9.03359866e-01,
        9.21244282e-01, 9.37434939e-01, 9.51869515e-01,
        9.64492643e-01, 9.75256067e-01, 9.84118763e-01,
        9.91047043e-01, 9.96014629e-01, 9.99002715e-01,
        1.00000000e+00]

    airfoilY = [1.32931687e-17, -7.38342600e-05, -2.94838388e-04,
        -6.61534665e-04, -1.17151910e-03, -1.82154316e-03,
        -2.60761565e-03, -3.52511341e-03, -4.56888783e-03,
        -5.73335407e-03, -7.01255010e-03, -8.40015486e-03,
        -9.88945762e-03, -1.14732744e-02, -1.31438115e-02,
        -1.48924812e-02, -1.67096794e-02, -1.85845380e-02,
        -2.05046708e-02, -2.24559320e-02, -2.44222083e-02,
        -2.63852669e-02, -2.83246777e-02, -3.02178266e-02,
        -3.20400333e-02, -3.37647782e-02, -3.53640411e-02,
        -3.68087440e-02, -3.80696794e-02, -3.91912332e-02,
        -4.01948507e-02, -4.10450021e-02, -4.17065851e-02,
        -4.21457179e-02, -4.23304737e-02, -4.22315281e-02,
        -4.18226899e-02, -4.10812967e-02, -3.99884614e-02,
        -3.85291639e-02, -3.66921921e-02, -3.44699408e-02,
        -3.18580895e-02, -2.88551809e-02, -2.54621317e-02,
        -2.16817065e-02, -1.75179898e-02, -1.29758887e-02,
        -8.06069877e-03, -2.77775776e-03, 2.82808768e-03,
        8.51222505e-03, 1.42221385e-02, 1.99372801e-02,
        2.56298579e-02, 3.12655584e-02, 3.68044732e-02,
        4.22021973e-02, 4.74110551e-02, 5.23814122e-02,
        5.70630246e-02, 6.14063789e-02, 6.53639803e-02,
        6.88915470e-02, 7.19490770e-02, 7.45017605e-02,
        7.65207179e-02, 7.79835532e-02, 7.88747205e-02,
        7.91857090e-02, 7.89150602e-02, 7.80682354e-02,
        7.67163113e-02, 7.49590252e-02, 7.28229631e-02,
        7.03373443e-02, 6.75339270e-02, 6.44465399e-02,
        6.11106600e-02, 5.75630455e-02, 5.38414246e-02,
        4.99842364e-02, 4.60304159e-02, 4.20192105e-02,
        3.79900151e-02, 3.39822078e-02, 3.00349748e-02,
        2.61871084e-02, 2.24767697e-02, 1.89412087e-02,
        1.56164396e-02, 1.25368744e-02, 9.73492038e-03,
        7.24055476e-03, 5.08088827e-03, 3.27973662e-03,
        1.85721674e-03, 8.29386282e-04, 2.07943232e-04,
        -1.32931687e-17]
    airfoilCtrlX = [9.99491981e-01, 9.97462072e-01, 9.93410919e-01,
        9.87355797e-01, 9.79322495e-01, 9.69345170e-01,
        9.57466157e-01, 9.43735731e-01, 9.28211845e-01,
        9.10959821e-01, 8.92052022e-01, 8.71567487e-01,
        8.49591549e-01, 8.26215436e-01, 8.01535839e-01,
        7.75654476e-01, 7.48677637e-01, 7.20715707e-01,
        6.91882685e-01, 6.62295690e-01, 6.32074449e-01,
        6.01340793e-01, 5.70218139e-01, 5.38830977e-01,
        5.07304360e-01, 4.75763409e-01, 4.44332826e-01,
        4.13146114e-01, 3.82442667e-01, 3.52333944e-01,
        3.22819100e-01, 2.94008279e-01, 2.66008731e-01,
        2.38924647e-01, 2.12857001e-01, 1.87903405e-01,
        1.64157944e-01, 1.41710998e-01, 1.20649023e-01,
        1.01054307e-01, 8.30046719e-02, 6.65731498e-02,
        5.18276105e-02, 3.88303707e-02, 2.76377844e-02,
        1.82998328e-02, 1.08597285e-02, 5.35354969e-03,
        1.80991969e-03, 2.51728808e-04, 7.05847836e-04,
        3.19104469e-03, 7.70593373e-03, 1.42387870e-02,
        2.27694184e-02, 3.32690902e-02, 4.57004365e-02,
        6.00174193e-02, 7.61653308e-02, 9.40808554e-02,
        1.13692205e-01, 1.34919333e-01, 1.57674245e-01,
        1.81861385e-01, 2.07378124e-01, 2.34115321e-01,
        2.61957963e-01, 2.90785852e-01, 3.20474356e-01,
        3.50895168e-01, 3.81917092e-01, 4.13293133e-01,
        4.44884777e-01, 4.76678630e-01, 5.08553616e-01,
        5.40379081e-01, 5.72025050e-01, 6.03362763e-01,
        6.34265207e-01, 6.64607609e-01, 6.94267932e-01,
        7.23127337e-01, 7.51070634e-01, 7.77986712e-01,
        8.03768950e-01, 8.28315603e-01, 8.51530175e-01,
        8.73321754e-01, 8.93605328e-01, 9.12302074e-01,
        9.29339610e-01, 9.44652227e-01, 9.58181079e-01,
        9.69874355e-01, 9.79687415e-01, 9.87582903e-01,
        9.93530836e-01, 9.97508672e-01, 9.99501358e-01]
    airfoilCtrlY = [-3.69171300e-05, -1.84336324e-04, -4.78186527e-04,
        -9.16526881e-04, -1.49653113e-03, -2.21457940e-03,
        -3.06636453e-03, -4.04700062e-03, -5.15112095e-03,
        -6.37295209e-03, -7.70635248e-03, -9.14480624e-03,
        -1.06813660e-02, -1.23085429e-02, -1.40181464e-02,
        -1.58010803e-02, -1.76471087e-02, -1.95446044e-02,
        -2.14803014e-02, -2.34390701e-02, -2.54037376e-02,
        -2.73549723e-02, -2.92712522e-02, -3.11289300e-02,
        -3.29024058e-02, -3.45644097e-02, -3.60863926e-02,
        -3.74392117e-02, -3.86304563e-02, -3.96930420e-02,
        -4.06199264e-02, -4.13757936e-02, -4.19261515e-02,
        -4.22380958e-02, -4.22810009e-02, -4.20271090e-02,
        -4.14519933e-02, -4.05348791e-02, -3.92588127e-02,
        -3.76106780e-02, -3.55810665e-02, -3.31640152e-02,
        -3.03566352e-02, -2.71586563e-02, -2.35719191e-02,
        -1.95998481e-02, -1.52469393e-02, -1.05182938e-02,
        -5.41922826e-03, 2.51649599e-05, 5.67015636e-03,
        1.13671818e-02, 1.70797093e-02, 2.27835690e-02,
        2.84477082e-02, 3.40350158e-02, 3.95033352e-02,
        4.48066262e-02, 4.98962337e-02, 5.47222184e-02,
        5.92347017e-02, 6.33851796e-02, 6.71277636e-02,
        7.04203120e-02, 7.32254187e-02, 7.55112392e-02,
        7.72521355e-02, 7.84291368e-02, 7.90302147e-02,
        7.90503846e-02, 7.84916478e-02, 7.73922733e-02,
        7.58376682e-02, 7.38909942e-02, 7.15801537e-02,
        6.89356356e-02, 6.59902334e-02, 6.27785999e-02,
        5.93368528e-02, 5.57022351e-02, 5.19128305e-02,
        4.80073261e-02, 4.40248132e-02, 4.00046128e-02,
        3.59861114e-02, 3.20085913e-02, 2.81110416e-02,
        2.43319390e-02, 2.07089892e-02, 1.72788242e-02,
        1.40766570e-02, 1.11358974e-02, 8.48773757e-03,
        6.16072151e-03, 4.18031245e-03, 2.56847668e-03,
        1.34330151e-03, 5.18664757e-04, 1.03971616e-04]

    airfoilXY = copy(hcat(airfoilX, airfoilY)')
    airfoilCtrlXY = copy(hcat(airfoilCtrlX, airfoilCtrlY)')

    Airfoil, Amat = VPM.setup(airfoilX, airfoilY, airfoilCtrlXY)


    V = [0.9961947, 0.0, -0.08715574]
    angle = rad2deg(atan(V[3], V[1]))
    println("Angle airfoil is rotated wrt xz axis: ", angle, "deg")
    cl, cm, Gamma, cpDist = VPM.solve(Airfoil, Amat, V)
    println("cl", cl)
    println("cm", cm)
    println("Cp dist: $(cpDist)")
    p = plot(airfoilX, -cpDist, label="-Cp", layout=(2, 1), xlims=(-0.05, 1.05), ylims=(-1.2, 2.2))
    plot!(airfoilX, airfoilY, subplot=2, aspect_ratio=:equal, ylims=(-0.1, 0.1), xlims=(-0.05, 1.05), label="Airfoil")
    savefig(p, "test_VPM.png")
end

function test_FSVPM()
    depth = 0.94
    # depth = 0.6
    # NACA4412
    airfoilX = [
        1.00000000e+00, 9.98974707e-01, 9.95903443e-01, 9.90800015e-01,
        9.83687326e-01, 9.74597208e-01, 9.63570206e-01, 9.50655298e-01,
        9.35909574e-01, 9.19397869e-01, 9.01192368e-01, 8.81372177e-01,
        8.60022873e-01, 8.37236034e-01, 8.13108750e-01, 7.87743130e-01,
        7.61245782e-01, 7.33727299e-01, 7.05301720e-01, 6.76085992e-01,
        6.46199427e-01, 6.15763152e-01, 5.84899565e-01, 5.53731800e-01,
        5.22383202e-01, 4.90976832e-01, 4.59634987e-01, 4.28478769e-01,
        3.97666445e-01, 3.67744364e-01, 3.38361482e-01, 3.09618475e-01,
        2.81613157e-01, 2.54440647e-01, 2.28193549e-01, 2.02962114e-01,
        1.78834370e-01, 1.55896178e-01, 1.34231209e-01, 1.13920809e-01,
        9.50437679e-02, 7.76759680e-02, 6.18899436e-02, 4.77543501e-02,
        3.53333750e-02, 2.46861149e-02, 1.58659480e-02, 8.91993411e-03,
        3.88827065e-03, 8.03831091e-04, -3.00373474e-04, 6.39806776e-04,
        3.64117721e-03, 8.70426511e-03, 1.58209115e-02, 2.49740042e-02,
        3.61371925e-02, 4.92746077e-02, 6.43406188e-02, 8.12796507e-02,
        1.00026096e-01, 1.20504341e-01, 1.42628934e-01, 1.66304896e-01,
        1.91428199e-01, 2.17886387e-01, 2.45559353e-01, 2.74320231e-01,
        3.04036400e-01, 3.34570555e-01, 3.65781823e-01, 3.97526886e-01,
        4.29206393e-01, 4.61115056e-01, 4.93157204e-01, 5.25198713e-01,
        5.57106400e-01, 5.88748612e-01, 6.19995783e-01, 6.50720948e-01,
        6.80800229e-01, 7.10113293e-01, 7.38543775e-01, 7.65979685e-01,
        7.92313780e-01, 8.17443917e-01, 8.41273378e-01, 8.63711165e-01,
        8.84672266e-01, 9.04077890e-01, 9.21855664e-01, 9.37939803e-01,
        9.52271240e-01, 9.64797727e-01, 9.75473910e-01, 9.84261376e-01,
        9.91128682e-01, 9.96051369e-01, 9.99011969e-01, 1.00000000e+00
    ]
    airfoilY = [
        1.32058085e-17, -5.85693979e-06, -2.39190185e-05, -5.56612324e-05,
        -1.03544425e-04, -1.71013800e-04, -2.62486805e-04, -3.83321485e-04,
        -5.39755372e-04, -7.38804941e-04, -9.88116680e-04, -1.29576290e-03,
        -1.66997844e-03, -2.11883812e-03, -2.64987941e-03, -3.26967902e-03,
        -3.98339693e-03, -4.79430543e-03, -5.70332403e-03, -6.70858346e-03,
        -7.80504293e-03, -8.98418462e-03, -1.02338069e-02, -1.15379346e-02,
        -1.28768604e-02, -1.42273231e-02, -1.55628261e-02, -1.68540860e-02,
        -1.80703699e-02, -1.93232053e-02, -2.06768516e-02, -2.20814301e-02,
        -2.34848011e-02, -2.48336311e-02, -2.60744178e-02, -2.71544412e-02,
        -2.80226144e-02, -2.86302139e-02, -2.89314790e-02, -2.88840740e-02,
        -2.84494170e-02, -2.75928839e-02, -2.62839022e-02, -2.44959539e-02,
        -2.22065076e-02, -1.93969057e-02, -1.60522283e-02, -1.21611574e-02,
        -7.71586162e-03, -2.71191984e-03, 2.81257968e-03, 8.61891417e-03,
        1.46536569e-02, 2.08908090e-02, 2.72932086e-02, 3.38133609e-02,
        4.03945385e-02, 4.69721178e-02, 5.34751124e-02, 5.98278572e-02,
        6.59517953e-02, 7.17673140e-02, 7.71955810e-02, 8.21603284e-02,
        8.65895389e-02, 9.04169913e-02, 9.35836311e-02, 9.60387372e-02,
        9.77408669e-02, 9.86585682e-02, 9.87708593e-02, 9.80674819e-02,
        9.66692205e-02, 9.47527942e-02, 9.23436929e-02, 8.94714823e-02,
        8.61701354e-02, 8.24775313e-02, 7.84349707e-02, 7.40867174e-02,
        6.94795688e-02, 6.46624552e-02, 5.96860611e-02, 5.46024591e-02,
        4.94647467e-02, 4.43266720e-02, 3.92422389e-02, 3.42652799e-02,
        2.94489926e-02, 2.48454339e-02, 2.05049761e-02, 1.64757284e-02,
        1.28029354e-02, 9.52836501e-03, 6.68970403e-03, 4.31997948e-03,
        2.44702538e-03, 1.09301481e-03, 2.74074884e-04, -1.32058085e-17
    ]
    airfoilCtrlX = [
        9.99487354e-01, 9.97439075e-01, 9.93351729e-01, 9.87243671e-01,
        9.79142267e-01, 9.69083707e-01, 9.57112752e-01, 9.43282436e-01,
        9.27653721e-01, 9.10295118e-01, 8.91282272e-01, 8.70697525e-01,
        8.48629454e-01, 8.25172392e-01, 8.00425940e-01, 7.74494456e-01,
        7.47486541e-01, 7.19514510e-01, 6.90693856e-01, 6.61142710e-01,
        6.30981290e-01, 6.00331359e-01, 5.69315683e-01, 5.38057501e-01,
        5.06680017e-01, 4.75305910e-01, 4.44056878e-01, 4.13072607e-01,
        3.82705404e-01, 3.53052923e-01, 3.23989978e-01, 2.95615816e-01,
        2.68026902e-01, 2.41317098e-01, 2.15577831e-01, 1.90898242e-01,
        1.67365274e-01, 1.45063694e-01, 1.24076009e-01, 1.04482289e-01,
        8.63598679e-02, 6.97829558e-02, 5.48221468e-02, 4.15438626e-02,
        3.00097450e-02, 2.02760315e-02, 1.23929411e-02, 6.40410238e-03,
        2.34605087e-03, 2.51728808e-04, 1.69716651e-04, 2.14049200e-03,
        6.17272116e-03, 1.22625883e-02, 2.03974578e-02, 3.05555983e-02,
        4.27059001e-02, 5.68076133e-02, 7.28101347e-02, 9.06528732e-02,
        1.10265218e-01, 1.31566637e-01, 1.54466915e-01, 1.78866548e-01,
        2.04657293e-01, 2.31722870e-01, 2.59939792e-01, 2.89178315e-01,
        3.19303477e-01, 3.50176189e-01, 3.81654355e-01, 4.13366640e-01,
        4.45160724e-01, 4.77136130e-01, 5.09177959e-01, 5.41152557e-01,
        5.72927506e-01, 6.04372198e-01, 6.35358366e-01, 6.65760589e-01,
        6.95456761e-01, 7.24328534e-01, 7.52261730e-01, 7.79146733e-01,
        8.04878848e-01, 8.29358647e-01, 8.52492271e-01, 8.74191715e-01,
        8.94375078e-01, 9.12966777e-01, 9.29897734e-01, 9.45105522e-01,
        9.58534483e-01, 9.70135818e-01, 9.79867643e-01, 9.87695029e-01,
        9.93590026e-01, 9.97531669e-01, 9.99505985e-01
    ]
    airfoilCtrlY = [
        -2.92846989e-06, -1.48879792e-05, -3.97901255e-05, -7.96028289e-05,
        -1.37279113e-04, -2.16750303e-04, -3.22904145e-04, -4.61538428e-04,
        -6.39280157e-04, -8.63460810e-04, -1.14193979e-03, -1.48287067e-03,
        -1.89440828e-03, -2.38435876e-03, -2.95977922e-03, -3.62653797e-03,
        -4.38885118e-03, -5.24881473e-03, -6.20595374e-03, -7.25681319e-03,
        -8.39461378e-03, -9.60899575e-03, -1.08858707e-02, -1.22073975e-02,
        -1.35520917e-02, -1.48950746e-02, -1.62084560e-02, -1.74622279e-02,
        -1.86967876e-02, -2.00000284e-02, -2.13791408e-02, -2.27831156e-02,
        -2.41592161e-02, -2.54540245e-02, -2.66144295e-02, -2.75885278e-02,
        -2.83264141e-02, -2.87808465e-02, -2.89077765e-02, -2.86667455e-02,
        -2.80211504e-02, -2.69383930e-02, -2.53899280e-02, -2.33512307e-02,
        -2.08017066e-02, -1.77245670e-02, -1.41066928e-02, -9.93850949e-03,
        -5.21389073e-03, 5.03299198e-05, 5.71574692e-03, 1.16362856e-02,
        1.77722330e-02, 2.40920088e-02, 3.05532847e-02, 3.71039497e-02,
        4.36833282e-02, 5.02236151e-02, 5.66514848e-02, 6.28898263e-02,
        6.88595547e-02, 7.44814475e-02, 7.96779547e-02, 8.43749337e-02,
        8.85032651e-02, 9.20003112e-02, 9.48111842e-02, 9.68898020e-02,
        9.81997175e-02, 9.87147137e-02, 9.84191706e-02, 9.73683512e-02,
        9.57110074e-02, 9.35482436e-02, 9.09075876e-02, 8.78208088e-02,
        8.43238333e-02, 8.04562510e-02, 7.62608441e-02, 7.17831431e-02,
        6.70710120e-02, 6.21742582e-02, 5.71442601e-02, 5.20336029e-02,
        4.68957094e-02, 4.17844555e-02, 3.67537594e-02, 3.18571362e-02,
        2.71472132e-02, 2.26752050e-02, 1.84903523e-02, 1.46393319e-02,
        1.11656502e-02, 8.10903452e-03, 5.50484175e-03, 3.38350243e-03,
        1.77002009e-03, 6.83544845e-04, 1.37037442e-04
    ]
    # # NACA2412
    # airfoilX = [1.00000000e+00, 9.98983961e-01, 9.95940183e-01,
    #     9.90881655e-01, 9.83829939e-01, 9.74815051e-01,
    #     9.63875290e-01, 9.51057023e-01, 9.36414438e-01,
    #     9.20009251e-01, 9.01910391e-01, 8.82193653e-01,
    #     8.60941320e-01, 8.38241778e-01, 8.14189094e-01,
    #     7.88882584e-01, 7.62426369e-01, 7.34928905e-01,
    #     7.06502508e-01, 6.77262862e-01, 6.47328517e-01,
    #     6.16820381e-01, 5.85861206e-01, 5.54575073e-01,
    #     5.23086880e-01, 4.91521839e-01, 4.60004979e-01,
    #     4.28660672e-01, 3.97631556e-01, 3.67253779e-01,
    #     3.37414109e-01, 3.08224091e-01, 2.79792467e-01,
    #     2.52224996e-01, 2.25624298e-01, 2.00089704e-01,
    #     1.75717105e-01, 1.52598783e-01, 1.30823213e-01,
    #     1.10474834e-01, 9.16337794e-02, 7.43755645e-02,
    #     5.87707351e-02, 4.48844859e-02, 3.27762555e-02,
    #     2.24993132e-02, 1.41003523e-02, 7.61910468e-03,
    #     3.08799469e-03, 5.31844686e-04, -2.83870691e-05,
    #     1.44008274e-03, 4.94200664e-03, 1.04698608e-02,
    #     1.80077131e-02, 2.75311237e-02, 3.90070567e-02,
    #     5.23938162e-02, 6.76410223e-02, 8.46896392e-02,
    #     1.03472072e-01, 1.23912338e-01, 1.45926329e-01,
    #     1.69422161e-01, 1.94300609e-01, 2.20455639e-01,
    #     2.47775004e-01, 2.76140921e-01, 3.05430784e-01,
    #     3.35517928e-01, 3.66272407e-01, 3.97561776e-01,
    #     4.29024490e-01, 4.60745064e-01, 4.92612197e-01,
    #     5.24495035e-01, 5.56263127e-01, 5.87786972e-01,
    #     6.18938555e-01, 6.49591858e-01, 6.79623359e-01,
    #     7.08912505e-01, 7.37342170e-01, 7.64799099e-01,
    #     7.91174326e-01, 8.16363574e-01, 8.40267633e-01,
    #     8.62792718e-01, 8.83850790e-01, 9.03359866e-01,
    #     9.21244282e-01, 9.37434939e-01, 9.51869515e-01,
    #     9.64492643e-01, 9.75256067e-01, 9.84118763e-01,
    #     9.91047043e-01, 9.96014629e-01, 9.99002715e-01,
    #     1.00000000e+00]

    # airfoilY = [1.32931687e-17, -7.38342600e-05, -2.94838388e-04,
    #     -6.61534665e-04, -1.17151910e-03, -1.82154316e-03,
    #     -2.60761565e-03, -3.52511341e-03, -4.56888783e-03,
    #     -5.73335407e-03, -7.01255010e-03, -8.40015486e-03,
    #     -9.88945762e-03, -1.14732744e-02, -1.31438115e-02,
    #     -1.48924812e-02, -1.67096794e-02, -1.85845380e-02,
    #     -2.05046708e-02, -2.24559320e-02, -2.44222083e-02,
    #     -2.63852669e-02, -2.83246777e-02, -3.02178266e-02,
    #     -3.20400333e-02, -3.37647782e-02, -3.53640411e-02,
    #     -3.68087440e-02, -3.80696794e-02, -3.91912332e-02,
    #     -4.01948507e-02, -4.10450021e-02, -4.17065851e-02,
    #     -4.21457179e-02, -4.23304737e-02, -4.22315281e-02,
    #     -4.18226899e-02, -4.10812967e-02, -3.99884614e-02,
    #     -3.85291639e-02, -3.66921921e-02, -3.44699408e-02,
    #     -3.18580895e-02, -2.88551809e-02, -2.54621317e-02,
    #     -2.16817065e-02, -1.75179898e-02, -1.29758887e-02,
    #     -8.06069877e-03, -2.77775776e-03, 2.82808768e-03,
    #     8.51222505e-03, 1.42221385e-02, 1.99372801e-02,
    #     2.56298579e-02, 3.12655584e-02, 3.68044732e-02,
    #     4.22021973e-02, 4.74110551e-02, 5.23814122e-02,
    #     5.70630246e-02, 6.14063789e-02, 6.53639803e-02,
    #     6.88915470e-02, 7.19490770e-02, 7.45017605e-02,
    #     7.65207179e-02, 7.79835532e-02, 7.88747205e-02,
    #     7.91857090e-02, 7.89150602e-02, 7.80682354e-02,
    #     7.67163113e-02, 7.49590252e-02, 7.28229631e-02,
    #     7.03373443e-02, 6.75339270e-02, 6.44465399e-02,
    #     6.11106600e-02, 5.75630455e-02, 5.38414246e-02,
    #     4.99842364e-02, 4.60304159e-02, 4.20192105e-02,
    #     3.79900151e-02, 3.39822078e-02, 3.00349748e-02,
    #     2.61871084e-02, 2.24767697e-02, 1.89412087e-02,
    #     1.56164396e-02, 1.25368744e-02, 9.73492038e-03,
    #     7.24055476e-03, 5.08088827e-03, 3.27973662e-03,
    #     1.85721674e-03, 8.29386282e-04, 2.07943232e-04,
    #     -1.32931687e-17]
    # airfoilCtrlX = [9.99491981e-01, 9.97462072e-01, 9.93410919e-01,
    #     9.87355797e-01, 9.79322495e-01, 9.69345170e-01,
    #     9.57466157e-01, 9.43735731e-01, 9.28211845e-01,
    #     9.10959821e-01, 8.92052022e-01, 8.71567487e-01,
    #     8.49591549e-01, 8.26215436e-01, 8.01535839e-01,
    #     7.75654476e-01, 7.48677637e-01, 7.20715707e-01,
    #     6.91882685e-01, 6.62295690e-01, 6.32074449e-01,
    #     6.01340793e-01, 5.70218139e-01, 5.38830977e-01,
    #     5.07304360e-01, 4.75763409e-01, 4.44332826e-01,
    #     4.13146114e-01, 3.82442667e-01, 3.52333944e-01,
    #     3.22819100e-01, 2.94008279e-01, 2.66008731e-01,
    #     2.38924647e-01, 2.12857001e-01, 1.87903405e-01,
    #     1.64157944e-01, 1.41710998e-01, 1.20649023e-01,
    #     1.01054307e-01, 8.30046719e-02, 6.65731498e-02,
    #     5.18276105e-02, 3.88303707e-02, 2.76377844e-02,
    #     1.82998328e-02, 1.08597285e-02, 5.35354969e-03,
    #     1.80991969e-03, 2.51728808e-04, 7.05847836e-04,
    #     3.19104469e-03, 7.70593373e-03, 1.42387870e-02,
    #     2.27694184e-02, 3.32690902e-02, 4.57004365e-02,
    #     6.00174193e-02, 7.61653308e-02, 9.40808554e-02,
    #     1.13692205e-01, 1.34919333e-01, 1.57674245e-01,
    #     1.81861385e-01, 2.07378124e-01, 2.34115321e-01,
    #     2.61957963e-01, 2.90785852e-01, 3.20474356e-01,
    #     3.50895168e-01, 3.81917092e-01, 4.13293133e-01,
    #     4.44884777e-01, 4.76678630e-01, 5.08553616e-01,
    #     5.40379081e-01, 5.72025050e-01, 6.03362763e-01,
    #     6.34265207e-01, 6.64607609e-01, 6.94267932e-01,
    #     7.23127337e-01, 7.51070634e-01, 7.77986712e-01,
    #     8.03768950e-01, 8.28315603e-01, 8.51530175e-01,
    #     8.73321754e-01, 8.93605328e-01, 9.12302074e-01,
    #     9.29339610e-01, 9.44652227e-01, 9.58181079e-01,
    #     9.69874355e-01, 9.79687415e-01, 9.87582903e-01,
    #     9.93530836e-01, 9.97508672e-01, 9.99501358e-01]
    # airfoilCtrlY = [-3.69171300e-05, -1.84336324e-04, -4.78186527e-04,
    #     -9.16526881e-04, -1.49653113e-03, -2.21457940e-03,
    #     -3.06636453e-03, -4.04700062e-03, -5.15112095e-03,
    #     -6.37295209e-03, -7.70635248e-03, -9.14480624e-03,
    #     -1.06813660e-02, -1.23085429e-02, -1.40181464e-02,
    #     -1.58010803e-02, -1.76471087e-02, -1.95446044e-02,
    #     -2.14803014e-02, -2.34390701e-02, -2.54037376e-02,
    #     -2.73549723e-02, -2.92712522e-02, -3.11289300e-02,
    #     -3.29024058e-02, -3.45644097e-02, -3.60863926e-02,
    #     -3.74392117e-02, -3.86304563e-02, -3.96930420e-02,
    #     -4.06199264e-02, -4.13757936e-02, -4.19261515e-02,
    #     -4.22380958e-02, -4.22810009e-02, -4.20271090e-02,
    #     -4.14519933e-02, -4.05348791e-02, -3.92588127e-02,
    #     -3.76106780e-02, -3.55810665e-02, -3.31640152e-02,
    #     -3.03566352e-02, -2.71586563e-02, -2.35719191e-02,
    #     -1.95998481e-02, -1.52469393e-02, -1.05182938e-02,
    #     -5.41922826e-03, 2.51649599e-05, 5.67015636e-03,
    #     1.13671818e-02, 1.70797093e-02, 2.27835690e-02,
    #     2.84477082e-02, 3.40350158e-02, 3.95033352e-02,
    #     4.48066262e-02, 4.98962337e-02, 5.47222184e-02,
    #     5.92347017e-02, 6.33851796e-02, 6.71277636e-02,
    #     7.04203120e-02, 7.32254187e-02, 7.55112392e-02,
    #     7.72521355e-02, 7.84291368e-02, 7.90302147e-02,
    #     7.90503846e-02, 7.84916478e-02, 7.73922733e-02,
    #     7.58376682e-02, 7.38909942e-02, 7.15801537e-02,
    #     6.89356356e-02, 6.59902334e-02, 6.27785999e-02,
    #     5.93368528e-02, 5.57022351e-02, 5.19128305e-02,
    #     4.80073261e-02, 4.40248132e-02, 4.00046128e-02,
    #     3.59861114e-02, 3.20085913e-02, 2.81110416e-02,
    #     2.43319390e-02, 2.07089892e-02, 1.72788242e-02,
    #     1.40766570e-02, 1.11358974e-02, 8.48773757e-03,
    #     6.16072151e-03, 4.18031245e-03, 2.56847668e-03,
    #     1.34330151e-03, 5.18664757e-04, 1.03971616e-04]
    # # --- NACA 0012 ---
    # airfoilX = [1.00000000e+00, 9.98993338e-01, 9.95977406e-01, 9.90964349e-01, 9.83974351e-01, 9.75035559e-01, 9.64183967e-01, 9.51463269e-01, 9.36924689e-01, 9.20626766e-01, 9.02635129e-01, 8.83022222e-01, 8.61867019e-01, 8.39254706e-01, 8.15276334e-01, 7.90028455e-01, 7.63612734e-01, 7.36135537e-01, 7.07707507e-01, 6.78443111e-01, 6.48460188e-01, 6.17879468e-01, 5.86824089e-01, 5.55419100e-01, 5.23790958e-01, 4.92067018e-01, 4.60375022e-01, 4.28842581e-01, 3.97596666e-01, 3.66763093e-01, 3.36466018e-01, 3.06827437e-01, 2.77966694e-01, 2.50000000e-01, 2.23039968e-01, 1.97195156e-01, 1.72569633e-01, 1.49262556e-01, 1.27367775e-01, 1.06973453e-01, 8.81617093e-02, 7.10082934e-02, 5.55822757e-02, 4.19457713e-02, 3.01536896e-02, 2.02535132e-02, 1.22851066e-02, 6.28055566e-03, 2.26403871e-03, 2.51728808e-04, 2.51728808e-04, 2.26403871e-03, 6.28055566e-03, 1.22851066e-02, 2.02535132e-02, 3.01536896e-02, 4.19457713e-02, 5.55822757e-02, 7.10082934e-02, 8.81617093e-02, 1.06973453e-01, 1.27367775e-01, 1.49262556e-01, 1.72569633e-01, 1.97195156e-01, 2.23039968e-01, 2.50000000e-01, 2.77966694e-01, 3.06827437e-01, 3.36466018e-01, 3.66763093e-01, 3.97596666e-01, 4.28842581e-01, 4.60375022e-01, 4.92067018e-01, 5.23790958e-01, 5.55419100e-01, 5.86824089e-01, 6.17879468e-01, 6.48460188e-01, 6.78443111e-01, 7.07707507e-01, 7.36135537e-01, 7.63612734e-01, 7.90028455e-01, 8.15276334e-01, 8.39254706e-01, 8.61867019e-01, 8.83022222e-01, 9.02635129e-01, 9.20626766e-01, 9.36924689e-01, 9.51463269e-01, 9.64183967e-01, 9.75035559e-01, 9.83974351e-01, 9.90964349e-01, 9.95977406e-01, 9.98993338e-01, 1.00000000e+00]
    # airfoilY = [1.33226763e-17, -1.41200438e-04, -5.63343432e-04, -1.26208774e-03,
    #     -2.23030811e-03, -3.45825298e-03, -4.93375074e-03, -6.64245132e-03,
    #     -8.56808791e-03, -1.06927428e-02, -1.29971013e-02, -1.54606806e-02,
    #     -1.80620201e-02, -2.07788284e-02, -2.35880799e-02, -2.64660655e-02,
    #     -2.93884006e-02, -3.23300020e-02, -3.52650469e-02, -3.81669313e-02,
    #     -4.10082448e-02, -4.37607812e-02, -4.63956016e-02, -4.88831639e-02,
    #     -5.11935307e-02, -5.32966591e-02, -5.51627743e-02, -5.67628192e-02,
    #     -5.80689678e-02, -5.90551853e-02, -5.96978089e-02, -5.99761253e-02,
    #     -5.98729133e-02, -5.93749219e-02, -5.84732545e-02, -5.71636340e-02,
    #     -5.54465251e-02, -5.33271006e-02, -5.08150416e-02, -4.79241724e-02,
    #     -4.46719377e-02, -4.10787401e-02, -3.71671601e-02, -3.29610920e-02,
    #     -2.84848303e-02, -2.37621474e-02, -1.88154040e-02, -1.36647314e-02,
    #     -8.32732576e-03, -2.81688492e-03, 2.81688492e-03, 8.32732576e-03,
    #     1.36647314e-02, 1.88154040e-02, 2.37621474e-02, 2.84848303e-02,
    #     3.29610920e-02, 3.71671601e-02, 4.10787401e-02, 4.46719377e-02,
    #     4.79241724e-02, 5.08150416e-02, 5.33271006e-02, 5.54465251e-02,
    #     5.71636340e-02, 5.84732545e-02, 5.93749219e-02, 5.98729133e-02,
    #     5.99761253e-02, 5.96978089e-02, 5.90551853e-02, 5.80689678e-02,
    #     5.67628192e-02, 5.51627743e-02, 5.32966591e-02, 5.11935307e-02,
    #     4.88831639e-02, 4.63956016e-02, 4.37607812e-02, 4.10082448e-02,
    #     3.81669313e-02, 3.52650469e-02, 3.23300020e-02, 2.93884006e-02,
    #     2.64660655e-02, 2.35880799e-02, 2.07788284e-02, 1.80620201e-02,
    #     1.54606806e-02, 1.29971013e-02, 1.06927428e-02, 8.56808791e-03,
    #     6.64245132e-03, 4.93375074e-03, 3.45825298e-03, 2.23030811e-03,
    #     1.26208774e-03, 5.63343432e-04, 1.41200438e-04, -1.33226763e-17]
    # airfoilCtrlX = [9.99496669e-01, 9.97485372e-01, 9.93470878e-01, 9.87469350e-01,
    #     9.79504955e-01, 9.69609763e-01, 9.57823618e-01, 9.44193979e-01,
    #     9.28775727e-01, 9.11630948e-01, 8.92828675e-01, 8.72444620e-01,
    #     8.50560862e-01, 8.27265520e-01, 8.02652394e-01, 7.76820594e-01,
    #     7.49874136e-01, 7.21921522e-01, 6.93075309e-01, 6.63451649e-01,
    #     6.33169828e-01, 6.02351778e-01, 5.71121594e-01, 5.39605029e-01,
    #     5.07928988e-01, 4.76221020e-01, 4.44608801e-01, 4.13219623e-01,
    #     3.82179880e-01, 3.51614556e-01, 3.21646728e-01, 2.92397065e-01,
    #     2.63983347e-01, 2.36519984e-01, 2.10117562e-01, 1.84882395e-01,
    #     1.60916095e-01, 1.38315166e-01, 1.17170614e-01, 9.75675810e-02,
    #     7.95850013e-02, 6.32952845e-02, 4.87640235e-02, 3.60497304e-02,
    #     2.52036014e-02, 1.62693099e-02, 9.28283111e-03, 4.27229719e-03,
    #     1.25788376e-03, 2.51728808e-04, 1.25788376e-03, 4.27229719e-03,
    #     9.28283111e-03, 1.62693099e-02, 2.52036014e-02, 3.60497304e-02,
    #     4.87640235e-02, 6.32952845e-02, 7.95850013e-02, 9.75675810e-02,
    #     1.17170614e-01, 1.38315166e-01, 1.60916095e-01, 1.84882395e-01,
    #     2.10117562e-01, 2.36519984e-01, 2.63983347e-01, 2.92397065e-01,
    #     3.21646728e-01, 3.51614556e-01, 3.82179880e-01, 4.13219623e-01,
    #     4.44608801e-01, 4.76221020e-01, 5.07928988e-01, 5.39605029e-01,
    #     5.71121594e-01, 6.02351778e-01, 6.33169828e-01, 6.63451649e-01,
    #     6.93075309e-01, 7.21921522e-01, 7.49874136e-01, 7.76820594e-01,
    #     8.02652394e-01, 8.27265520e-01, 8.50560862e-01, 8.72444620e-01,
    #     8.92828675e-01, 9.11630948e-01, 9.28775727e-01, 9.44193979e-01,
    #     9.57823618e-01, 9.69609763e-01, 9.79504955e-01, 9.87469350e-01,
    #     9.93470878e-01, 9.97485372e-01, 9.99496669e-01]
    # airfoilCtrlY = [-7.06002188e-05, -3.52271935e-04, -9.12715585e-04, -1.74619792e-03,
    #     -2.84428054e-03, -4.19600186e-03, -5.78810103e-03, -7.60526962e-03,
    #     -9.63041534e-03, -1.18449221e-02, -1.42288910e-02, -1.67613504e-02,
    #     -1.94204243e-02, -2.21834542e-02, -2.50270727e-02, -2.79272331e-02,
    #     -3.08592013e-02, -3.37975244e-02, -3.67159891e-02, -3.95875880e-02,
    #     -4.23845130e-02, -4.50781914e-02, -4.76393828e-02, -5.00383473e-02,
    #     -5.22450949e-02, -5.42297167e-02, -5.59627967e-02, -5.74158935e-02,
    #     -5.85620766e-02, -5.93764971e-02, -5.98369671e-02, -5.99245193e-02,
    #     -5.96239176e-02, -5.89240882e-02, -5.78184443e-02, -5.63050796e-02,
    #     -5.43868129e-02, -5.20710711e-02, -4.93696070e-02, -4.62980550e-02,
    #     -4.28753389e-02, -3.91229501e-02, -3.50641261e-02, -3.07229612e-02,
    #     -2.61234889e-02, -2.12887757e-02, -1.62400677e-02, -1.09960286e-02,
    #     -5.57210534e-03, 0.00000000e+00, 5.57210534e-03, 1.09960286e-02,
    #     1.62400677e-02, 2.12887757e-02, 2.61234889e-02, 3.07229612e-02,
    #     3.50641261e-02, 3.91229501e-02, 4.28753389e-02, 4.62980550e-02,
    #     4.93696070e-02, 5.20710711e-02, 5.43868129e-02, 5.63050796e-02,
    #     5.78184443e-02, 5.89240882e-02, 5.96239176e-02, 5.99245193e-02,
    #     5.98369671e-02, 5.93764971e-02, 5.85620766e-02, 5.74158935e-02,
    #     5.59627967e-02, 5.42297167e-02, 5.22450949e-02, 5.00383473e-02,
    #     4.76393828e-02, 4.50781914e-02, 4.23845130e-02, 3.95875880e-02,
    #     3.67159891e-02, 3.37975244e-02, 3.08592013e-02, 2.79272331e-02,
    #     2.50270727e-02, 2.21834542e-02, 1.94204243e-02, 1.67613504e-02,
    #     1.42288910e-02, 1.18449221e-02, 9.63041534e-03, 7.60526962e-03,
    #     5.78810103e-03, 4.19600186e-03, 2.84428054e-03, 1.74619792e-03,
    #     9.12715585e-04, 3.52271935e-04, 7.06002188e-05]
    # # --- NACA0012 very reduced ---
    # airfoilX = [1., 0.9777864, 0.91311939, 0.8117449, 0.68267051,
    #     0.53736505, 0.38873953, 0.25, 0.13347406, 0.04951557,
    #     0.00558459, 0.00558459, 0.04951557, 0.13347406, 0.25,
    #     0.38873953, 0.53736505, 0.68267051, 0.8117449, 0.91311939,
    #     0.9777864, 1.]
    # airfoilY = [1.33226763e-17, -3.08157579e-03, -1.16594904e-02, -2.39955739e-02,
    #     -3.77554367e-02, -5.02269995e-02, -5.83844746e-02, -5.93749219e-02,
    #     -5.15721044e-02, -3.53992016e-02, -1.29133180e-02, 1.29133180e-02,
    #     3.53992016e-02, 5.15721044e-02, 5.93749219e-02, 5.83844746e-02,
    #     5.02269995e-02, 3.77554367e-02, 2.39955739e-02, 1.16594904e-02,
    #     3.08157579e-03, -1.33226763e-17]
    # airfoilCtrlX = [0.9888932, 0.9454529, 0.86243214, 0.74720771, 0.61001778,
    #     0.46305229, 0.31936977, 0.19173703, 0.09149482, 0.02755008,
    #     0.00558459, 0.02755008, 0.09149482, 0.19173703, 0.31936977,
    #     0.46305229, 0.61001778, 0.74720771, 0.86243214, 0.9454529,
    #     0.9888932]
    # airfoilCtrlY = [-0.00154079, -0.00737053, -0.01782753, -0.03087551, -0.04399122,
    #     -0.05430574, -0.0588797, -0.05547351, -0.04348565, -0.02415626,
    #     0., 0.02415626, 0.04348565, 0.05547351, 0.0588797,
    #     0.05430574, 0.04399122, 0.03087551, 0.01782753, 0.00737053,
    #     0.00154079]
    airfoilXY = copy(hcat(airfoilX, airfoilY)')
    airfoilCtrlXY = copy(hcat(airfoilCtrlX, airfoilCtrlY)')

    # Rotate the airfoil
    alpha = deg2rad(5.0)  #  degrees
    R = [cos(-alpha) -sin(-alpha); sin(-alpha) cos(-alpha)]
    airfoilXY = R * airfoilXY
    airfoilCtrlXY = R * airfoilCtrlXY
    airfoilXY[2, :] .-= depth  # Translate the airfoil in the y-direction
    airfoilCtrlXY[2, :] .-= depth  # Translate the airfoil in the y-direction

    Fr = 1.03
    GRAV = 9.81
    chord = maximum(airfoilXY[1, :]) - minimum(airfoilXY[1, :])
    V = [Fr*√(GRAV*chord), 0.0, 0]
    Vinf = norm(V)
    ν = GRAV * chord / Vinf^2
    μ = 1e-3
    println("h/c: $(depth/chord)")
    println("Fr: $Fr")
    println("Vinf: $Vinf")

    # ==============================================================================
    #                         DIPOLE METHOD
    # ==============================================================================
    println("μ: ", μ)
    println("ν: ", ν)
    Airfoil, Amat, Bvec = LiftingLine.setup_FSDPM(airfoilXY[1, :], airfoilXY[2, :], airfoilCtrlXY, ν, μ, Vinf)
    cℓ, cm, Γi, cpDist = LiftingLine.solve_FSDPM(Airfoil, Amat, Bvec, Vinf)


    # ************************************************
    #     Post processing
    # ************************************************
    Ausman0p6 = [-0.00216618545219191 -0.990936596249487;
        0.0038289357847812178 -0.773024364216712;
        0.001168744798993926 -0.5514723119851871;
        0.009685232561093854 -0.07387713930458373;
        0.00531200011734595 0.08593312694763522;
        0.022632578014763066 0.04414984555123924;
        0.023472335035286257 0.13313213390745549;
        0.05026807527219415 0.24569860266632026;
        0.04765450835055321 0.30562833653459054;
        0.06841583740879167 0.33648115615106305;
        0.09435138985549416 0.4308885996567107;
        0.14451521581572066 0.5379859085314442;
        0.19643346863916142 0.56336249607428;
        0.2440308756077826 0.5669512393488243;
        0.29249475675728054 0.5669072346141244;
        0.3652120569829882 0.49238600219866524;
        0.42752066585449977 0.49959334910344294;
        0.4898491816298043 0.43779341401042665;
        0.5980432037437552 0.38321575600003266;
        0.6984452446204866 0.3395410568103746;
        0.8023110867571679 0.2885992900045604;
        0.9018634171614794 0.19044594345758314;
        0.9018634171614794 0.19044594345758314;
        0.9962640504105666 -0.049349214502388716;
        0.9002006668288902 -0.04563002862141252;
        0.8032702852004479 -0.03646211362598728;
        0.6976908777399168 -0.045446151694273684;
        0.6024897774120028 -0.030831864980900958;
        0.4960438957705948 -0.03618315503994329;
        0.4311353404905637 -0.03067627681178342;
        0.369686395543368 -0.017908617358138867;
        0.2961178132475729 0.007581911015616227;
        0.24418803537456799 0.022156907787292623;
        0.20263394530473403 0.06941013486665382;
        0.14809950623020604 0.11304318671811386;
        0.10222666576910974 0.1312446451083631;
        0.0736482099109663 0.19846187736247145;
    ]
    Ausman0p9 = [
        -0.001732351667388471 -0.9632597914499772;
        0.0051970550021654405 -0.8528198807787206;
        0.0017323516673884987 -0.3165553712121627;
        0.01126028583802513 -0.10256010278486993;
        0.012992637505413629 0.10226506475636188;
        0.013858813339107878 0.12067304553503311;
        0.025985275010827202 0.32775186504402387;
        0.050238198354265906 0.5715988021057342;
        0.07449112169770467 0.6106125976254423;
        0.10134257254222606 0.7117548233983098;
        0.14898224339540928 0.8151028571343135;
        0.19835426591598093 0.8309860699190291;
        0.2477262884365527 0.828457314924688;
        0.2962321351234301 0.8305355388754966;
        0.3663923776526636 0.7381527529490097;
        0.42702468601126026 0.7102557112534581;
        0.49631875270679965 0.6408918720511771;
        0.5967951494153314 0.5621785169454993;
        0.6946730186227805 0.47657263492383095;
        0.801212646167172 0.37941940303639976;
        0.8973581637072328 0.22247511987165836;
        0.9987007362494589 -0.01964843627353008;
        0.8990905153746211 -0.019189931229226964;
        0.797747942832395 -0.025627940101299762;
        0.6972715461238632 -0.022863952084229533;
        0.5993936769164141 -0.02241342104069699;
        0.49458640103941115 -0.03343847385599252;
        0.4278908618449547 -0.03313147482632872;
        0.3611953226504981 -0.016714003989990678;
        0.2962321351234301 -0.011811987016333125;
        0.2442615851017757 0.0068392007858377735;
        0.19662191424859254 0.04618391733752092;
        0.14724989172802078 0.05331566427662593;
        0.10047639670853187 0.07194293007648556;
        0.07362494586401042 0.0996844787570128;
        0.052836725855348654 0.12049363051769713;]
    GiesingAndSmith0p6 = [
        -0.004708411749055452 -0.5001084461303075;
        0.0006980920828783543 -0.4943491352382171;
        0.0007020932823233739 -0.5082192931144204;
        0.0007060944817684212 -0.5220894509906234;
        0.001911953012036005 -0.536108080765242;
        0.005517744668935981 -0.5373149632152022;
        -0.00489693796078558 -0.5409306607341583;
        0.000714096880658488 -0.54982976674303;
        0.0007180980801035075 -0.5636999246192332;
        -0.004686785431393892 -0.5750760763041664;
        0.0019239566103710914 -0.5777185543938518;
        0.005829565416595378 -0.5767190936271055;
        0.0007261004789935743 -0.5914402403716398;
        0.0007301016784386216 -0.605310398247843;
        0.0009597646806751847 -0.6202953269887456;
        0.00043779760382980926 -0.6335608149658619;
        0.0004417988032748288 -0.647430972842065;
        0.0004458000027198761 -0.6613011307182682;
        0.0004492858961757662 -0.6733849803831425;
        0.0006925709472805075 -0.47521011868826146;
        0.0012877233829144408 -0.4552610541142962;
        0.0014208661924194899 -0.4360979149484553;
        0.0018768870102880975 -0.41455178570614326;
        0.002009909903725876 -0.3949729574930637;
        0.00216564889750942 -0.3740223764094076;
        0.002599345340852044 -0.3552059175919856;
        0.0029555220424991313 -0.33472432325097345;
        0.004395616079121117 -0.317583080901198;
        0.0033449114975815653 -0.2958462141615653;
        0.0037786639017555745 -0.2772237435661883;
        0.003935179352074797 -0.2589647490634036;
        0.004488254238999456 -0.24194278922435675;
        0.003925085417111174 -0.2239741235120728;
        0.006054469199946594 -0.21067331812962753;
        0.004516899189571821 -0.1924515830478989;
        0.005553041963340916 -0.1789703459548152;
        0.00581001410112536 -0.16177029476704285;
        0.006144464009981754 -0.1460898879363277;
        0.0057005267344931165 -0.12940650211955107;
        0.0068553534379545145 -0.11531223160625048;
        0.011399425183722861 -0.09571064073450353;
        0.01500477042580875 -0.09537002623133384;
        0.005268627566518913 -0.09035952422629245;
        0.008187999680388663 -0.07287737448743759;
        0.004547607695802064 -0.0585512757374107;
        0.009824999496972436 -0.053880952631340295;
        0.013491498608427493 -0.05722572802650949;
        0.00807826981682036 -0.03967296621108496;
        0.008773508532511659 -0.019498827672882912;
        0.00976859470964428 0.0027765771284181673;
        0.011306007097010545 0.019818658583765192;
        0.01276359858666018 0.035820550947477714;
        0.012709718798678854 0.049007754093407185;
        0.013580033169944467 0.06473141679514882;
        0.015420996208482057 0.08192939578625236;
        0.017054112032473284 0.09124473371239294;
        0.014860666621553581 0.10658795654791775;
        0.018551288115723402 0.10889629787320432;
        0.01904880089217273 0.12013907722746253;
        0.013490455514989202 0.13018824208577806;
        0.018407487432638342 0.14448248074344616;
        0.021455121496077395 0.1604049436997126;
        0.022536886924535343 0.17978457845309115;
        0.02459571824463111 0.19463358127413433;
        0.0260338038869862 0.20881766052770556;
        0.02753298056995887 0.2195341457504154;
        0.02904490999358697 0.23563956460295055;
        0.03283026551401105 0.25672656770090174;
        0.03574509656185956 0.27392689985868157;
        0.04412502496569684 0.27100086069612006;
        0.03833083791297329 0.28357135477217743;
        0.039828029910084856 0.30116775353234504;
        0.04354374447422418 0.3194986867331584;
        0.04562877385753947 0.3289903591043985;
        0.04882847798671894 0.3467892363648533;
        0.054432983814674624 0.3605229556133897;
        0.05942211144692297 0.374823898470752;
        0.06376959851525335 0.38916814751686296;
        0.06649850980481739 0.4045812630324839;
        0.07183823527881197 0.41310124381674823;
        0.07684697561493539 0.42943306743734627;
        0.0824758074997152 0.43895762542432804;
        0.0871437251644234 0.45129285332981656;
        0.09235809096794487 0.47214708261991906;
        0.09531862803896998 0.48580057389278564;
        0.10136626243421012 0.4909954352957272;
        0.1079729536748841 0.5023950591881483;
        0.1145800086609621 0.5125337596372782;
        0.12118666683387205 0.5240480129336349;
        0.12756245138976838 0.5347145124155346;
        0.1304255912622068 0.5441623697874904;
        0.13680294869901657 0.5493764775829628;
        0.14368103479472014 0.5573560787298251;
        0.15001611899554537 0.5729112640430581;
        0.1566232070493874 0.5829353350882525;
        0.16379151788720578 0.5916559426651468;
        0.1692759937942339 0.5961850321746874;
        0.17644612459911393 0.6072760780270572;
        0.18305438655857822 0.6132308052325389;
        0.18966283039074458 0.618555070716375;
        0.19627150569725907 0.6230769303726623;
        0.2028803794103575 0.6269110136053362;
        0.2093481025854579 0.6240803456896866;
        0.21609830870925653 0.6339487183490382;
        0.2227074138967031 0.6369803957541632;
        0.22931658521967765 0.6397828143514169;
        0.23592578961041624 0.6424706035447352;
        0.24253499400115483 0.6451583927380533;
        0.24956140546752945 0.6437059191843945;
        0.25528863903368804 0.6414914091838556;
        0.2623644755020369 0.6467451989956485;
        0.2689743577819377 0.6470830854082877;
        0.2755843392651305 0.6470770836091202;
        0.2821943207483233 0.6470710818099527;
        0.288804302231516 0.6470650800107851;
        0.2954142837147088 0.6470590782116176;
        0.3021510974769607 0.6459247668856934;
        0.30863439548603233 0.6465312422955725;
        0.3152445753758092 0.6458374640727915;
        0.3218547387317041 0.6452010005519784;
        0.3284649847570088 0.6442779635213263;
        0.3350753134517237 0.6430683529808353;
        0.3416856917480846 0.641686798334441;
        0.34871224693204905 0.6397361277559852;
        0.3543432443844189 0.6384155325786628;
        0.3615171738486893 0.6363385256539347;
        0.3681278166871623 0.6340399357760558;
        0.3747385752628092 0.6313401429844026;
        0.3813495322450403 0.6279525737691358;
        0.3879605718966813 0.6242784310440302;
        0.3945716942177324 0.6203177148090857;
        0.4012333991951287 0.6145993022456204;
        0.4054716636448553 0.6081227152957439;
        0.4120012331440496 0.6091255250002902;
        0.4186125373378028 0.6045343470437001;
        0.4252239242009659 0.599656595577271;
        0.431835277996365 0.5948934735147776;
        0.4384466152578821 0.590187666154252;
        0.4450579525193993 0.5854818587937263;
        0.4516693393825624 0.5806041073272973;
        0.4582806931779615 0.5758409852648038;
        0.4648918485667767 0.5717656396259236;
        0.4715031196927658 0.5672890910732692;
        0.4781142420138168 0.5633283748383247;
        0.4847253312671038 0.5594822880073158;
        0.4913363213170989 0.5559800893881135;
        0.4979472286976838 0.5527644642787501;
        0.5049744045381419 0.5486622879648884;
        0.5106057123982485 0.5462656643684113;
        0.5177793886874508 0.5450662888175648;
        0.5243900149920417 0.5428250136416537;
        0.5310007239660427 0.5402971649559037;
        0.5376114494739257 0.537712001568186;
        0.5442221915156906 0.5350695234785006;
        0.5508328343541635 0.5327709336006217;
        0.5580078085531032 0.5270723539453044;
        0.563493906916194 0.525977199512998;
        0.5706645479291164 0.5266202550925664;
        0.5772751742337073 0.5243789799166554;
        0.5838862469531123 0.5205902077876141;
        0.5904973362063993 0.5167441209566053;
        0.5971084254596865 0.5128980341255962;
        0.6037193328402715 0.509682409016233;
        0.6108941737347874 0.5044459291455308;
        0.6166902445642032 0.5050711279258608;
        0.6235507818731122 0.5044487657396621;
        0.6301615073809951 0.5018636023519445;
        0.6367724808971081 0.49841871843471;
        0.6433834048115752 0.49514577862337883;
        0.6499943783276882 0.49170089470614436;
        0.6566054675809753 0.4878548078751354;
        0.6624904949015616 0.47932164222089013;
        0.6692269040188065 0.4795900328115459;
        0.6758382743480876 0.47476959604708463;
        0.6824494958724306 0.4704649916003335;
        0.6890607173967738 0.46616038715358227;
        0.6956718893194709 0.46202772681273446;
        0.7022829951066399 0.4581243252797577;
        0.7093619592343346 0.4525360839693119;
        0.7152547610811418 0.4505517845877169;
        0.7221161801970911 0.44687263829656976;
        0.728727219848732 0.4431984955714641;
        0.7353382925681371 0.439409723442423;
        0.7419493983553063 0.4355063219094463;
        0.7485605041424752 0.4316029203764696;
        0.7549414353990775 0.4244283818612188;
        0.7587793029829748 0.4199014456672377;
        0.7653890695257015 0.4206405349936513;
        0.7720004067872186 0.41593472763312567;
        0.7786118267181459 0.41094234676276103;
        0.7852233293184829 0.4056633923825576;
        0.7918349807237581 0.39986860568464405;
        0.7982471465560901 0.39124223002887615;
        0.7998518936065748 0.3831732000536096;
        0.8062606935691867 0.3862146117817646;
        0.8128727748553941 0.3789296428326887;
        0.8194848396077195 0.3717019885855806;
        0.8260968712922807 0.3645889637424081;
        0.8327090756640538 0.3568773186786832;
        0.8383916800644236 0.34290425130302077;
        0.8447314160729504 0.34233394521761834;
        0.85134384457068 0.33384536752721905;
        0.8579564218733475 0.32484095751910974;
        0.8645690983793068 0.3154926592991937;
        0.8711820633096525 0.3051445379449508;
        0.8747541577668416 0.2905586868742819;
        0.8808007740882973 0.2906033469563105;
        0.8874139466107169 0.2795356076773611;
        0.8940273175397204 0.26778009197479824;
        0.9006408538075441 0.2554514292525576;
        0.907154696278176 0.24153715071222592;
        0.9098325206582758 0.23336874101791516;
        0.91627367231034 0.22345563948377556;
        0.9225872194764666 0.2095167064452932;
        0.9279990963476552 0.1966503767607677;
        0.931565087407596 0.18462333633346883;
        0.9361185303214863 0.17173977291133358;
        0.9406297008230577 0.156604599456043;
        0.9457416776918022 0.14181913689467185;
        0.9507334932871119 0.12696373215451662;
        0.9538620766013579 0.11356358335539563;
        0.9586734734658333 0.09921621424310911;
        0.9568874200206303 0.09103184305404255;
        0.96258476717182 0.08050896998326629;
        0.96649501510977 0.06542688062288582;
        0.9704050032295743 0.05124545086485599;
        0.9740147268798129 0.03640831350550067;
        0.9761212947321767 0.024648292733438604;
        0.9772613284008214 0.007362103253010122;
        0.9799695543908458 -0.007000869021506206;
        0.9833432323895819 -0.013658804273805392;
        0.9842486726458151 -0.027799978457872965;
        0.9877576163015471 -0.040457428643188464;
        0.9878125873257411 -0.057427402118991555;
        0.9908710757983594 -0.0708656970639785;
        0.9908749649348263 -0.08434738862684454;
        0.9914814901880749 -0.10382219631355927;
        0.987546419657507 -0.10684483877886286;
        0.9723100075893161 -0.11394948838410623;
        0.9736606328537745 -0.12482140036459721;
        0.9785949278649568 -0.13790391653396272;
        0.9906664049397638 -0.1425160342546622;
        0.991196192542797 -0.15636040030830645;
        0.9819360832938449 -0.15402119494928412;
        0.9859921355458141 -0.1695533369911678;
        0.9919525859545517 -0.17459193378185778;
        0.9873473569539051 -0.18057713706534384;
        0.9915065431527819 -0.19066829847024114;
        0.9910669295510293 -0.1248792358838482;
        0.9668430739933123 -0.09959674782690353;
        0.9608301199667055 -0.08614144158406933;
        0.950010362279899 -0.07426110885060921;
        0.9433981670936156 -0.06658130528797757;
        0.9367862676801104 -0.05992679806054735;
        0.92536736280119 -0.05414770656347334;
        0.9187563669643362 -0.05062544779858247;
        0.9121462201423236 -0.050046298979737;
        0.9055357591765527 -0.04837817082350393;
        0.8989256784900679 -0.04802828081252963;
        0.8920438452655728 -0.043018460003426906;
        0.8857054013799242 -0.04692729787680672;
        0.8790953702950852 -0.0467493519717358;
        0.8722351981094041 -0.04739268945399222;
        0.8664393697769242 -0.04885850386318302;
        0.859264599151407 -0.043865611475844;
        0.852654286990574 -0.04271331563732084;
        0.846043974829741 -0.04156101979879767;
        0.8394337288044362 -0.040637982768145675;
        0.8322945833454409 -0.044265925936606054;
        0.826630254325738 -0.042275256792803195;
        0.819603503278864 -0.03964562743719069;
        0.8131428791261146 -0.03662556594531341;
        0.8063830938976642 -0.038086126885725546;
        0.799772818938066 -0.03706278912662986;
        0.7928935056608981 -0.04078836572743905;
        0.787116977558519 -0.039723595024517055;
        0.779942208999737 -0.03473786697492398;
        0.7733320456438422 -0.034101403454110835;
        0.7667218657540653 -0.033407625231329896;
        0.7601117685336984 -0.03300042051838781;
        0.7535016547794496 -0.03253590110347793;
        0.7466227521855398 -0.03768511121816476;
        0.7405321415463708 -0.032772488554547397;
        0.7336711812456471 -0.03068382524300628;
        0.7270610344236345 -0.030104676424161037;
        0.7204508214660935 -0.029296268797444514;
        0.7138407077118447 -0.028831749382534744;
        0.7075027581472773 -0.034454151283603296;
        0.7002425104395038 -0.03327788168373169;
        0.6937310573758827 -0.03130094742865097;
        0.6879316442492358 -0.031061627915677237;
        0.6807899405340166 -0.025821375884372277;
        0.6741798267797678 -0.025356856469462508;
        0.667569696491637 -0.024835022352584946;
        0.6609595166018601 -0.024141244129803896;
        0.6538352085652114 -0.026689372802823796;
        0.6483037403244738 -0.0270277266666894;
        0.6411291422713498 -0.022633056481139002;
        0.6345189293138089 -0.02182464885442259;
        0.6279088155595601 -0.02136012943951271;
        0.6212986026020193 -0.020551721812796298;
        0.6144563334874821 -0.0253936947852621;
        0.6075146379194395 -0.018282440307505143;
        0.601468682953264 -0.020619688468245245;
        0.5946491756666625 -0.023358233096975378;
        0.5882475874159612 -0.01668162778511706;
        0.5816373579245383 -0.015815905456432855;
        0.5750271284331154 -0.01495018312774865;
        0.5684168162722827 -0.013797887289225375;
        0.5618065702469778 -0.012874850258573267;
        0.555196406891083 -0.01223838673776012;
        0.5487137850307646 -0.015188731801795252;
        0.5419759148404732 -0.010392312676456128;
        0.5353657514845784 -0.009755849155642982;
        0.5287555054592735 -0.008832812124990874;
        0.5221452594339688 -0.007909775094338767;
        0.5155350960780739 -0.00727331157352562;
        0.5089248500527691 -0.006350274542873513;
        0.49833418399791574 -0.007474215462671241;
        0.49209884742125354 -0.004271667965061399;
        0.48548865099759475 -0.003520575040312779;
        0.47887840497228984 -0.0025975380096607825;
        0.472268192014749 -0.0017891303829442595;
        0.4656579955910901 -0.0010380374581955287;
        0.4590477495657854 -0.00011500042754342132;
        0.4524375531421265 0.0006360924972053095;
        0.44570123974735654 0.000035879947788641786;
        0.4392170941592809 0.002367537154573829;
        0.432606947337268 0.002946685973419183;
        0.4259967013119632 0.0038697230040711794;
        0.41938647182054045 0.004735445332755495;
        0.4127763084646455 0.005371908853568641;
        0.40616607897322266 0.006237631182252956;
        0.39904168590518074 0.0039842638336388125;
        0.3933269078432773 0.0018420912546825985;
        0.3863354566344822 0.008605539360434622;
        0.37930979882242144 0.007445470235386331;
        0.37273714389984125 0.0045596571220771764;
        0.3669218321844604 0.007558331240711369;
        0.35989472054149285 0.011437966953526169;
        0.35328452411783406 0.012189059878274788;
        0.3466744764991133 0.0124243204853135;
        0.3400638832622863 0.014550966257288911;
        0.33345347189816144 0.01604715030761883;
        0.3268431101356827 0.01737139025204537;
        0.32023263263602975 0.019096833110246347;
        0.31362172525544463 0.0223124582196097;
        0.30701108452963455 0.02460372455223714;
        0.29993390119288643 0.024018854045741778;
        0.29401601373269626 0.026168891273676498;
        0.2889808964917331 0.034901076968397615;
        0.2823699271090906 0.03833163221014024;
        0.27575897012685957 0.04171920142540686;
        0.26914793047521846 0.04539334415051255;
        0.2625369238913414 0.04895285747168254;
        0.25700764323081704 0.05219042634924198;
        0.2534482269773126 0.04886534663100006;
        0.24811302977293345 0.05630223415810609;
        0.24150187438411838 0.06037757979698599;
        0.23489081819859534 0.06410903722405947;
        0.2282796132081342 0.06835632696884286;
        0.2216683090143811 0.072947504925433;
        0.21343859086798886 0.07957261936208937;
        0.2072434559308168 0.08362113432598783;
        0.2011932389432076 0.08737882937041375;
        0.19342044423543894 0.09105657427727865;
        0.1868087186114045 0.09710864030402533;
        0.18019723254494938 0.10233027998226096;
        0.17302189852740266 0.10927617083915231;
        0.1675041431254181 0.11391600049729078;
        0.16036302235381397 0.11713547848745143;
        0.15375079226266855 0.12493627975423738;
        0.14713952113667947 0.12941282830689183;
        0.14063604766321078 0.12780498781481242;
        0.13518877166497137 0.1408503336154977;
        0.1296672641705572 0.149817437586516;
        0.12249517478957148 0.15419514989932537;
        0.11588332497771223 0.16067771302085232;
        0.10927139249644294 0.16744684965221812;
        0.1024282971763524 0.17437081986563652;
        0.09656168638655735 0.18412058822138178;
        0.08943465262136088 0.19102119755847913;
        0.082704915495972 0.18955113649058664;
        0.08041726406700764 0.20412388073964383;
        0.07560682148308859 0.21516323630335665;
        0.06820758623956272 0.2271354978447061;
        0.06110461256583202 0.2336102403605551;
        0.0561459397340566 0.2514327477508983;
        0.049028268397644265 0.24937148623469096;
    ]

    GiesingAndSmith0p9 = [
        0.0019269418604419275 -0.8369435970148367;
        0.0035718110991585506 -0.8197160215568753;
        0.0031713211975579747 -0.802289007658272;
        0.0031713211975579747 -0.784730820643164;
        0.0031713211975579747 -0.767172633628056;
        0.0031713211975579747 -0.7496144466129475;
        0.0031713211975579747 -0.7320562595978395;
        0.0031713211975579747 -0.714498072582731;
        0.0031713211975579747 -0.696939885567623;
        0.002870953771357543 -0.6792206960817164;
        0.004372790902359647 -0.6625473313615797;
        0.004673158328560079 -0.6486255069743929;
        0.0032313946827980666 -0.6219454208799522;
        0.003952276505679059 -0.5924722675587217;
        0.004509321550632578 -0.562212353103644;
        0.005273893180960915 -0.534536333799775;
        0.006114921974322091 -0.5113155122240067;
        0.002785134506728848 -0.5029999513774281;
        0.006675607836562875 -0.4877209538407552;
        0.0067757303119630186 -0.4696311614154094;
        0.0067757303119630186 -0.45273805724178273;
        0.006475362885762614 -0.43342266893632053;
        0.006835803797203138 -0.40940153580862515;
        0.008037273502004838 -0.38243130611351384;
        0.007827016303664502 -0.3645529115223043;
        0.008202475586415015 -0.34415322359591727;
        0.00875815532488583 -0.32297621829830314;
        0.012949949628305035 -0.32311966189079255;
        0.01248271140977103 -0.3035197368013851;
        0.009408362459249087 -0.2931302932708211;
        0.012783078835971462 -0.2885567554568975;
        0.011624518763484137 -0.27022363973967956;
        0.011881976557370222 -0.2552682345780499;
        0.011815228240436793 -0.23949135617512662;
        0.016462579806926614 -0.22462597696179354;
        0.011844430629095126 -0.21304890528758946;
        0.01683803908967718 -0.1934685740744453;
        0.010880751803368782 -0.18587250999299587;
        0.017889325081378638 -0.16494135923584596;
        0.010079772000167686 -0.16304317996977313;
        0.018339876220679285 -0.14672016326252124;
        0.011681731606569934 -0.1434527793812861;
        0.018890549835380077 -0.1269564359599078;
        0.01018578403294429 -0.12539215461044284;
        0.022169560904734653 -0.10713210806005735;
        0.010079772000167686 -0.10318572423644978;
        0.009196338393695824 -0.08027151003564659;
        0.022334762989144885 -0.08185306967921435;
        0.02509814331018878 -0.05606722607550996;
        0.009307398618509433 -0.05290190619200397;
        0.008577934869165582 -0.032946063231800604;
        0.00827756744296515 -0.016823072565448793;
        0.028273456101450378 -0.021027225676917283;
        0.026299613014990453 -0.037132147223838396;
        0.02968269876272156 0.00505553721886054;
        0.00784066936849176 0.0076878458022222596;
        0.007161917002792145 0.03535496132371918;
        0.034609778473202124 0.029983965774216692;
        0.037713575210606476 0.05329191733078287;
        0.043420556308414515 0.07064647973347404;
        0.049866903378408156 0.08136045315069085;
        0.05633635563503264 0.09380446578856305;
        0.06294443901144192 0.0976355601197032;
        0.06967899288309365 0.1004485395055732;
        0.07625071599212069 0.10413924339208136;
        0.0826185054275696 0.09619525667130668;
        0.0893767725170791 0.09978150808483188;
        0.09598485589348837 0.09957143238868493;
        0.10264300050759773 0.09890593319942687;
        0.1093512063594072 0.0948085488654411;
        0.11580910602271621 0.09512691201715073;
        0.12241718939912549 0.0900176806367412;
        0.12902527277553477 0.08411034984655386;
        0.13471458520121338 0.07529653265716885;
        0.14164070467595247 0.07672427744354171;
        0.14839182016007624 0.07511658148999478;
        0.15429368250464526 0.07008625041359107;
        0.1584612805431761 0.060273722152577314;
        0.16506936391958538 0.061597655711587374;
        0.1714897176546194 0.05354629687564971;
        0.178285530672404 0.0544990360980806;
        0.18540852963658544 0.04924232558177222;
        0.19123136674164215 0.05483642323834448;
        0.19350414693322537 0.042554233105754236;
        0.1993112505064335 0.04438973501093235;
        0.20631982378444336 0.04295814030611256;
        0.21853476578326053 0.032547421653692066;
        0.2251428491596698 0.02794607171586816;
        0.2317509325360791 0.02379455962719179;
        0.23895975076488923 0.02212302797063903;
        0.2455678341412985 0.013413711806730344;
        0.25255532900343464 0.007196850797008381;
        0.25878400089411707 0.0007283963247801672;
        0.26539208427052635 -0.0025669727606805193;
        0.27200016764693563 -0.0041935885347879065;
        0.2786082510233449 -0.005457431849905214;
        0.28577952332387996 -0.007136521191204448;
        0.29036549027747566 -0.01517018196009956;
        0.29843250115257275 -0.01716546301143551;
        0.3052808784699424 -0.006740110214731088;
        0.3116486679053913 -0.01649269039236012;
        0.3182567512818006 -0.016813325314103844;
        0.32486483465820987 -0.016843742268655815;
        0.33147291803461915 -0.017889922108379253;
        0.3380810014110284 -0.02016952830866825;
        0.35163090974851413 -0.02234672750333866;
        0.35790525154025626 -0.021687684177684075;
        0.3650282505044377 -0.028769362268298293;
        0.3708510876094944 -0.02564030593840183;
        0.3777295016694841 -0.028840905576342113;
        0.3843375850458935 -0.029088986006287954;
        0.3914605840100747 -0.031288042939682015;
        0.3969905628745862 -0.03238950641353244;
        0.40416183517512133 -0.028382137460166046;
        0.4107699185515306 -0.02826744543112203;
        0.417908062091823 -0.028946179489049984;
        0.42351884708581516 -0.024585675873944668;
        0.4311573576048842 -0.02533727789279605;
        0.4367863586978133 -0.023934027433577754;
        0.44524542424764574 -0.02910331116291398;
        0.4516198885147879 -0.027294605645039383;
        0.4582279718911973 -0.0270348046323996;
        0.4648360552676066 -0.02706522158695135;
        0.47144413864401585 -0.02709563854150332;
        0.47805222202042513 -0.02712605549605529;
        0.4852234943209602 -0.028056926640688262;
        0.49777046011687626 -0.03280351425562156;
        0.5041051440403355 -0.02920222055232191;
        0.5097576725638029 -0.029488938234449424;
        0.5158985177216782 -0.0280983611000849;
        0.5225066010980874 -0.02812877805463687;
        0.5291146844744967 -0.028159195009188842;
        0.535722767850906 -0.02898771137351841;
        0.5423308512273155 -0.02901812832807038;
        0.5555470179801341 -0.02907896223717432;
        0.5621551013565433 -0.02910937919172607;
        0.5687631847329526 -0.02913979614627804;
        0.5753712681093619 -0.02917021310083001;
        0.5819793514857712 -0.029853620481563592;
        0.588987924763781 -0.0351388260940797;
        0.5969376493105523 -0.03198302097302097;
        0.6070463785086788 -0.03319595120213914;
        0.6097333016666902 -0.023167053821016692;
        0.6156205032202184 -0.025872864399525852;
        0.6222285865966277 -0.02292854719036108;
        0.628836669973037 -0.023321736603902687;
        0.6354447533494463 -0.024513025427222335;
        0.6420528367258556 -0.025559205266945773;
        0.6486609201022648 -0.027113266549254833;
        0.654754087890902 -0.03226245123091065;
        0.6618770868550834 -0.029568398687691788;
        0.6684851702314927 -0.02959881564224376;
        0.675093253607902 -0.02962923259679573;
        0.6817013369843112 -0.0296596495513477;
        0.6885797510443008 -0.030110313025991875;
        0.7076036103260652 -0.03621061980921647;
        0.7147417538663576 -0.040477244618408514;
        0.7213498372427669 -0.04101554301554633;
        0.7279579206191762 -0.0410459599700983;
        0.7345660039955855 -0.04107637692465027;
        0.7411740873719947 -0.04154212082998998;
        0.747782170748404 -0.04262976037931221;
        0.7543902541248133 -0.039353765493356896;
        0.7609983375012226 -0.037497765661161475;
        0.7676064208776319 -0.036439865238744096;
        0.7742145042540411 -0.036470282193296066;
        0.7808225876304504 -0.03650069914784804;
        0.7879607311707428 -0.03691956482992431;
        0.794038754383269 -0.03684712698095183;
        0.8010262492454052 -0.03726335659532598;
        0.8069845904525073 -0.03804630015355692;
        0.8138630045124968 -0.03665278392060767;
        0.8204710878889061 -0.03668320087515964;
        0.8270791712653154 -0.037123032462000216;
        0.8336872546417247 -0.037153449416551965;
        0.8402953380181339 -0.035395916394654;
        0.8469034213945432 -0.034991006398418234;
        0.8535115047709525 -0.03393310597600041;
        0.8601195881473618 -0.03345564148796676;
        0.8667276715237713 -0.03239774106554916;
        0.8733357549001806 -0.03192027657751528;
        0.8799438382765898 -0.031080039630491774;
        0.8870668372407713 -0.027412547655649933;
        0.8925592701770075 -0.03250512712007092;
        0.8997680884058177 -0.03066340905156162;
        0.906376171782227 -0.031714771354984794;
        0.9129842551586362 -0.030796797452463442;
        0.9195923385350455 -0.03198808627578287;
        0.9262004219114548 -0.03310682060730463;
        0.9328085052878641 -0.0347334363814118;
        0.9394165886642734 -0.03737581504069065;
        0.9460246720406826 -0.04030841166716126;
        0.9526327554170919 -0.044111662195207346;
        0.9592408387935012 -0.04885812111662724;
        0.9658489221699105 -0.06056981125065186;
        0.9724570055463198 -0.07779564276132245;
        0.979065088922729 -0.09197418561647797;
        0.9843063840884434 -0.1041853567357609;
        0.997567722376675 -0.10238385727207411;
        0.9993184353751003 -0.12552539725331568;
        0.9862739071515392 -0.12739219339858532;
        0.9916805208231467 -0.14485998598701566;
        1.0000908087567586 -0.14434129571224297;
        1.0009919110353598 -0.1624166944001666;
        0.9953395421968609 -0.16424081613278219;
        0.9997085229415945 -0.18044057818690384;
        1.0024937481663618 -0.18516944052304574;
        1.00129227846156 -0.20192399777300318;
        0.9960286968976666 -0.20481093215846236;
        1.0008631821384166 -0.22546595523452484;
        1.0006164517526093 -0.25289948674765306;
        0.9964863996423534 -0.2693412764777221;
        0.9955852973637522 -0.07699517095477959;
        0.9951462988177671 -0.05216875047464997;
        0.9907794185445455 -0.03698826910342734;
        0.9903023643970505 -0.013709738675984351;
        0.98687464200394 0.0039522517878842756;
        0.9844717025943366 0.020823162530183126;
        0.9832702328895349 0.037109920845020294;
        0.9790250399325691 0.04782399352502553;
        0.9781639866441276 0.06514671413855111;
        0.9763617820869253 0.07664764117240885;
        0.9733581078249209 0.09549661313159397;
        0.9703544335629168 0.11394653538589017;
        0.9670503918747122 0.13363489431418518;
        0.9606826024392632 0.14479769196406078;
        0.9633601634956783 0.15465899701304986;
        0.9577723758209657 0.16832841698852352;
        0.9542347150234942 0.18155767993010752;
        0.9493287137288873 0.2003178628018829;
        0.9427635399847922 0.21797467777968915;
        0.938515486385672 0.23851678778761176;
        0.9328085052878641 0.2565002936956311;
        0.9262004219114548 0.27125927248517057;
        0.9201930733874464 0.2888370496669441;
        0.9141857248634377 0.3033111069218051;
        0.9075776414870285 0.31658715251509606;
        0.9012699255368197 0.3335822486267397;
        0.8949622095866108 0.35259153804679666;
        0.8883541262102015 0.3640130102136294;
        0.8817460428337922 0.37666790874102773;
        0.8721342851953788 0.3953264095352911;
        0.8655262018189696 0.4090293173886601;
        0.8589181184425603 0.422554869817634;
        0.852310035066151 0.4354274318204262;
        0.8457019516897417 0.44387416982354244;
        0.8390938683133324 0.45211130596146465;
        0.8324857849369232 0.4666445596654102;
        0.8258777015605139 0.47712282343886947;
        0.8192696181841046 0.4897051674744699;
        0.8126615348076953 0.5017070755756866;
        0.806053451431286 0.5129834387589235;
        0.7964416937928724 0.5268694663149204;
        0.7898336104164632 0.5376540712759712;
        0.7832255270400539 0.546898908688865;
        0.7766174436636446 0.555708419150971;
        0.7700093602872353 0.5645179296130771;
        0.763401276910826 0.5733274400751832;
        0.7507858450104083 0.5905809233627188;
        0.7441777616339988 0.594746946349755;
        0.7375696782575896 0.6023955849430935;
        0.7309615948811803 0.6137445026181285;
        0.724353511504771 0.6219735771458508;
        0.7177454281283617 0.6286064528540178;
        0.7093351401947501 0.6342399228202726;
        0.7027270568183408 0.6434041441311688;
        0.6961189734419315 0.6510527827245073;
        0.6895108900655222 0.6594269662358256;
        0.682902806689113 0.66722071381276;
        0.6762947233127037 0.6730554901111494;
        0.6696866399362944 0.6790353753931346;
        0.6594741474454799 0.6908120262544775;
        0.6528660640690707 0.6977593047604356;
        0.6462579806926614 0.7030861996162392;
        0.6396498973162521 0.7093563028654164;
        0.6330418139398428 0.7185285857865121;
        0.6264337305634335 0.7272655417568203;
        0.6198256471870243 0.7317217827110483;
        0.6090124198438092 0.7360522709254861;
        0.6024043364673999 0.7423949286664612;
        0.5957962530909906 0.7500435672597998;
        0.5891881697145813 0.7572568789023506;
        0.582580086338172 0.7638172001187197;
        0.5759720029617628 0.7697245309089069;
        0.5693639195853533 0.7751965347483065;
        0.562755836208944 0.7789272307845548;
        0.5561477528325347 0.7856326609845198;
        0.5495396694561254 0.7921929822008887;
        0.5429315860797161 0.7985356399418639;
        0.5363235027033069 0.8053861791254249;
        0.5297154193268976 0.8114386188992082;
        0.5231073359504883 0.8170557317222036;
        0.516499252574079 0.8212217547092396;
        0.5096575500884029 0.8224383037100366;
        0.5032830858212605 0.8318755444208469;
        0.49667500244485124 0.8433695710794777;
        0.49006691906844196 0.8519614180661899;
        0.4834588356920327 0.8588119572497508;
        0.4768507523156234 0.8660978233840995;
        0.4702426689392141 0.8731660260430543;
        0.46363458556280485 0.8800891197184132;
        0.45282135821958963 0.8863060247195982;
        0.44621327484318035 0.8922859100015835;
        0.4396051914667711 0.8971774779065993;
        0.4329971080903618 0.9032299176803826;
        0.4263890247139525 0.9105883383065292;
        0.41978094133754323 0.9183095313916658;
        0.41317285796113395 0.9263934969357921;
        0.40609753636619067 0.9380452633650349;
        0.4002069973968158 0.9416593909268507;
        0.3933486078319061 0.9472353324536668;
        0.38674052445549684 0.95328777222745;
        0.38013244107908756 0.9597029844602232;
        0.3735243577026783 0.9682948314469353;
        0.366916274326269 0.9784828772532027;
        0.3603081909498597 0.9846078715187839;
        0.35342977688987004 0.9925407169600935;
        0.34709202419704116 0.9962774241155627;
        0.3404839408206319 1.0018219824467602;
        0.3338758574442226 1.0075842042533516;
        0.3272677740678132 1.013128762584549;
        0.32065969069140393 1.0183105484567567;
        0.31405160731499465 1.0229118983945806;
        0.3074435239385854 1.0267877034144248;
        0.3008354405621761 1.0300830724998853;
        0.2942273571857668 1.033088223618154;
        0.28761927380935753 1.035077611851251;
        0.28101119043294825 1.0359786827073787;
        0.274403107056539 1.0360816541537285;
        0.2677950236801297 1.0361120711082805;
        0.2611869403037204 1.0361424880628323;
        0.24797077355090186 1.0362033219719362;
        0.24136269017449258 1.0355807485003066;
        0.2347546067980833 1.0354660564712626;
        0.22814652342167402 1.034263047065249;
        0.22153844004526474 1.0329874831674375;
        0.2146800504803551 1.0292401724982148;
        0.2017141899160369 1.0247349674743271;
        0.19510610653962762 1.020629778396395;
        0.18849802316321834 1.015653935416887;
        0.181889939786809 1.0099525475193996;
        0.17528185641039973 1.004468823097306;
        0.16814371287010738 0.9937550793058587;
        0.16231599584608153 0.9921700564127864;
        0.1554576062811719 0.985986124060682;
        0.1488495229047626 0.9804298451467903;
        0.14224143952835333 0.9737126943641312;
        0.13563335615194405 0.9683015244338355;
        0.12929560345911517 0.9536355102896683;
        0.12241718939912549 0.9471764467379329;
        0.11580910602271621 0.9355981450048101;
        0.10920102264630693 0.9243826157306769;
        0.10259293926989765 0.9105551247518171;
        0.09575123678422143 0.8984780785331863;
        0.0899775073694799 0.888919400435642;
        0.08336942399307062 0.8744389190306003;
        0.07886391260006426 0.8528202806845055;
        0.07315693150225627 0.8379910176768938;
        0.066548848125847 0.8307841240745985;
        0.06564774584724575 0.8158093970094378;
        0.06054149960183858 0.7980922912824303;
        0.05633635563503264 0.7810095173167151;
        0.05255172606490732 0.762077777663991;
        0.0491275374062225 0.7460299746927342;
        0.0428429266426445 0.7297632878724039;
        0.040717249472610684 0.7143623742646338;
        0.03549085625672338 0.6926418501186558;
        0.032640703123666015 0.6758328073466998;
        0.03017101539712924 0.656069045479365;
        0.025198265785588925 0.6335234572515024;
        0.024497408457787917 0.611844982659844;
        0.02357320099255583 0.5891954919941348;
        0.020011921559861645 0.5743549570603541;
        0.01682648649636176 0.5482954487651607;
        0.01818969250757907 0.5220989181395922;
        0.017588957655178206 0.5053305109979234;
        0.015271837510203556 0.48407752812278126;
        0.014485160917773854 0.4615823467798297;
        0.013083446262171894 0.44256742959473094;
        0.01248271140977103 0.4157274512902295;
        0.012783078835971462 0.39800826180432247;
        0.012783078835971462 0.3804500747892142;
        0.012783078835971462 0.36289188777410597;
        0.014284915966973566 0.3424971626346607;
        0.013984548540773134 0.32505644539527223;
        0.014084671016173334 0.310477368647053;
        0.010966571067997477 0.28206418035439396;
        0.014284915966973566 0.27429594034457083;
        0.01323362997527211 0.25075684681708177;
        0.01231107288051364 0.2288133595710724;
        0.011109603175711968 0.21046260350155954;
        0.009533649407075961 0.1868896477697224;
        0.010680506852568494 0.16520094067394608;
        0.009946275366300827 0.13659942402956426;
        0.009527934170636632 0.10688400470266712;
        0.007795582503248133 0.09078150689676323;
        0.008597959364245622 0.06373028643334222;
        0.43795317062266875 -0.035121998835699664;
        0.0037720560499588107 -0.8602109871455366;
        0.003471688623758379 -0.8717250388977706;
        0.0029138634036718902 -0.8955514393247082;
        0.001369116640355439 -0.9132165300843054;
    ]

    push!(cpDist, -cpDist[end])
    p1 = plot(airfoilCtrlX[1:(end÷2)], -cpDist[1:(end÷2)], label="-Cp", xlabel="Control Point X", ylabel="-Cp lower")
    p1 = plot!(p1, airfoilCtrlX[((end÷2)+1):end], -cpDist[((end÷2)+1):end], label="-Cp upper", xlabel="Control Point X", ylabel="-Cp")
    plot!(p1, GiesingAndSmith0p9[:, 1], GiesingAndSmith0p9[:, 2], label="Giesing and Smith")
    plot!(p1, Ausman0p9[:, 1], Ausman0p9[:, 2], label="Ausman")
    yticks!(p1, -1.0:0.5:2.0)
    ylims!(p1, -1.2, 2.0)
    # plot!(p1, GiesingAndSmith0p6[:, 1], GiesingAndSmith0p6[:, 2], label="Giesing and Smith")
    # plot!(p1, Ausman0p6[:, 1], Ausman0p6[:, 2], label="Ausman")
    p2 = plot(airfoilXY[1, :], airfoilXY[2, :], xlabel="X", ylabel="Y", aspect_ratio=:equal)
    # p3 = plot(airfoilCtrlX, Φ, label="Φ", xlabel="Control Point X", ylabel="Φ")
    p = plot(p1, p2, layout=(2, 1), size=(1600, 1600))
    title!(p, "h/c=$(depth/chord)")
    # savefig(p, "test_FSVPM.png")
    savefig(p1, "test_FSVPM_hc$(depth/chord).png")

end
# using Debugger: @run
phi, vt = test_FSVPM()

function test_LL()
    airfoilX = [1.00000000e+00, 9.99748469e-01, 9.98994144e-01, 9.97737820e-01,
        9.95980826e-01, 9.93725017e-01, 9.90972776e-01, 9.87727006e-01,
        9.83991130e-01, 9.79769082e-01, 9.75065307e-01, 9.69884749e-01,
        9.64232850e-01, 9.58115535e-01, 9.51539214e-01, 9.44510764e-01,
        9.37037527e-01, 9.29127295e-01, 9.20788306e-01, 9.12029227e-01,
        9.02859150e-01, 8.93287575e-01, 8.83324401e-01, 8.72979917e-01,
        8.62264785e-01, 8.51190028e-01, 8.39767022e-01, 8.28007478e-01,
        8.15923431e-01, 8.03527225e-01, 7.90831502e-01, 7.77849185e-01,
        7.64593465e-01, 7.51077787e-01, 7.37315835e-01, 7.23321515e-01,
        7.09108945e-01, 6.94692435e-01, 6.80086473e-01, 6.65305712e-01,
        6.50364951e-01, 6.35279120e-01, 6.20063267e-01, 6.04732540e-01,
        5.89302170e-01, 5.73787460e-01, 5.58203764e-01, 5.42566474e-01,
        5.26891007e-01, 5.11192785e-01, 4.95487223e-01, 4.79789714e-01,
        4.64115613e-01, 4.48480227e-01, 4.32898794e-01, 4.17386477e-01,
        4.01958344e-01, 3.86738630e-01, 3.71649882e-01, 3.56690274e-01,
        3.41873673e-01, 3.27213781e-01, 3.12724131e-01, 2.98418080e-01,
        2.84308803e-01, 2.70409290e-01, 2.56732338e-01, 2.43290550e-01,
        2.30096330e-01, 2.17161876e-01, 2.04499178e-01, 1.92120011e-01,
        1.80035933e-01, 1.68258275e-01, 1.56798139e-01, 1.45666390e-01,
        1.34873649e-01, 1.24430284e-01, 1.14346405e-01, 1.04631850e-01,
        9.52961805e-02, 8.63486692e-02, 7.77982885e-02, 6.96537000e-02,
        6.19232428e-02, 5.46149212e-02, 4.77363922e-02, 4.12949532e-02,
        3.52975292e-02, 2.97506599e-02, 2.46604877e-02, 2.00327452e-02,
        1.58727434e-02, 1.21853607e-02, 8.97503220e-03, 6.24573957e-03,
        4.00100246e-03, 2.24387023e-03, 9.76914964e-04, 2.02225557e-04,
        -7.76150742e-05, 1.44393051e-04, 8.69839323e-04, 2.09882656e-03,
        3.83095023e-03, 6.06529546e-03, 8.80043400e-03, 1.20344211e-02,
        1.57647923e-02, 1.99885608e-02, 2.47022142e-02, 2.99017124e-02,
        3.55824857e-02, 4.17394326e-02, 4.83669198e-02, 5.54587816e-02,
        6.30083206e-02, 7.10083097e-02, 7.94509947e-02, 8.83280981e-02,
        9.76308247e-02, 1.07349867e-01, 1.17475415e-01, 1.27997163e-01,
        1.38904321e-01, 1.50185627e-01, 1.61829360e-01, 1.73823353e-01,
        1.86155013e-01, 1.98811334e-01, 2.11778916e-01, 2.25043987e-01,
        2.38592420e-01, 2.52409755e-01, 2.66481224e-01, 2.80791769e-01,
        2.95326068e-01, 3.10068557e-01, 3.25003456e-01, 3.40114791e-01,
        3.55386417e-01, 3.70802044e-01, 3.86345261e-01, 4.01983707e-01,
        4.17609168e-01, 4.33314925e-01, 4.49084928e-01, 4.64903087e-01,
        4.80753287e-01, 4.96619410e-01, 5.12485349e-01, 5.28335024e-01,
        5.44152404e-01, 5.59921518e-01, 5.75626476e-01, 5.91251481e-01,
        6.06780846e-01, 6.22199013e-01, 6.37490561e-01, 6.52640229e-01,
        6.67632925e-01, 6.82453740e-01, 6.97087967e-01, 7.11521109e-01,
        7.25738899e-01, 7.39727306e-01, 7.53472554e-01, 7.66961129e-01,
        7.80179797e-01, 7.93115612e-01, 8.05755930e-01, 8.18088419e-01,
        8.30101069e-01, 8.41782206e-01, 8.53120497e-01, 8.64104966e-01,
        8.74724998e-01, 8.84970350e-01, 8.94831160e-01, 9.04297954e-01,
        9.13361654e-01, 9.22013585e-01, 9.30245482e-01, 9.38049496e-01,
        9.45418198e-01, 9.52344587e-01, 9.58822093e-01, 9.64844583e-01,
        9.70406363e-01, 9.75502182e-01, 9.80127238e-01, 9.84277176e-01,
        9.87948094e-01, 9.91136544e-01, 9.93839535e-01, 9.96054533e-01,
        9.97779463e-01, 9.99012710e-01, 9.99753120e-01, 1.00000000e+00,]

    airfoilY = [1.32931687e-17, -1.82812930e-05, -7.30944855e-05, -1.64347800e-04,
        -2.91889208e-04, -4.55507800e-04, -6.54935663e-04, -8.89850197e-04,
        -1.15987683e-03, -1.46459203e-03, -1.80352658e-03, -2.17616897e-03,
        -2.58196882e-03, -3.02034029e-03, -3.49066530e-03, -3.99229653e-03,
        -4.52455996e-03, -5.08675710e-03, -5.67816655e-03, -6.29804503e-03,
        -6.94562764e-03, -7.62012742e-03, -8.32073416e-03, -9.04661229e-03,
        -9.79689802e-03, -1.05706957e-02, -1.13670732e-02, -1.21850570e-02,
        -1.30236259e-02, -1.38817046e-02, -1.47581570e-02, -1.56517783e-02,
        -1.65612877e-02, -1.74853203e-02, -1.84224193e-02, -1.93710285e-02,
        -2.03294850e-02, -2.12960119e-02, -2.22687127e-02, -2.32455654e-02,
        -2.42244185e-02, -2.52029874e-02, -2.61788525e-02, -2.71494585e-02,
        -2.81121150e-02, -2.90639992e-02, -3.00021597e-02, -3.09235223e-02,
        -3.18248972e-02, -3.27029883e-02, -3.35544039e-02, -3.43756691e-02,
        -3.51632398e-02, -3.59135179e-02, -3.66228683e-02, -3.72876365e-02,
        -3.79041676e-02, -3.84811382e-02, -3.90342882e-02, -3.95595359e-02,
        -4.00525040e-02, -4.05088092e-02, -4.09240870e-02, -4.12940166e-02,
        -4.16143446e-02, -4.18809091e-02, -4.20896616e-02, -4.22366881e-02,
        -4.23182288e-02, -4.23306958e-02, -4.22706893e-02, -4.21350113e-02,
        -4.19206774e-02, -4.16249266e-02, -4.12452281e-02, -4.07792865e-02,
        -4.02250441e-02, -3.95806809e-02, -3.88446129e-02, -3.80154873e-02,
        -3.70921764e-02, -3.60737690e-02, -3.49595605e-02, -3.37490410e-02,
        -3.24418823e-02, -3.10379237e-02, -2.95371569e-02, -2.79397099e-02,
        -2.62458312e-02, -2.44558733e-02, -2.25702762e-02, -2.05895518e-02,
        -1.85142688e-02, -1.63450378e-02, -1.40824985e-02, -1.17273072e-02,
        -9.28012641e-03, -6.74161609e-03, -4.11242610e-03, -1.39319109e-03,
        1.40565117e-03, 4.22447832e-03, 7.05238110e-03, 9.88778382e-03,
        1.27286299e-02, 1.55723930e-02, 1.84160921e-02, 2.12563096e-02,
        2.40892141e-02, 2.69105849e-02, 2.97158410e-02, 3.25000716e-02,
        3.52580700e-02, 3.79843691e-02, 4.06732791e-02, 4.33189260e-02,
        4.59152923e-02, 4.84562576e-02, 5.09356399e-02, 5.33472370e-02,
        5.56848679e-02, 5.79424128e-02, 6.01138529e-02, 6.21933088e-02,
        6.41750771e-02, 6.60536652e-02, 6.78238247e-02, 6.94805814e-02,
        7.10192641e-02, 7.24355299e-02, 7.37253873e-02, 7.48852162e-02,
        7.59117853e-02, 7.68022664e-02, 7.75542465e-02, 7.81657358e-02,
        7.86351743e-02, 7.89614351e-02, 7.91438251e-02, 7.91820838e-02,
        7.90763791e-02, 7.88273019e-02, 7.84358584e-02, 7.79037359e-02,
        7.72536172e-02, 7.65010834e-02, 7.56491024e-02, 7.47008558e-02,
        7.36597232e-02, 7.25292662e-02, 7.13132131e-02, 7.00154437e-02,
        6.86399746e-02, 6.71909460e-02, 6.56726076e-02, 6.40893072e-02,
        6.24454789e-02, 6.07456324e-02, 5.89943438e-02, 5.71962460e-02,
        5.53560212e-02, 5.34783935e-02, 5.15681218e-02, 4.96299943e-02,
        4.76688224e-02, 4.56894355e-02, 4.36966763e-02, 4.16953954e-02,
        3.96904467e-02, 3.76866826e-02, 3.56889483e-02, 3.37020768e-02,
        3.17308826e-02, 2.97801554e-02, 2.78546533e-02, 2.59590948e-02,
        2.40981510e-02, 2.22764362e-02, 2.04984987e-02, 1.87688101e-02,
        1.70917545e-02, 1.54716170e-02, 1.39125715e-02, 1.24186681e-02,
        1.09938204e-02, 9.64179183e-03, 8.36618292e-03, 7.17041746e-03,
        6.05772950e-03, 5.03115026e-03, 4.09349554e-03, 3.24735368e-03,
        2.49507408e-03, 1.83875666e-03, 1.28024209e-03, 8.21103204e-04,
        4.62637439e-04, 2.05860579e-04, 5.15017842e-05, -1.32931687e-17,]
    airfoilCtrlX = [9.99874235e-01, 9.99371307e-01, 9.98365982e-01, 9.96859323e-01,
        9.94852922e-01, 9.92348897e-01, 9.89349891e-01, 9.85859068e-01,
        9.81880106e-01, 9.77417195e-01, 9.72475028e-01, 9.67058800e-01,
        9.61174192e-01, 9.54827375e-01, 9.48024989e-01, 9.40774145e-01,
        9.33082411e-01, 9.24957800e-01, 9.16408766e-01, 9.07444188e-01,
        8.98073362e-01, 8.88305988e-01, 8.78152159e-01, 8.67622351e-01,
        8.56727406e-01, 8.45478525e-01, 8.33887250e-01, 8.21965454e-01,
        8.09725328e-01, 7.97179363e-01, 7.84340343e-01, 7.71221325e-01,
        7.57835626e-01, 7.44196811e-01, 7.30318675e-01, 7.16215230e-01,
        7.01900690e-01, 6.87389454e-01, 6.72696093e-01, 6.57835332e-01,
        6.42822035e-01, 6.27671194e-01, 6.12397903e-01, 5.97017355e-01,
        5.81544815e-01, 5.65995612e-01, 5.50385119e-01, 5.34728741e-01,
        5.19041896e-01, 5.03340004e-01, 4.87638468e-01, 4.71952664e-01,
        4.56297920e-01, 4.40689511e-01, 4.25142636e-01, 4.09672411e-01,
        3.94348487e-01, 3.79194256e-01, 3.64170078e-01, 3.49281974e-01,
        3.34543727e-01, 3.19968956e-01, 3.05571106e-01, 2.91363442e-01,
        2.77359046e-01, 2.63570814e-01, 2.50011444e-01, 2.36693440e-01,
        2.23629103e-01, 2.10830527e-01, 1.98309594e-01, 1.86077972e-01,
        1.74147104e-01, 1.62528207e-01, 1.51232265e-01, 1.40270020e-01,
        1.29651967e-01, 1.19388344e-01, 1.09489127e-01, 9.99640151e-02,
        9.08224248e-02, 8.20734789e-02, 7.37259943e-02, 6.57884714e-02,
        5.82690820e-02, 5.11756567e-02, 4.45156727e-02, 3.82962412e-02,
        3.25240945e-02, 2.72055738e-02, 2.23466164e-02, 1.79527443e-02,
        1.40290521e-02, 1.05801965e-02, 7.61038589e-03, 5.12337102e-03,
        3.12243635e-03, 1.61039260e-03, 5.89570260e-04, 6.23052412e-05,
        3.33889886e-05, 5.07116187e-04, 1.48433294e-03, 2.96488839e-03,
        4.94812284e-03, 7.43286473e-03, 1.04174275e-02, 1.38996067e-02,
        1.78766765e-02, 2.23453875e-02, 2.73019633e-02, 3.27420991e-02,
        3.86609591e-02, 4.50531762e-02, 5.19128507e-02, 5.92335511e-02,
        6.70083152e-02, 7.52296522e-02, 8.38895464e-02, 9.29794614e-02,
        1.02490346e-01, 1.12412641e-01, 1.22736289e-01, 1.33450742e-01,
        1.44544974e-01, 1.56007493e-01, 1.67826356e-01, 1.79989183e-01,
        1.92483173e-01, 2.05295125e-01, 2.18411452e-01, 2.31818203e-01,
        2.45501088e-01, 2.59445490e-01, 2.73636497e-01, 2.88058919e-01,
        3.02697313e-01, 3.17536007e-01, 3.32559124e-01, 3.47750604e-01,
        3.63094230e-01, 3.78573652e-01, 3.94164484e-01, 4.09796437e-01,
        4.25462047e-01, 4.41199927e-01, 4.56994008e-01, 4.72828187e-01,
        4.88686349e-01, 5.04552380e-01, 5.20410186e-01, 5.36243714e-01,
        5.52036961e-01, 5.67773997e-01, 5.83438978e-01, 5.99016163e-01,
        6.14489929e-01, 6.29844787e-01, 6.45065395e-01, 6.60136577e-01,
        6.75043332e-01, 6.89770853e-01, 7.04304538e-01, 7.18630004e-01,
        7.32733103e-01, 7.46599930e-01, 7.60216841e-01, 7.73570463e-01,
        7.86647704e-01, 7.99435771e-01, 8.11922175e-01, 8.24094744e-01,
        8.35941638e-01, 8.47451352e-01, 8.58612732e-01, 8.69414982e-01,
        8.79847674e-01, 8.89900755e-01, 8.99564557e-01, 9.08829804e-01,
        9.17687620e-01, 9.26129534e-01, 9.34147489e-01, 9.41733847e-01,
        9.48881392e-01, 9.55583340e-01, 9.61833338e-01, 9.67625473e-01,
        9.72954273e-01, 9.77814710e-01, 9.82202207e-01, 9.86112635e-01,
        9.89542319e-01, 9.92488039e-01, 9.94947034e-01, 9.96916998e-01,
        9.98396086e-01, 9.99382915e-01, 9.99876560e-01,]
    airfoilCtrlY = [-9.14064650e-06, -4.56878893e-05, -1.18721143e-04, -2.28118504e-04,
        -3.73698504e-04, -5.55221732e-04, -7.72392930e-04, -1.02486351e-03,
        -1.31223443e-03, -1.63405931e-03, -1.98984778e-03, -2.37906889e-03,
        -2.80115455e-03, -3.25550279e-03, -3.74148092e-03, -4.25842824e-03,
        -4.80565853e-03, -5.38246183e-03, -5.98810579e-03, -6.62183633e-03,
        -7.28287753e-03, -7.97043079e-03, -8.68367322e-03, -9.42175515e-03,
        -1.01837968e-02, -1.09688845e-02, -1.17760651e-02, -1.26043414e-02,
        -1.34526652e-02, -1.43199308e-02, -1.52049676e-02, -1.61065330e-02,
        -1.70233040e-02, -1.79538698e-02, -1.88967239e-02, -1.98502568e-02,
        -2.08127485e-02, -2.17823623e-02, -2.27571391e-02, -2.37349920e-02,
        -2.47137030e-02, -2.56909200e-02, -2.66641555e-02, -2.76307867e-02,
        -2.85880571e-02, -2.95330794e-02, -3.04628410e-02, -3.13742097e-02,
        -3.22639427e-02, -3.31286961e-02, -3.39650365e-02, -3.47694545e-02,
        -3.55383789e-02, -3.62681931e-02, -3.69552524e-02, -3.75959020e-02,
        -3.81926529e-02, -3.87577132e-02, -3.92969120e-02, -3.98060199e-02,
        -4.02806566e-02, -4.07164481e-02, -4.11090518e-02, -4.14541806e-02,
        -4.17476268e-02, -4.19852853e-02, -4.21631748e-02, -4.22774584e-02,
        -4.23244623e-02, -4.23006926e-02, -4.22028503e-02, -4.20278444e-02,
        -4.17728020e-02, -4.14350774e-02, -4.10122573e-02, -4.05021653e-02,
        -3.99028625e-02, -3.92126469e-02, -3.84300501e-02, -3.75538319e-02,
        -3.65829727e-02, -3.55166647e-02, -3.43543007e-02, -3.30954616e-02,
        -3.17399030e-02, -3.02875403e-02, -2.87384334e-02, -2.70927706e-02,
        -2.53508523e-02, -2.35130747e-02, -2.15799140e-02, -1.95519103e-02,
        -1.74296533e-02, -1.52137682e-02, -1.29049028e-02, -1.05037168e-02,
        -8.01087125e-03, -5.42702110e-03, -2.75280860e-03, 6.23003887e-06,
        2.81506475e-03, 5.63842971e-03, 8.47008246e-03, 1.13082069e-02,
        1.41505115e-02, 1.69942426e-02, 1.98362009e-02, 2.26727619e-02,
        2.54998995e-02, 2.83132129e-02, 3.11079563e-02, 3.38790708e-02,
        3.66212196e-02, 3.93288241e-02, 4.19961025e-02, 4.46171091e-02,
        4.71857749e-02, 4.96959487e-02, 5.21414384e-02, 5.45160525e-02,
        5.68136403e-02, 5.90281328e-02, 6.11535809e-02, 6.31841929e-02,
        6.51143711e-02, 6.69387449e-02, 6.86522030e-02, 7.02499227e-02,
        7.17273970e-02, 7.30804586e-02, 7.43053018e-02, 7.53985007e-02,
        7.63570258e-02, 7.71782565e-02, 7.78599911e-02, 7.84004550e-02,
        7.87983047e-02, 7.90526301e-02, 7.91629545e-02, 7.91292314e-02,
        7.89518405e-02, 7.86315802e-02, 7.81697972e-02, 7.75786766e-02,
        7.68773503e-02, 7.60750929e-02, 7.51749791e-02, 7.41802895e-02,
        7.30944947e-02, 7.19212396e-02, 7.06643284e-02, 6.93277092e-02,
        6.79154603e-02, 6.64317768e-02, 6.48809574e-02, 6.32673930e-02,
        6.15955556e-02, 5.98699881e-02, 5.80952949e-02, 5.62761336e-02,
        5.44172074e-02, 5.25232576e-02, 5.05990581e-02, 4.86494083e-02,
        4.66791290e-02, 4.46930559e-02, 4.26960358e-02, 4.06929210e-02,
        3.86885647e-02, 3.66878155e-02, 3.46955126e-02, 3.27164797e-02,
        3.07555190e-02, 2.88174043e-02, 2.69068740e-02, 2.50286229e-02,
        2.31872936e-02, 2.13874675e-02, 1.96336544e-02, 1.79302823e-02,
        1.62816858e-02, 1.46920943e-02, 1.31656198e-02, 1.17062442e-02,
        1.03178061e-02, 9.00398737e-03, 7.76830019e-03, 6.61407348e-03,
        5.54443988e-03, 4.56232290e-03, 3.67042461e-03, 2.87121388e-03,
        2.16691537e-03, 1.55949937e-03, 1.05067265e-03, 6.41870322e-04,
        3.34249009e-04, 1.28681182e-04, 2.57508921e-05]

    airfoilXY = copy(transpose(hcat(airfoilX, airfoilY)))
    airfoilCtrlXY = copy(transpose(hcat(airfoilCtrlX, airfoilCtrlY)))

    Uinf = [1.0, 0.0, 0.1]
    span = 8.0
    sweep = deg2rad(1.0)
    rootChord = 1.0
    TR = 1.0
    npt_wing = 99
    npt_wing = 21
    options = Dict(
        "make_plot" => true,
    )
    # options = nothing
    midchords = zeros(3, 2)
    midchords[2, :] = [-0.5 * span, 0.5 * span]
    LLSystem, FlowCond, LLHydro, Airfoil, Airfoil_influences = LiftingLine.setup(Uinf, sweep, rootChord, TR, midchords;
        npt_wing=npt_wing, npt_airfoil=99, airfoil_xy=airfoilXY, airfoil_ctrl_xy=airfoilCtrlXY, options=options)
    @time LLOutputs = LiftingLine.solve(FlowCond, LLSystem, LLSystem.HydroProperties[1], Airfoil, Airfoil_influences)

    Fdist = LLOutputs.Fdist
    circDist = LLOutputs.Γdist
    F = LLOutputs.F
    println("Forces: $(F)")
    println("CL:\n$(LLOutputs.CL)\nCDi\n$(LLOutputs.CDi)\nCside:\n$(LLOutputs.CS)")
    return LLOutputs, FlowCond, LLSystem
end

function test_45degwingLL()

    # I introduced a bug here...
    referenceCL = [0.0016496729653562032, 0.14341157629455348, 0.2852589091604507, 0.42593046097276493, 0.5648584898305323, 0.7014970094315793]
    airfoilCoordFile = "$(pwd())/INPUT/PROFILES/NACA0012.dat"

    airfoilX = [1.00000000e+00, 9.98993338e-01, 9.95977406e-01, 9.90964349e-01,
        9.83974351e-01, 9.75035559e-01, 9.64183967e-01, 9.51463269e-01,
        9.36924689e-01, 9.20626766e-01, 9.02635129e-01, 8.83022222e-01,
        8.61867019e-01, 8.39254706e-01, 8.15276334e-01, 7.90028455e-01,
        7.63612734e-01, 7.36135537e-01, 7.07707507e-01, 6.78443111e-01,
        6.48460188e-01, 6.17879468e-01, 5.86824089e-01, 5.55419100e-01,
        5.23790958e-01, 4.92067018e-01, 4.60375022e-01, 4.28842581e-01,
        3.97596666e-01, 3.66763093e-01, 3.36466018e-01, 3.06827437e-01,
        2.77966694e-01, 2.50000000e-01, 2.23039968e-01, 1.97195156e-01,
        1.72569633e-01, 1.49262556e-01, 1.27367775e-01, 1.06973453e-01,
        8.81617093e-02, 7.10082934e-02, 5.55822757e-02, 4.19457713e-02,
        3.01536896e-02, 2.02535132e-02, 1.22851066e-02, 6.28055566e-03,
        2.26403871e-03, 2.51728808e-04, 2.51728808e-04, 2.26403871e-03,
        6.28055566e-03, 1.22851066e-02, 2.02535132e-02, 3.01536896e-02,
        4.19457713e-02, 5.55822757e-02, 7.10082934e-02, 8.81617093e-02,
        1.06973453e-01, 1.27367775e-01, 1.49262556e-01, 1.72569633e-01,
        1.97195156e-01, 2.23039968e-01, 2.50000000e-01, 2.77966694e-01,
        3.06827437e-01, 3.36466018e-01, 3.66763093e-01, 3.97596666e-01,
        4.28842581e-01, 4.60375022e-01, 4.92067018e-01, 5.23790958e-01,
        5.55419100e-01, 5.86824089e-01, 6.17879468e-01, 6.48460188e-01,
        6.78443111e-01, 7.07707507e-01, 7.36135537e-01, 7.63612734e-01,
        7.90028455e-01, 8.15276334e-01, 8.39254706e-01, 8.61867019e-01,
        8.83022222e-01, 9.02635129e-01, 9.20626766e-01, 9.36924689e-01,
        9.51463269e-01, 9.64183967e-01, 9.75035559e-01, 9.83974351e-01,
        9.90964349e-01, 9.95977406e-01, 9.98993338e-01, 1.00000000e+00]

    airfoilY = [1.33226763e-17, -1.41200438e-04, -5.63343432e-04, -1.26208774e-03,
        -2.23030811e-03, -3.45825298e-03, -4.93375074e-03, -6.64245132e-03,
        -8.56808791e-03, -1.06927428e-02, -1.29971013e-02, -1.54606806e-02,
        -1.80620201e-02, -2.07788284e-02, -2.35880799e-02, -2.64660655e-02,
        -2.93884006e-02, -3.23300020e-02, -3.52650469e-02, -3.81669313e-02,
        -4.10082448e-02, -4.37607812e-02, -4.63956016e-02, -4.88831639e-02,
        -5.11935307e-02, -5.32966591e-02, -5.51627743e-02, -5.67628192e-02,
        -5.80689678e-02, -5.90551853e-02, -5.96978089e-02, -5.99761253e-02,
        -5.98729133e-02, -5.93749219e-02, -5.84732545e-02, -5.71636340e-02,
        -5.54465251e-02, -5.33271006e-02, -5.08150416e-02, -4.79241724e-02,
        -4.46719377e-02, -4.10787401e-02, -3.71671601e-02, -3.29610920e-02,
        -2.84848303e-02, -2.37621474e-02, -1.88154040e-02, -1.36647314e-02,
        -8.32732576e-03, -2.81688492e-03, 2.81688492e-03, 8.32732576e-03,
        1.36647314e-02, 1.88154040e-02, 2.37621474e-02, 2.84848303e-02,
        3.29610920e-02, 3.71671601e-02, 4.10787401e-02, 4.46719377e-02,
        4.79241724e-02, 5.08150416e-02, 5.33271006e-02, 5.54465251e-02,
        5.71636340e-02, 5.84732545e-02, 5.93749219e-02, 5.98729133e-02,
        5.99761253e-02, 5.96978089e-02, 5.90551853e-02, 5.80689678e-02,
        5.67628192e-02, 5.51627743e-02, 5.32966591e-02, 5.11935307e-02,
        4.88831639e-02, 4.63956016e-02, 4.37607812e-02, 4.10082448e-02,
        3.81669313e-02, 3.52650469e-02, 3.23300020e-02, 2.93884006e-02,
        2.64660655e-02, 2.35880799e-02, 2.07788284e-02, 1.80620201e-02,
        1.54606806e-02, 1.29971013e-02, 1.06927428e-02, 8.56808791e-03,
        6.64245132e-03, 4.93375074e-03, 3.45825298e-03, 2.23030811e-03,
        1.26208774e-03, 5.63343432e-04, 1.41200438e-04, -1.33226763e-17]
    airfoilCtrlX = [9.99496669e-01, 9.97485372e-01, 9.93470878e-01, 9.87469350e-01,
        9.79504955e-01, 9.69609763e-01, 9.57823618e-01, 9.44193979e-01,
        9.28775727e-01, 9.11630948e-01, 8.92828675e-01, 8.72444620e-01,
        8.50560862e-01, 8.27265520e-01, 8.02652394e-01, 7.76820594e-01,
        7.49874136e-01, 7.21921522e-01, 6.93075309e-01, 6.63451649e-01,
        6.33169828e-01, 6.02351778e-01, 5.71121594e-01, 5.39605029e-01,
        5.07928988e-01, 4.76221020e-01, 4.44608801e-01, 4.13219623e-01,
        3.82179880e-01, 3.51614556e-01, 3.21646728e-01, 2.92397065e-01,
        2.63983347e-01, 2.36519984e-01, 2.10117562e-01, 1.84882395e-01,
        1.60916095e-01, 1.38315166e-01, 1.17170614e-01, 9.75675810e-02,
        7.95850013e-02, 6.32952845e-02, 4.87640235e-02, 3.60497304e-02,
        2.52036014e-02, 1.62693099e-02, 9.28283111e-03, 4.27229719e-03,
        1.25788376e-03, 2.51728808e-04, 1.25788376e-03, 4.27229719e-03,
        9.28283111e-03, 1.62693099e-02, 2.52036014e-02, 3.60497304e-02,
        4.87640235e-02, 6.32952845e-02, 7.95850013e-02, 9.75675810e-02,
        1.17170614e-01, 1.38315166e-01, 1.60916095e-01, 1.84882395e-01,
        2.10117562e-01, 2.36519984e-01, 2.63983347e-01, 2.92397065e-01,
        3.21646728e-01, 3.51614556e-01, 3.82179880e-01, 4.13219623e-01,
        4.44608801e-01, 4.76221020e-01, 5.07928988e-01, 5.39605029e-01,
        5.71121594e-01, 6.02351778e-01, 6.33169828e-01, 6.63451649e-01,
        6.93075309e-01, 7.21921522e-01, 7.49874136e-01, 7.76820594e-01,
        8.02652394e-01, 8.27265520e-01, 8.50560862e-01, 8.72444620e-01,
        8.92828675e-01, 9.11630948e-01, 9.28775727e-01, 9.44193979e-01,
        9.57823618e-01, 9.69609763e-01, 9.79504955e-01, 9.87469350e-01,
        9.93470878e-01, 9.97485372e-01, 9.99496669e-01]
    airfoilCtrlY = [-7.06002188e-05, -3.52271935e-04, -9.12715585e-04, -1.74619792e-03,
        -2.84428054e-03, -4.19600186e-03, -5.78810103e-03, -7.60526962e-03,
        -9.63041534e-03, -1.18449221e-02, -1.42288910e-02, -1.67613504e-02,
        -1.94204243e-02, -2.21834542e-02, -2.50270727e-02, -2.79272331e-02,
        -3.08592013e-02, -3.37975244e-02, -3.67159891e-02, -3.95875880e-02,
        -4.23845130e-02, -4.50781914e-02, -4.76393828e-02, -5.00383473e-02,
        -5.22450949e-02, -5.42297167e-02, -5.59627967e-02, -5.74158935e-02,
        -5.85620766e-02, -5.93764971e-02, -5.98369671e-02, -5.99245193e-02,
        -5.96239176e-02, -5.89240882e-02, -5.78184443e-02, -5.63050796e-02,
        -5.43868129e-02, -5.20710711e-02, -4.93696070e-02, -4.62980550e-02,
        -4.28753389e-02, -3.91229501e-02, -3.50641261e-02, -3.07229612e-02,
        -2.61234889e-02, -2.12887757e-02, -1.62400677e-02, -1.09960286e-02,
        -5.57210534e-03, 0.00000000e+00, 5.57210534e-03, 1.09960286e-02,
        1.62400677e-02, 2.12887757e-02, 2.61234889e-02, 3.07229612e-02,
        3.50641261e-02, 3.91229501e-02, 4.28753389e-02, 4.62980550e-02,
        4.93696070e-02, 5.20710711e-02, 5.43868129e-02, 5.63050796e-02,
        5.78184443e-02, 5.89240882e-02, 5.96239176e-02, 5.99245193e-02,
        5.98369671e-02, 5.93764971e-02, 5.85620766e-02, 5.74158935e-02,
        5.59627967e-02, 5.42297167e-02, 5.22450949e-02, 5.00383473e-02,
        4.76393828e-02, 4.50781914e-02, 4.23845130e-02, 3.95875880e-02,
        3.67159891e-02, 3.37975244e-02, 3.08592013e-02, 2.79272331e-02,
        2.50270727e-02, 2.21834542e-02, 1.94204243e-02, 1.67613504e-02,
        1.42288910e-02, 1.18449221e-02, 9.63041534e-03, 7.60526962e-03,
        5.78810103e-03, 4.19600186e-03, 2.84428054e-03, 1.74619792e-03,
        9.12715585e-04, 3.52271935e-04, 7.06002188e-05]

    airfoilXY = copy(transpose(hcat(airfoilX, airfoilY)))
    airfoilCtrlXY = copy(transpose(hcat(airfoilCtrlX, airfoilCtrlY)))


    Uinf = 50.0
    angles = [0.01, 2.1, 4.2, 6.3, 8.4, 10.5]
    # angles = [4.2]
    alpha = deg2rad(4.2)

    rootChord = 20.0 * 0.0254 # 20 inches
    span = 5 * rootChord # this is an aerodynamic span
    sweepAng = deg2rad(45)
    TR = 1.0
    npt_wing = 40
    npt_airfoil = 99
    answers = []
    midchords = zeros(3, 2)
    midchords[2, :] = [-0.5 * span, 0.5 * span]
    for alpha in deg2rad.(angles)
        println("alpha: $(rad2deg(alpha))")
        Uvec = [cos(alpha), 0.0, sin(alpha)] * Uinf
        LLSystem, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, zeros(6, npt_wing), zeros(size(midchords));
            npt_wing=npt_wing, npt_airfoil=npt_airfoil,
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
        )
        @time LLOutputs = LiftingLine.solve(FlowCond, LLSystem, LLHydro, Airfoils, AirfoilInfluences)
        F = LLOutputs.F
        # println("Forces: $(F)")
        # println("Spanwise cl: $(LLOutputs.Fdist[3,:])")
        println("CL\t\t\t\t|\tCDi\t\t\t|\tCside:\n$(LLOutputs.CL)\t\t|$(LLOutputs.CDi)\t\t|$(LLOutputs.CS)")
        push!(answers, LLOutputs.CL)
    end

    println("CL:\n", real(answers))
    println("Reference CL:\n", referenceCL)
    normError = norm(answers .- referenceCL)
    return normError
end

function test_clvent()

    FnhSweep = 0.9:0.1:30.0
    grav = 9.81
    h = 0.5 # water depth

    solverOptions = Dict(
        "rhof" => 1025.0, # water density
        "Uinf" => 1.0
    )
    clventSweep = []
    clventSweep2 = []
    for Fnh in FnhSweep
        # println("Fnh: $(Fnh)")

        Uinf = Fnh * √(grav * h)
        solverOptions["Uinf"] = Uinf
        # println("Uinf: $(Uinf)")
        pvap = 2.34e3 # Pa, water vapor pressure at 20 deg C

        clvent = compute_cl_ventilation(Fnh, solverOptions, pvap)
        push!(clventSweep, clvent)
        clvent2 = compute_cl_ventilation(Fnh, solverOptions, 700/760*101.3e3)
        push!(clventSweep2, clvent2)
    end

    plot(FnhSweep, clventSweep, xlabel="Fnh", ylabel="CLventilation",
        title="CL ventilation vs Fnh", legend=:topright, label="CLventilation")
    plot!(FnhSweep, clventSweep2, label="CLventilation (atmospheric)", linestyle=:dash, xaxis=:log)
    ylims!(-0.2, 1.5)

end
# ==============================================================================
#                         Run some tests
# ==============================================================================
# test_VPM()
# LLOutputs, FlowCond, LLSystem = test_LL()
# vals, FncRange = test_wavedrag()
# test_45degwingLL()
# TecplotIO.write_hydroLoads(LLOutputs, FlowCond, LLSystem, "./OUTPUT/")
# TecplotIO.write_hydromesh(LLSystem, FlowCond.uvec, "./OUTPUT/")