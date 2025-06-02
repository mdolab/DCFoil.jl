# --- Julia 1.11---
"""
@File          :   SolverSetup.jl
@Date created  :   2025/05/28
@Last modified :   2025/05/28
@Author        :   Galen Ng
@Desc          :   General setup routines for the dynamic solvers
"""

function setup_solverOM(displCol, LECoords, TECoords, nodeConn, appendageParams, solverOptions::AbstractDict, solverType="FLUTTER")
    """
    """

    # ************************************************
    #     Initializations
    # ************************************************
    uRange = solverOptions["uRange"]
    nModes = solverOptions["nModes"]
    debug = solverOptions["debug"]
    println("====================================================================================")
    println("        BEGINNING $(solverType) SOLUTION")
    println("====================================================================================")
    if solverOptions["run_flutter"] && uppercase(solverType) == "FLUTTER"
        println("Speed range [m/s]: ", uRange)
    elseif solverOptions["run_forced"]
        println("Frequency sweep [Hz]: ", solverOptions["fRange"])
    end
    if debug
        rm("DebugOutput/", recursive=true)
        mkpath("DebugOutput/")
        println("+---------------------------+")
        println("|    Running debug mode!    |")
        println("+---------------------------+")
    end

    # --- Init model structure ---
    if length(solverOptions["appendageList"]) == 1
        appendageOptions = solverOptions["appendageList"][1]
        tipMass = appendageOptions["use_tipMass"]

        idxTip = LiftingLine.get_tipnode(LECoords)
        midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        # ---------------------------
        #   Hydrodynamics
        # ---------------------------
        α0 = appendageParams["alfa0"]
        β0 = appendageParams["beta"]
        rake = appendageParams["rake"]
        depth0 = appendageParams["depth0"]
        airfoilXY, airfoilCtrlXY, _, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
        npt_wing = size(displCol, 2) # overwrite
        LLSystem, FlowCond, _, _, _ = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol, pretwistDist;
            npt_wing=npt_wing,
            npt_airfoil=npt_airfoil,
            rhof=solverOptions["rhof"],
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
            options=options,
        )

    else
        error("Only one appendage is supported at the moment")
    end


    # ************************************************
    #     FLUTTER SOLUTION
    # ************************************************
    N_MAX_Q_ITER = solverOptions["maxQIter"]    # TEST VALUE
    N_R = 8                                     # reduced problem size (Nr x Nr)


    x_αbVec = appendageParams["x_ab"]

    # ************************************************
    #     FEM assembly
    # ************************************************

    globalKs, globalMs, _, _, FEMESH, WingStructModel, StrutStructModel = FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, [appendageParams], appendageOptions, solverOptions)

    abVec = WingStructModel.ab
    chordVec = WingStructModel.chord
    ebVec = 0.25 * chordVec .+ abVec
    b_ref = sum(chordVec) / WingStructModel.nNodes         # mean semichord

    # ---------------------------
    #   Get structural damping
    # ---------------------------
    alphaConst = solverOptions["alphaConst"]
    betaConst = solverOptions["betaConst"]
    globalCs = alphaConst * globalMs .+ betaConst * globalKs

    # ---------------------------
    #   Add any discrete masses
    # ---------------------------
    if tipMass
        bulbMass = 2200 #[kg]
        bulbInertia = 900 #[kg-m²]
        x_αbBulb = -0.1 # [m]
        dR = (structMesh[end, :] - structMesh[end-1, :])
        elemLength = √(dR[XDIM]^2 + dR[YDIM]^2 + dR[ZDIM]^2)
        # transMat = SolverRoutines.get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength, ELEMTYPE)
        transMat = get_transMat(dR[XDIM], dR[YDIM], dR[ZDIM], elemLength)
        globalMs = FEMMethods.apply_tip_mass(globalMs, bulbMass, bulbInertia, elemLength, x_αbBulb, transMat, ELEMTYPE)
    end

    alphaCorrection = 0.0
    SOLVERPARAMS = DCFoilSolverParams(globalKs, globalMs, globalCs, zeros(2, 2), 0.0, alphaCorrection)

    dim = size(globalKs)[1]

    return FEMESH, LLSystem, FlowCond, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, LLSystem.sweepAng, WingStructModel, dim, N_R, N_MAX_Q_ITER, nModes, SOLVERPARAMS, debug
end

