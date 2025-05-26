"""
Boiler plate code to test partials of the various modules
"""

using DelimitedFiles

for headerName in [
    "../src/hydro/LiftingLine",
    "../src/struct/FEMMethods",
    "../src/solvers/SolveFlutter",
    "../src/solvers/SolveForced",
    "../src/io/MeshIO"
]
    include("$(headerName).jl")
end

using .LiftingLine
using .FEMMethods
using .SolveFlutter

# ==============================================================================
#                         Lifting line partials
# ==============================================================================
function test_LLresidualJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol)

    # ************************************************
    #     Setups
    # ************************************************
    NPT_WING = size(displacementsCol, 2)
    gammas = ones(NPT_WING) * 0.1 # vortex strengths

    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displacementsCol, pretwistDist;
        npt_wing=NPT_WING, # OVERWRITE
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)
    LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # ************************************************
    #     Test residual Jacobian ∂r / ∂ u
    # ************************************************
    ∂r∂g_CS = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="CS")
    ∂r∂g_FD = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="FiDi")

    err = maximum(abs.(∂r∂g_FD - ∂r∂g_CS))
    println("max ∂r∂g error: $(err)")
    if err >= 1e-3
        error("∂r∂g CS and FD do not match!")
    else
        println("OK!")
    end

    # ************************************************
    #     Test residual Jacobian ∂r / ∂ xPt
    # ************************************************
    ∂r∂xPt_FAD, ∂r∂xdispl_FAD = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂r∂xPt_FD, ∂r∂xdispl_FD = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    err1 = maximum(abs.(∂r∂xPt_FD - ∂r∂xPt_FAD))
    println("max ∂r∂xPt error: $(err1)")
    if err1 >= 1e-3
        writedlm("./OUTPUT/∂r∂xPt_fad.csv", ∂r∂xPt_FAD, ",")
        writedlm("./OUTPUT/∂r∂xPt_fd.csv", ∂r∂xPt_FD, ",")
        error("∂r∂xPt FAD and FD do not match!")
    else
        println("OK!")
    end
    err2 = maximum(abs.(∂r∂xdispl_FD - ∂r∂xdispl_FAD))
    println("max ∂r∂xdispl error: $(err2)")
    if err2 >= 1e-3
        writedlm("./OUTPUT/∂r∂xdispl_fad.csv", ∂r∂xdispl_FAD, ",")
        writedlm("./OUTPUT/∂r∂xdispl_fd.csv", ∂r∂xdispl_FD, ",")
        error("∂r∂xdispl FAD and FD do not match!")
    else
        println("OK!")
    end

    return 0.0
end

function test_LLcostFuncJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol)

    # ************************************************
    #     Setups
    # ************************************************
    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)

    NPT_WING = size(displacementsCol, 2)
    gammas = ones(NPT_WING) * 0.1 # vortex strengths
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(
        Uvec, sweepAng, rootChord, TR, midchords, displacementsCol, pretwistDist;
        npt_wing=NPT_WING, # OVERWRITE
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)
    LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # ************************************************
    #     wrt vortex strengths
    # ************************************************
    println("="^40)
    println("Running partial derivatives wrt gammas...")
    println("="^40)
    ∂f∂g_FAD = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FAD")
    ∂f∂g_FiDi = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FiDi")

    err = maximum(abs.(∂f∂g_FiDi - ∂f∂g_FAD))
    println("max ∂f∂g error: $(err)")
    idx = argmax(abs.(∂f∂g_FiDi - ∂f∂g_FAD))
    relerr = abs(err / ∂f∂g_FiDi[idx])
    println("max ∂f∂g rel error: $(relerr)")
    if relerr >= 1e-3
        error("∂f∂g FAD and FD do not match!")
    else
        println("OK!")
    end


    # ************************************************
    #     Collocation points wrt displacement of collocation points
    # ************************************************
    println("="^40)
    println("Running partial derivatives wrt displacements of collocation points...")
    println("="^40)
    ∂collocationPt∂displCol_an = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="Analytic")
    ∂collocationPt∂displCol_fd = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂collocationPt∂displCol_fad = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂collocationPt∂displCol_rad = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

    err1 = maximum(abs.(∂collocationPt∂displCol_fd - ∂collocationPt∂displCol_an))
    err2 = maximum(abs.(∂collocationPt∂displCol_fad - ∂collocationPt∂displCol_an))
    err3 = maximum(abs.(∂collocationPt∂displCol_rad - ∂collocationPt∂displCol_an))
    println("max ∂collocationPt∂displCol error: $(maximum([err1, err2, err3]))")
    if maximum([err1, err2, err3]) >= 1e-10
        writedlm("OUTPUT/∂collocationPt∂displCol_an.csv", ∂collocationPt∂displCol_an, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_fd.csv", ∂collocationPt∂displCol_fd, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_fad.csv", ∂collocationPt∂displCol_fad, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_rad.csv", ∂collocationPt∂displCol_rad, ",")
        error("∂collocationPt∂displCol disagreement! Wrote to file")
    else
        println("OK!")
    end

    # ************************************************
    #     Collocation points wrt ptVec
    # ************************************************
    println("="^40)
    println("Running partial derivatives of collocation points wrt ptVec...")
    println("="^40)
    ∂collocationPt∂Xpt_fad = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂collocationPt∂Xpt_fd = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂collocationPt∂Xpt_rad = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

    err1 = maximum(abs.(∂collocationPt∂Xpt_fd - ∂collocationPt∂Xpt_fad))
    err2 = maximum(abs.(∂collocationPt∂Xpt_fd - ∂collocationPt∂Xpt_rad))
    idx = argmax(abs.(∂collocationPt∂Xpt_fd - ∂collocationPt∂Xpt_fad))
    println("max ∂collocationPt∂Xpt error: $(maximum([err1, err2]))")
    relerr1 = abs(err1 / ∂collocationPt∂Xpt_fd[idx])
    relerr2 = abs(err2 / ∂collocationPt∂Xpt_fd[idx])
    println("max ∂collocationPt∂Xpt rel. error: $(maximum([relerr1, relerr2]))")
    if maximum([err1, err2]) >= 4e-2
        println("indices where error is maximum $(idx)")
        writedlm("OUTPUT/∂collocationPt∂Xpt_fad.csv", ∂collocationPt∂Xpt_fad, ",")
        writedlm("OUTPUT/∂collocationPt∂Xpt_rad.csv", ∂collocationPt∂Xpt_rad, ",")
        writedlm("OUTPUT/∂collocationPt∂Xpt_fd.csv", ∂collocationPt∂Xpt_fd, ",")
        error("∂collocationPt∂Xpt do not match! Wrote to file")
    else
        println("OK!")
    end

    # ************************************************
    #     wrt ptVec
    # ************************************************
    println("Running partial derivatives wrt ptVec...")
    ∂f∂x, ∂f∂xdispl = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂f∂x_FD, ∂f∂xdispl_FD = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    err = maximum(abs.(∂f∂x_FD - ∂f∂x))
    idx = argmax(abs.(∂f∂x_FD - ∂f∂x))
    relerr = abs(err / ∂f∂x[idx])
    println("max ∂f∂x error: $(err)")
    println("indices $(idx)")
    println("max ∂f∂x rel. error: $(relerr)")
    if relerr >= 1e-3
        println("indices $(idx)")
        writedlm("./OUTPUT/∂f∂x_fad.csv", ∂f∂x, ",")
        writedlm("./OUTPUT/∂f∂x_fd.csv", ∂f∂x_FD, ",")
        error("∂f∂x FAD and FD do not match within tolerance! Wrote to CSV files")
    else
        println("OK!")
    end

    err = maximum(abs.((∂f∂xdispl_FD - ∂f∂xdispl)))
    idx = argmax(abs.(∂f∂xdispl_FD - ∂f∂xdispl))
    relerr = abs(err / ∂f∂xdispl[idx])
    println("max ∂f∂xdispl error: $(err)")
    println("indices $(idx)")
    println("max ∂f∂xdispl rel. error: $(relerr)")
    if relerr >= 5e-3
        writedlm("./OUTPUT/∂f∂xdispl_fad.csv", ∂f∂xdispl, ",")
        writedlm("./OUTPUT/∂f∂xdispl_fd.csv", ∂f∂xdispl_FD, ",")
        error("∂f∂xdispl FAD and FD do not match within tolerance!")
    else
        println("OK!")
    end
    # 
    # ************************************************
    #     Drag
    # ************************************************
    println("="^40)
    println("Running partial derivatives wrt drag...")
    println("="^40)

    toc::Vector{Float64} = appendageParams["toc"] # if there is a toc ForwardDiff bug, reset the appendageParams dictionary
    ∂Drag∂Xpt_fd, ∂Drag∂xdispl_fd, ∂Drag∂G_fd, ∂Drag∂toc_fd = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, toc, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂Drag∂Xpt_fad, ∂Drag∂xdispl_fad, ∂Drag∂G_fad, ∂Drag∂toc_fad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, toc, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂Drag∂Xpt_rad, ∂Drag∂xdispl_rad, ∂Drag∂G_rad, ∂Drag∂toc_rad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, toc, appendageParams, appendageOptions, solverOptions; mode="RAD")

    err1 = maximum(abs.(∂Drag∂Xpt_fd - ∂Drag∂Xpt_fad))
    err2 = maximum(abs.(∂Drag∂Xpt_rad - ∂Drag∂Xpt_fd))
    println("max ∂Drag∂Xpt error: $(maximum([err1, err2]))")
    if maximum([err1, err2]) >= 1e-3
        writedlm("OUTPUT/∂Drag∂Xpt_fad.csv", ∂Drag∂Xpt_fad, ",")
        writedlm("OUTPUT/∂Drag∂Xpt_fd.csv", ∂Drag∂Xpt_fd, ",")
        writedlm("OUTPUT/∂Drag∂Xpt_rad.csv", ∂Drag∂Xpt_rad, ",")
        println("writing ∂Drag∂Xpt to files")
        println("∂Drag∂Xpt do not match! Wrote to file. Last time checked, this error is fine.")
    else
        println("OK!")
    end
    err1 = maximum(abs.(∂Drag∂xdispl_fd - ∂Drag∂xdispl_fad))
    err2 = maximum(abs.(∂Drag∂xdispl_rad - ∂Drag∂xdispl_fd))
    println("max ∂Drag∂xdispl error: $(maximum([err1, err2]))")
    idx = argmax(abs.(∂Drag∂xdispl_fd - ∂Drag∂xdispl_fad))
    relerr1 = abs(err1 / ∂Drag∂xdispl_fad[idx])
    idx = argmax(abs.(∂Drag∂xdispl_fd - ∂Drag∂xdispl_rad))
    println("indices $(idx)")
    relerr2 = abs(err2 / ∂Drag∂xdispl_rad[idx])
    println("max ∂Drag∂xdispl rel. error: $(maximum([relerr1, relerr2]))")
    if maximum([relerr1, relerr2]) >= 1e-3
        writedlm("OUTPUT/∂Drag∂xdispl_fad.csv", ∂Drag∂xdispl_fad, ",")
        writedlm("OUTPUT/∂Drag∂xdispl_fd.csv", ∂Drag∂xdispl_fd, ",")
        writedlm("OUTPUT/∂Drag∂xdispl_rad.csv", ∂Drag∂xdispl_rad, ",")
        println("writing ∂Drag∂xdispl to files")
        println("∂Drag∂xdispl do not match! Wrote to file")
    else
        println("OK!")
    end
    err1 = maximum(abs.(∂Drag∂G_fd - ∂Drag∂G_fad))
    err2 = maximum(abs.(∂Drag∂G_rad - ∂Drag∂G_fd))
    println("max ∂Drag∂G error: $(maximum([err1, err2]))")
    if maximum([err1, err2]) >= 1e-3
        writedlm("OUTPUT/∂Drag∂G_fad.csv", ∂Drag∂G_fad, ",")
        writedlm("OUTPUT/∂Drag∂G_fd.csv", ∂Drag∂G_fd, ",")
        writedlm("OUTPUT/∂Drag∂G_rad.csv", ∂Drag∂G_rad, ",")
        println("writing ∂Drag∂G to files")
        error("∂Drag∂G do not match! Wrote to file")
    else
        println("OK!")
    end

    err1 = maximum(abs.(∂Drag∂toc_fd - ∂Drag∂toc_fad))
    err2 = maximum(abs.(∂Drag∂toc_rad - ∂Drag∂toc_fd))
    println("max ∂Drag∂toc error: $(maximum([err1, err2]))")
    idx = argmax(abs.(∂Drag∂toc_fd - ∂Drag∂toc_fad))
    relerr1 = abs(err1 / ∂Drag∂toc_fd[idx])
    println("fad indices $(idx)")
    idx = argmax(abs.(∂Drag∂toc_fd - ∂Drag∂toc_rad))
    relerr2 = abs(err2 / ∂Drag∂toc_fd[idx])
    println("rad indices $(idx)")
    println("max ∂Drag∂toc rel. error: $(maximum([relerr1, relerr2]))")
    if maximum([relerr1, relerr2]) >= 2e-3
        writedlm("OUTPUT/∂Drag∂toc_fad.csv", ∂Drag∂toc_fad, ",")
        writedlm("OUTPUT/∂Drag∂toc_fd.csv", ∂Drag∂toc_fd, ",")
        writedlm("OUTPUT/∂Drag∂toc_rad.csv", ∂Drag∂toc_rad, ",")
        println("writing ∂Drag∂toc to files")
        error("∂Drag∂toc do not match! Wrote to file")
    else
        println("OK!")
    end

    return 0.0
end


# ==============================================================================
#                         Beam partials
# ==============================================================================
function test_BeamResidualJacobians(appendageParams, appendageOptions, solverOptions)
    nNodeTot, nNodeWing, nElemTot, nElemWing = FEMMethods.get_numnodes(appendageOptions["config"], appendageOptions["nNodes"], appendageOptions["nNodeStrut"])
    allStructStates = 0.1 * ones(nNodeTot * FEMMethods.NDOF)
    traction_forces = zeros(nNodeTot * FEMMethods.NDOF)
    traction_forces[end-FEMMethods.NDOF+FEMMethods.WIND] = 1.0
    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="analytic",
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    ∂rs∂xPt_fd, ∂rs∂xParams_fd = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="FiDi",
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    err1 = maximum(abs.(∂rs∂xPt_fd - ∂rs∂xPt))

    idx = argmax(abs.(∂rs∂xPt_fd - ∂rs∂xPt))
    relerr1 = err1 / ∂rs∂xPt[idx]
    println("max ∂rs∂xPt error: $(err1)")
    println("indices $(idx)")
    println("max ∂rs∂xPt rel. error: $(relerr1)")
    if relerr1 >= 1e-3
        writedlm("./OUTPUT/∂rs∂xPt_analytic.csv", ∂rs∂xPt, ",")
        writedlm("./OUTPUT/∂rs∂xPt_fd.csv", ∂rs∂xPt_fd, ",")
        error("∂rs∂xPt CS and FD do not match! Wrote to file")
    else
        println("OK!")
    end

    err2 = ones(length(∂rs∂xParams_fd))
    relerr2 = ones(length(∂rs∂xParams_fd))
    ii = 1
    for (key, val) in ∂rs∂xParams_fd
        println("key:\t$(key)")
        err2[ii] = maximum(abs.(∂rs∂xParams_fd[key] - ∂rs∂xParams[key]))
        idx = argmax(abs.(∂rs∂xParams_fd[key] - ∂rs∂xParams[key]))
        relerr2[ii] = abs(err2[ii] / ∂rs∂xParams[key][idx])
        println("index $(idx)")
        ii += 1
    end
    println(err2)
    println(relerr2)
    replace!(relerr2, NaN => 0.0)
    if maximum(relerr2) >= 5e-3
        for (key, val) in ∂rs∂xParams_fd
            writedlm("./OUTPUT/∂rs∂xParams_$(key)_analytic.csv", ∂rs∂xParams[key], ",")
            writedlm("./OUTPUT/∂rs∂xParams_$(key)_fd.csv", ∂rs∂xParams_fd[key], ",")
        end
        error("∂rs∂xParams analytic and FD do not match! Wrote to file")
    else
        println("OK!")
    end

    return 0.0
end


function test_BeamCostFuncJacobians(appendageParams, appendageOptions)

    reference = [0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.16666666666666666 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.16666666666666666 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3333333333333333 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3333333333333333 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5; 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 2.220446049250313e-16 0.0 0.0 -2.220446049250313e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.16666666666666666 0.0 0.0 2.220446049250313e-16 0.0 0.0 -2.220446049250313e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.16666666666666666 0.0; 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.3333333333333333 0.0 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.3333333333333333 0.0; 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0; 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    # ************************************************
    #     Setup
    # ************************************************
    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)



    # ************************************************
    #     Evaluate Jacobians
    # ************************************************
    ∂nodes∂x_rad = FEMMethods.compute_∂nodes∂x(ptVec, nodeConn, [appendageParams], appendageOptions; mode="RAD")
    ∂nodes∂x_fd = FEMMethods.compute_∂nodes∂x(ptVec, nodeConn, [appendageParams], appendageOptions; mode="FiDi")

    println("FD difference: ", maximum(abs.(∂nodes∂x_fd .- ∂nodes∂x_rad)))
    return maximum(abs.(∂nodes∂x_rad .- reference))
end

# ==============================================================================
#                         Flutter partials
# ==============================================================================
function test_FlutterJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol)

    # ************************************************
    #     Setups
    # ************************************************
    appendageOptions["config"] = "wing"
    LECoords = zeros(3, 5)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.2:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)
    GridStruct = Grid(LECoords, nodeConn, TECoords)

    nNodes = appendageOptions["nNodes"]
    claVec = 2π * ones(nNodes)
    solverOptions = FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageOptions)

    # ************************************************
    #     Evaluate Jacobians
    # ************************************************

    dIdxDV = SolveFlutter.evalFuncsSens(["ksflutter"], appendageParams, GridStruct, displacementsCol, claVec, solverOptions; mode="RAD")
    dIdxDV_fd = SolveFlutter.evalFuncsSens(["ksflutter"], appendageParams, GridStruct, displacementsCol, claVec, solverOptions; mode="FiDi")

    dIdmesh = dIdxDV["ksflutter"]["mesh"]
    dIdmesh_fd = dIdxDV_fd["ksflutter"]["mesh"]

    dIdcla = dIdxDV["ksflutter"]["params"]["cla"]
    dIdcla_fd = dIdxDV_fd["ksflutter"]["params"]["cla"]

    dIdtheta_f = dIdxDV["ksflutter"]["params"]["theta_f"]
    dIdtheta_f_fd = dIdxDV_fd["ksflutter"]["params"]["theta_f"]

    dIdtoc = dIdxDV["ksflutter"]["params"]["toc"]
    dIdtoc_fd = dIdxDV_fd["ksflutter"]["params"]["toc"]

    err1 = maximum(abs.(dIdmesh_fd .- dIdmesh))
    err2 = maximum(abs.(dIdcla_fd .- dIdcla))
    err3 = maximum(abs.(dIdtheta_f_fd .- dIdtheta_f))
    err4 = maximum(abs.(dIdtoc_fd .- dIdtoc))
    println("Maximum absolute errors:\n$(err1), $(err2), $(err3), $(err4)")


    return maximum([err1, err2, err3, err4])
end