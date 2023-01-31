"""
Run tests on hydro module file
"""

using ForwardDiff, ReverseDiff, FiniteDifferences
using Plots, LaTeXStrings, Printf
using LinearAlgebra

include("../src/hydro/Hydro.jl")
using .Hydro # Using the Hydro module

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
    Matrix, SweepMatrix = Hydro.compute_node_stiff(clα, b, eb, ab, U, Λ, ω, ρ, Ck)

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
    Matrix, SweepMatrix = Hydro.compute_node_damp(clα, b, eb, ab, U, Λ, ω, ρ, Ck)

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


function test_FSeffect()
    """
    Test the high-speed FS asymptotic effect
    """

    neval = 3
    # Fnh = 6
    depth = 0.5 #[m]
    chordVec = vcat(LinRange(0.12, 0.12, neval))

    Usweep = 1:1:20
    FnhVec = zeros(length(Usweep))
    cl_rc_FS = zeros(length(Usweep))
    cl_rc = zeros(length(Usweep))
    uCtr = 1
    for U∞ in Usweep
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * deg2rad(6)
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * deg2rad(6)

        FnhVec[uCtr] = U∞ / (sqrt(9.81 * depth))

        uCtr += 1
    end
    label = @sprintf("h/c =%.2f", (depth / 0.09))
    p1 = plot(FnhVec, cl_rc_FS ./ cl_rc, label=label)
    plot!(title="High Fn_h free surface effect")

    depth = 0.1 #[m]
    uCtr = 1
    for U∞ in Usweep
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * deg2rad(6)
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * deg2rad(6)

        FnhVec[uCtr] = U∞ / (sqrt(9.81 * depth))

        uCtr += 1
    end
    label = @sprintf("h/c =%.2f", (depth / 0.09))
    plot!(FnhVec, cl_rc_FS ./ cl_rc, label=label, line=:dash)


    depth = 0.05 #[m]
    uCtr = 1
    for U∞ in Usweep
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * 1 # rad
        cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, neval=neval, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * 1 # rad

        FnhVec[uCtr] = U∞ / (sqrt(9.81 * depth))

        uCtr += 1
    end
    label = @sprintf("h/c =%.2f", (depth / 0.09))
    p1 = plot!(FnhVec, [cl_rc_FS ./ cl_rc cl_rc_FS / π], label=label, layout=(2, 1))
    # plot!(
    #     title=["High Fn_h free surface effect" "2D CL"],
    #     # ylabel=["C_L/C_L(h/c-->inf)" "c_l_alpha/pi"]
    # )


    xlabel!("Fn_h")
    xlims!(0, 20)
    ylims!(0, 1)
end

function test_theodorsenDeriv()

    neval = 3 # Number of spatial nodes
    chordVec = vcat(LinRange(0.81, 0.405, neval))
    # ---------------------------
    #   Test glauert lift distribution
    # ---------------------------
    cl_α = Hydro.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=1.0, neval=neval)
    pGlauert = plot(LinRange(0, 2.7, 250), cl_α)
    plot!(title="lift slope")

    # ---------------------------
    #   Test 𝙲(k)
    # ---------------------------
    kSweep = 0.01:0.01:2

    datar = []
    datai = []
    dADr = []
    dADi = []
    dFDr = []
    dFDi = []
    for k ∈ kSweep
        datum = Hydro.compute_theodorsen(k)
        push!(datar, datum[1])
        push!(datai, datum[2])
        derivAD = ForwardDiff.derivative(Hydro.compute_theodorsen, k)
        derivFD = FiniteDifferences.forward_fdm(2, 1)(Hydro.compute_theodorsen, k)
        push!(dADr, derivAD[1])
        push!(dADi, derivAD[2])
        push!(dFDr, derivFD[1])
        push!(dFDi, derivFD[2])
    end

    # --- Derivatives ---
    dADr
    println("Forward AD:", ForwardDiff.derivative(Hydro.compute_theodorsen, 0.1))
    println("Finite difference check:", FiniteDifferences.central_fdm(5, 1)(Hydro.compute_theodorsen, 0.1))

    # --- Plot ---
    p1 = plot(kSweep, datar, label="Real")
    plot!(kSweep, datai, label="Imag")
    plot!(title="Theodorsen function")
    plot!(xlabel=L"k", ylabel=L"C(k)")
    p2 = plot(kSweep, dADr, label="Real FAD")
    plot!(kSweep, dFDr, label="Real FD", line=:dash)
    plot!(kSweep, dADi, label="Imag FAD")
    plot!(kSweep, dFDi, label="Imag FD", line=:dash)
    plot!(title="Derivatives wrt k")
    xlabel!(L"k")
    ylabel!(L"\partial C(k)/ \partial k")

    plot(p1, p2)


end
