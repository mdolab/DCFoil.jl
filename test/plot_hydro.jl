# --- Julia 1.7---
"""
@File    :   plot_hydro.jl
@Time    :   2023/03/29
@Author  :   Galen Ng
@Desc    :   For tests that require plotting rather than comparing to a reference value
"""

# --- Imports ---
include("../src/hydro/HydroStrip.jl")
# include("src/hydro/HydroStrip.jl")
using .HydroStrip # Using the Hydro module
using PythonCall
using LinearAlgebra
using Plots, Printf

# ==============================================================================
#                         Functions
# ==============================================================================

function test_theofs()
    """
    Comparison of submerged forces in theodorsen to ROM of forces with FS effect
    """
    # ************************************************
    #     Problem setup
    # ************************************************
    plt = pyimport("matplotlib.pyplot")
    niceplots = pyimport("niceplots")
    kSweep = [1e-10, 0.02, 0.1, 0.2, 1.0, 2.0]
    b = 1.0f0
    clα = 2π
    eb = 0.0 * b
    ab = -0.5 * b # pitch about 1/4 chord, elastic axis is at 1/4 chord
    Uinf = 10.0f0
    Λ = 0.0f0
    rho_f = 1.0f0
    q = 0.5 * rho_f * Uinf^2
    AA = deg2rad(10) # pitch amplitude
    alfa0 = 0 # mean AOA
    hA = 0.0f0


    for k in kSweep
        ω = k * Uinf / b # freq of oscillation
        T = 2π / ω # period of oscillation
        tSweep = 0.0:T*0.01:T # time sweep
        tNDSweep = tSweep ./ T # non-dim time sweep
        AOA = alfa0 .+ AA * cos.(ω * tSweep) # AOA time
        dAOA = -AA * ω * sin.(ω * tSweep) # pitch
        ddAOA = -AA * ω^2 * cos.(ω * tSweep) # pitch rate
        h = hA * sin.(ω * tSweep)
        dh = hA * ω * cos.(ω * tSweep)
        ddh = -hA * ω^2 * sin.(ω * tSweep)
        u = hcat(h, AOA)
        du = hcat(dh, dAOA)
        ddu = hcat(ddh, ddAOA)
        Ck_r, Ck_i = HydroStrip.compute_theodorsen(k)
        Ck = Ck_r + Ck_i * im
        clHist = Vector{Float64}()
        cmHist = Vector{Float64}()
        clROMHist = Vector{Float64}()
        cmROMHist = Vector{Float64}()

        # Assume span is 1m

        # ---------------------------
        #   Submerged vals
        # ---------------------------
        # --- Stiffness forces ---
        Kmat, _ = HydroStrip.compute_node_stiff(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        # --- Damping forces ---
        Cmat, _ = HydroStrip.compute_node_damp(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        Mmat = HydroStrip.compute_node_mass(b, ab, rho_f)
        # ---------------------------
        #   FS effect
        # ---------------------------
        # Now do the same with the FS effect for stiffness forces but this Ck is purely real
        hcRatio = 1.0
        Cls = HydroStrip.compute_clsROM(k, hcRatio, 4.0)
        # Kmat1, _ = HydroStrip.compute_node_stiff(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        Cms = HydroStrip.compute_cmsROM(k, hcRatio, 4.0)
        # Kmat2, _ = HydroStrip.compute_node_stiff(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        # KmatROM = vcat(Kmat1[1, :], Kmat2[2, :])

        # Damping
        Cld = HydroStrip.compute_cldROM(k, hcRatio, 4.0)
        # Cmat1, _ = HydroStrip.compute_node_damp(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        Cmd = HydroStrip.compute_cmdROM(k, hcRatio, 4.0)
        # Cmat2, _ = HydroStrip.compute_node_damp(clα, b, eb, ab, Uinf, Λ, rho_f, Ck)
        # CmatROM = vcat(Cmat1[1, :], Cmat2[2, :])

        for (ii, t) in enumerate(tSweep)
            sForces = -Kmat * u[ii, :] # matrix of non-dimensional fluid stiffness lift
            dForces = -Cmat * du[ii, :] # matrix of non-dimensional fluid stiffness lift
            iForces = -Mmat * ddu[ii, :] # matrix of non-dimensional fluid stiffness lift
            liftperlength = sForces[1] + dForces[1] + iForces[1]
            momperlength = sForces[2] + dForces[2] + iForces[2]
            clTotal = real(liftperlength) / (q * 2 * b)
            cmTotal = real(momperlength) / (q * 4 * b * b)
            push!(clHist, clTotal)
            push!(cmHist, cmTotal)

            ClROM = Cls * AOA[ii] + Cld * dAOA[ii]
            CmROM = Cms * AOA[ii] + Cmd * dAOA[ii]

            liftperlength = iForces[1]
            clTotalROM = real(liftperlength) / (q * 2 * b) + ClROM
            momperlength = iForces[2]
            cmTotalROM = real(momperlength) / (q * 4 * b * b) + CmROM
            # if k >= 1.0 # just take the theodorsen values
            #     clTotalROM = clTotal
            #     cmTotalROM = cmTotal
            # end
            push!(clROMHist, clTotalROM)
            push!(cmROMHist, cmTotalROM)
        end


        # plt.style.use(niceplots.get_style())
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=true, constrained_layout=true, figsize=(7, 11))
        # ax = axes[1]
        # ax.plot(tNDSweep, clHist, "--", label="Theodorsen " * L"h/c=\infty")
        # ax.plot(tNDSweep, clROMHist, "-", label="ROM " * L"h/c=1")
        # ax.set_ylabel(L"C_{L}", rotation=0, labelpad=20)
        # ax.set_ylim(-1.5, 1.5)
        # ax.legend(labelcolor="linecolor", frameon=false)

        # ax = axes[2]
        # ax.plot(tNDSweep, cmHist, "--", label="Theodorsen " * L"h/c=\infty")
        # ax.plot(tNDSweep, cmROMHist, "-", label="ROM " * L"h/c=1")
        # ax.set_ylabel(L"C_{M}", rotation=0, labelpad=20)
        # ax.set_ylim(-0.06, 0.06)
        # if k > 0.4
        #     ax.set_ylim(-0.3, 0.3)
        # end

        # ax = axes[3]
        # ax.plot(tNDSweep, rad2deg.(AOA), "r-", label=L"\alpha")
        # ax.plot(tNDSweep, rad2deg.(dAOA), "g-", label=L"\dot{\alpha}")
        # ax.set_ylabel(L"\alpha [\degree]" * "\n" * L"\dot{\alpha} [\degree/s]", rotation=0, labelpad=20)
        # ax.legend(labelcolor="linecolor", frameon=false)
        # ax.set_xlabel(L"t/T [-]")

        # fname = @sprintf("theodorsen_fs-k%.2f.png", k)
        # plt.suptitle(@sprintf("Theodorsen forces, k=%.2f", k))
        # plt.savefig(fname)
        # plt.close()
    end

    # newkSweep = 0.01:0.01:1.0
    # CkSweep = Vector{Float64}()
    # CklsSweep = Vector{Float64}()
    # CkldSweep = Vector{Float64}()
    # CkmsSweep = Vector{Float64}()
    # CkmdSweep = Vector{Float64}()
    # hcRatio = 1.0
    # for k in newkSweep
    #     Ck_r, Ck_i = HydroStrip.compute_theodorsen(k)
    #     push!(CkSweep, Ck_r)
    #     Ck = HydroStrip.compute_clsROM(k, hcRatio, 4.0)
    #     push!(CklsSweep, Ck)
    #     Ck = HydroStrip.compute_cmsROM(k, hcRatio, 4.0)
    #     push!(CkmsSweep, Ck)

    #     Ck = HydroStrip.compute_cldROM(k, hcRatio, 4.0)
    #     push!(CkldSweep, Ck)
    #     Ck = HydroStrip.compute_cmdROM(k, hcRatio, 4.0)
    #     push!(CkmdSweep, Ck)

    # end
    # plt.style.use(niceplots.get_style())
    # fig, axes = plt.subplots(nrows=1, ncols=1, sharex=true, constrained_layout=true, figsize=(6, 4))
    # axes.plot(newkSweep, CkSweep, "-", label="Theodorsen")
    # axes.plot(newkSweep, CklsSweep, "--", label="cls")
    # axes.plot(newkSweep, CkmsSweep, "--", label="cms")
    # axes.plot(newkSweep, CkldSweep, "--", label="cld")
    # axes.plot(newkSweep, CkmdSweep, "--", label="cmd")
    # axes.legend(labelcolor="linecolor", frameon=false)
    # axes.set_ylabel(L"C(k)", rotation=0, labelpad=20)
    # axes.set_xlabel(L"k")

    # fname = @sprintf("theodorsen_fs_hc%.2f.png", hcRatio)
    # plt.suptitle(@sprintf("Theodorsen function h/c=%.1f", hcRatio))
    # plt.savefig(fname)
    # plt.close()

end

function test_FSeffect()
    """
    Test the high-speed FS asymptotic effect
    """

    nNodes = 3
    # Fnh = 6
    depth = 0.5 #[m]
    chordVec = vcat(LinRange(0.12, 0.12, nNodes))

    Usweep = 1:1:20
    FnhVec = zeros(length(Usweep))
    cl_rc_FS = zeros(length(Usweep))
    cl_rc = zeros(length(Usweep))
    uCtr = 1
    for U∞ in Usweep
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * deg2rad(6)
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * deg2rad(6)

        FnhVec[uCtr] = U∞ / (√(9.81 * depth))

        uCtr += 1
    end
    label = @sprintf("h/c =%.2f", (depth / 0.09))
    p1 = plot(FnhVec, cl_rc_FS ./ cl_rc, label=label)
    plot!(title="High Fn_h free surface effect")

    depth = 0.1 #[m]
    uCtr = 1
    for U∞ in Usweep
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * deg2rad(6)
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * deg2rad(6)

        FnhVec[uCtr] = U∞ / (√(9.81 * depth))

        uCtr += 1
    end
    label = @sprintf("h/c =%.2f", (depth / 0.09))
    plot!(FnhVec, cl_rc_FS ./ cl_rc, label=label, line=:dash)


    depth = 0.05 #[m]
    uCtr = 1
    for U∞ in Usweep
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=true)
        cl_rc_FS[uCtr] = cl_α[1] * 1 # rad
        cl_α, _, _ = HydroStrip.compute_glauert_circ(semispan=2.7, chordVec=chordVec, α₀=6.0, U∞=U∞, nNodes=nNodes, h=depth, useFS=false)
        cl_rc[uCtr] = cl_α[1] * 1 # rad

        FnhVec[uCtr] = U∞ / (√(9.81 * depth))

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

    savefig("FSeffect.png")
end

function test_sears()
    # plt = pyimport("matplotlib.pyplot")
    # niceplots = pyimport("niceplots")
    # ************************************************
    #     Re-Im vector diagram
    # ************************************************
    # Now get marker points
    SkSweep = []
    CkSweep = []
    labels = []
    kPlot = 1e-6:0.05:10.5
    for k in kPlot
        Sk, S0k = HydroStrip.compute_sears(k)
        ans = HydroStrip.compute_theodorsen(k)
        Ck = ans[1] + im * ans[2]
        push!(SkSweep, Sk)
        push!(CkSweep, Ck)
        push!(labels, @sprintf("%.1f", k))
    end
    plot(real(SkSweep), imag(SkSweep), seriestype=:scatter, label="", marker=:circle, markersize=3, color=:blue)
    plot!(real(CkSweep), imag(CkSweep), seriestype=:scatter, label="", marker=:circle, markersize=3, color=:red)
    offset = 0.1
    for ii in eachindex(labels)
        annotate!(real(SkSweep[ii]), imag(SkSweep[ii]), text(labels[ii], 8, :left, :bottom, offset))
        annotate!(real(CkSweep[ii]), imag(CkSweep[ii]), text(labels[ii], 8, :left, :bottom, offset))
    end
    kSweep::Vector = 1e-6:0.01:20.0
    SkSweep = []
    CkSweep = []
    for k in kSweep
        Sk, S0k = HydroStrip.compute_sears(k)
        ans = HydroStrip.compute_theodorsen(k)
        Ck = ans[1] + im * ans[2]
        push!(SkSweep, Sk)
        push!(CkSweep, Ck)
    end


    # Plot functions
    plot!(real(SkSweep), imag(SkSweep), label="Sears", xlabel="Re", ylabel="Im", color=:blue)
    plot!(real(CkSweep), imag(CkSweep), label="Theodorsen", color=:red)


    xlims!(-0.4, 1)
    ylims!(-0.4, 0.4)
    plot!(tick_direction=:out)
    title!("Vector diagrams")
    savefig("vector-diagram.png")

    # ************************************************
    #     Re and im
    # ************************************************
    plot(kSweep, real.(SkSweep), label="F_s", xlabel="k", ylabel="F,G", color=:blue)
    plot!(kSweep, imag.(SkSweep), label="G_s", color=:red)
    plot!(kSweep, real.(CkSweep), label="F_c", color=:blue, linestyle=:dash)
    plot!(kSweep, imag.(CkSweep), label="G_c", color=:red, linestyle=:dash)
    plot!(tick_direction=:out)
    title!("Real (F) and imaginary (G) parts")
    savefig("real-imag-mag.png")

    # ************************************************
    #     Magnitude and phase
    # ************************************************
    p1 = plot(kSweep, abs.(SkSweep), label="|S(k)|", xlabel="k", ylabel="|H(k)|", color=:blue)
    plot!(kSweep, abs.(CkSweep), label="|C(k)|", color=:red)
    plot!(tick_direction=:out)
    p2 = plot(kSweep, rad2deg.(angle.(SkSweep)), label="∠S(k)", xlabel="k", ylabel="phase angle", color=:blue)
    ylims!(-50, 50)
    plot!(kSweep, rad2deg.(angle.(CkSweep)), label="∠C(k)", color=:red)
    plot(p1, p2, layout=(2, 1), tick_direction=:out)
    savefig("mag-phase.png")

    save("circulation.jld2", "SkSweep_re", real.(SkSweep), "SkSweep_im", imag.(SkSweep), "CkSweep_re", real.(CkSweep), "CkSweep_im", imag.(CkSweep), "kSweep", vec(kSweep))
end

# ==============================================================================
#                         Driver code
# ==============================================================================
# test_theodorsenDeriv()
# test_FSeffect()
# test_theofs()
test_sears()