# --- Julia 1.9---
"""
@File    :   OceanWaves.jl
@Time    :   2024/01/22
@Author  :   Galen Ng
@Desc    :   Contains ocean wave loads on a submerged hydrofoil
"""





function compute_PMwave_spectrum(Vwind, w)
    """
    Parameters
    ----------
    Vwind : 
        wind speed at height 19.4 m
    w :
        list of angular frequencies, for example: w=0.01:0.01:4.
    """
    A = 8.1e-3 * GRAV^2
    β = 0.74 * (GRAV / Vwind)^4

    S = A * w .^ (-5) .* exp(-β ./ (w .^ 4)) # PM wave spectrum [m^2 / s]

    w_m = (4 * β / 5)^0.25 # modal frequency

    l_m = 2 * pi * Vwind^2 / GRAV # modal wavelength

    var = 2.74e-3 * Vwind^4 / GRAV^2 # variance of the wave elevation

    H_sig = 0.21 * Vwind^2 / GRAV # significant wave height

    return S, w_m, l_m, var, H_sig
end

function compute_waveloads(chordLengths, Uinf, ϱ, w_e, freqspan, waveamp, h, stripWidths, claVec)
    """
    Compute wave loads on a submerged hydrofoil

    Faltinsen Eqn 6.208 with modifications to strip theory and Sears function

    freqspan - frequency sweep [rad/s]
    cbar - mean chord [m]
    Py - power spectrum
    FT - frequency spectrum
    w_e - encounter frequency
    waveamp - Aw [m] that you get from WMO sea state (is this significant wave height/amplitude?)
    h - depth
    span - full span of hydrofoil
    claVec - lift slope
    """

    ωpk = w_e # the peak frequency is the encounter frequency
    ampDist = compute_AWave(freqspan, ωpk, waveamp)
    ampDist .= 1.0 # for now, force wave to be 1.0 m

    nStrip = length(chordLengths)
    fAey = zeros(ComplexF64, length(freqspan), nStrip)
    mAey = zeros(ComplexF64, length(freqspan), nStrip)

    bi = 0.5 * chordLengths

    S0k = zeros(ComplexF64, nStrip)

    for (ii, ω) in enumerate(freqspan)

        kw = ω^2 / GRAV # [1/m] wave number

        kf = (ω * bi / Uinf) # reduced freq using wave encounter

        for (ii, kk) in enumerate(kf)
            Skvec = compute_sears(kk)
            S0k[ii] = Skvec[2]
        end

        Aω = ampDist[ii]

        # This is the depth related term for infinite depth Airy waves
        coeff = ω * Aω * exp(-kw * h)

        # ---------------------------
        #   Sectional lift loads
        # ---------------------------
        # Circulatory
        Lc = 0.5 * ϱ * Uinf * coeff .* chordLengths .* claVec .* S0k

        # Noncirculatory (added mass type)
        Lnc = 1im * ϱ * π * bi .^ 2 * w_e * coeff

        fAey[ii, :] = (Lnc .+ Lc) .* stripWidths# [N]

        mAey[ii, :] = 0.25 * Lc .* stripWidths .* chordLengths # [N-m]

        if abs(ω) < MEPSLARGE
            fAey[ii, :] = 0
            mAey[ii, :] = 0
        end
    end

    p1 = plot(freqspan, ampDist, ylims=(0, :auto), xlims=(0, 2))
    xlabel!("Frequency [rad/s]")
    ylabel!("Wave amplitude [m]")
    xlims!(0, 20)

    p2 = plot(freqspan, abs.(fAey[:, end]), xlims=(0, 2))
    ylabel!("Wave load [N]")
    title!("Wave loads at the tip of the foil")
    xlabel!("Frequency [rad/s]")
    xlims!(0, 20)

    p3 = plot(eachindex(claVec) / length(claVec), abs.(fAey[2, :]))
    # p3 = plot!(eachindex(claVec) / length(claVec), abs.(mAey[2, :]))
    ylabel!("Wave load and moment [N]")
    title!("Spanwise at ω=$(freqspan[2])")
    xlabel!("Spanwise location")

    p4 = plot(2 * eachindex(claVec) / length(claVec), chordLengths)
    ylabel!("Chord lengths [m]")
    title!("Spanwise chord")
    xlabel!("Spanwise location")

    p5 = plot(2 * eachindex(claVec) / length(claVec), claVec, ylims=(0, :auto))
    ylabel!("Chord lengths [m]")
    title!("Spanwise lift slope")
    xlabel!("Spanwise location")


    plot(p1, p2)
    savefig("WaveLoads.png")
    println("Saved wave loads plot to WaveLoads.png")
    # figure()
    # tt = tiledlayout(1,2); nexttile;
    # plot(freqv, fAey); xlim([0 1])
    # xlabel("Frequency [Hz]")
    # ylabel("Wave force [N]")
    # l1 = xline(w_e,'--b','DisplayName',['\omega_e = ' num2str(w_e) 'Hz']);
    # nexttile;
    # plot(freqv, mAey); xlim([0 1])
    # xlabel("Frequency [Hz]")
    # ylabel("Wave moment [N-m]")
    # l1 = xline(w_e,'--b','DisplayName',['\omega_e = ' num2str(w_e) 'Hz']); legend(l1);
    # string = ["Wave loads with \omega_0=" num2str(w_0) " Hz"];
    # title(tt, string);

    return fAey, mAey, ampDist

end

function compute_encounterFreq(β, ω_wave, Ufwd)
    """
    Compute the encounter frequency

    Parameters
    ----------
    β : 
        heading angle wrt waves [rad]
    ω_wave :
        wave frequency [rad/s]
    Ufwd :
        forward velocity [m/s]

    Returns
    -------
    ωₑ : 
        encounter frequency [rad/s]
    """

    ωₑ = (1 - Ufwd * ω_wave * cos(β) / GRAV) * ω_wave

    return ωₑ
end

function compute_AWave(ωRange, ωe, waveamp)
    """
    This function is just to compute an A(ω) distribution to compute deflections of the foil
    We use the output to eventually compute RAOs for the hydrofoil
    """

    # ************************************************
    #     Rayleigh distribution
    # ************************************************
    pζ = ωRange / ωe .* exp.(-0.5 * (ωRange / ωe) .^ 2) # [-] normalized version

    ampDist = waveamp * pζ # [m] distribution

    println("Rayleigh distributed amplitude spectrum \nω_pk : $(ωe)\nwaveamp : $(waveamp)")

    return ampDist
end
