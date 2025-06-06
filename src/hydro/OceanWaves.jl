# --- Julia 1.9---
"""
@File    :   OceanWaves.jl
@Time    :   2024/01/22
@Author  :   Galen Ng
@Desc    :   Contains ocean wave loads on a submerged hydrofoil
"""


function compute_PMWaveSpectrum(Hsig, w)
    """
    One parameter  Pierson--Moskowitz wave energy spectrum
    for fully developed seas in the North Atlantic
    Parameters
    ----------
    Vwind : 
        wind speed at height 19.4 m
    Hsig :
        significant wave height [m]
    w :
        list of angular frequencies, for example: w=0.01:0.01:4.
    """

    Vwind = (Hsig * GRAV / 0.2092)^0.5 # wind speed [m/s]
    # H_sig = 0.2092 * Vwind^2 / GRAV # significant wave height

    A = 8.1e-3 * GRAV^2
    β = 0.74 * (GRAV / Vwind)^4

    S = A * w .^ (-5) .* exp.(-β ./ (w .^ 4)) # PM wave spectrum [m^2 - s]

    w_m = (4 * β / 5)^0.25 # modal frequency

    l_m = 2 * pi * Vwind^2 / GRAV # modal wavelength

    var = 2.74e-3 * Vwind^4 / GRAV^2 # variance of the wave elevation

    return S, w_m, l_m, var, Vwind
end

function compute_BSWaveSpectrum(Hsig, ω_z, w)
    """
    Two parameter Bretschneider spectrum for decaying to developing seas
    A.k.a. the modified Pierson-Moskowitz spectrum or ISSC spectrum
    Alternative forms use the significant zero-crossing frequency

    Parameters
    ----------
    Hsig : 
        significant wave height [m]
    ω_m :
        modal frequency [rad/s]

    ω_z :
        significant zero-crossing frequency [rad/s] is related to the average of the one-third largest zero-crossing periods

    w :
        list of angular frequencies, for example: w=0.01:0.01:4.
    """

    # S = 0.3125 * ω_m^4 * w .^ (-5) * Hsig^2 .* exp.(-1.25 * (ω_m ./ w) .^ 4) # Bretschneider wave spectrum [m^2 - s]

    S = 0.11 * Hsig^2 * ω_z^4 * w .^ (-5) .* exp.(-0.44 * (ω_z ./ w) .^ 4) # alternative form using the significant crossing frequency

    return S
end

function compute_waveloads(chordLengths, Uinf, ϱ, ω_e, freqspan, waveamp, h, stripWidths, claVec; method="Sears", debug=false) """
    Compute wave loads on a submerged hydrofoil

    Faltinsen Eqn 6.208 with modifications to strip theory and Sears function

    freqspan - frequency sweep [rad/s]
    cbar - mean chord [m]
    w_e - encounter frequency
    waveamp - Aw [m] that you get from WMO sea state (is this significant wave height/amplitude?)
    h - depth
    span - full span of hydrofoil
    claVec - lift slope
    """

    # ωpk = w_e # the peak frequency is the significant encounter frequency
    # println("ωpk: ", ωpk)
    # println("freqspan: ", freqspan)
    # ampDist = compute_AWave(freqspan, 0.125, waveamp)
    # ampDist = 1.0 # for now, force wave to be 1.0 m. This would then give the RAO.
    ampDist = ones(length(freqspan)) # [m] wave amplitude distribution
    # In other words, the transfer function of 1m input wave to load output

    nStrip = length(chordLengths)
    fAey = zeros(ComplexF64, length(freqspan), nStrip)
    mAey = zeros(ComplexF64, length(freqspan), nStrip)
    fAey_z = Zygote.Buffer(fAey)
    mAey_z = Zygote.Buffer(mAey)

    bi = 0.5 * chordLengths


    for (ii, ω) in enumerate(freqspan)

        kw = ω^2 / GRAV # [1/m] wave number

        kf = (ω_e[ii] * bi / Uinf) # reduced freq using wave encounter frequency


        Aω = ampDist[ii]

        # This is the depth related term for infinite depth Airy waves
        coeff = ω * Aω * exp(-kw * h)

        # ---------------------------
        #   Sectional lift loads
        # ---------------------------
        if uppercase(method) == "SEARS" # most accurate
            Lc, Lnc = HydroStrip.compute_gustLoadSears(kf, Uinf, ϱ, ω_e[ii], coeff, claVec, chordLengths)
        elseif uppercase(method) == "THEODORSEN"
            Lc, Lnc = HydroStrip.compute_gustLoadTheodorsen(kf, Uinf, ϱ, ω_e[ii], coeff, claVec, chordLengths)
        elseif uppercase(method) == "QUASISTEADY"
            Lc, Lnc = HydroStrip.compute_gustLoadQuasisteady(kf, Uinf, ϱ, ω_e[ii], coeff, claVec, chordLengths)
        else
            error("Unknown method: $(method)")
        end

        fAey_z[ii, :] = (Lnc .+ Lc) .* stripWidths# [N]

        mAey_z[ii, :] = 0.25 * Lc .* stripWidths .* chordLengths # [N-m]

        if abs(ω) < MEPSLARGE
            fAey_z[ii, :] = 0
            mAey_z[ii, :] = 0
        end
    end

    fAey = copy(fAey_z)
    mAey = copy(mAey_z)

    if debug
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
        # println("Saved wave loads plot to WaveLoads.png")
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
    end

    return fAey, mAey, ampDist

end

function compute_encounterFreq(β, ω_wave::AbstractVector, Ufwd)
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
    ωₑ = (1 .- Ufwd * ω_wave * cos(β) / GRAV) .* ω_wave

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

    # println("Rayleigh distributed amplitude spectrum \nω_pk : $(ωe)\nwaveamp : $(waveamp)")

    return ampDist
end

function compute_responseSpectralDensityFunc(Hω::Vector{<:Real}, waveEnergySpectrum::Vector{<:Real})
    """
    waveEnergySpectrum : [m^2 - s] wave energy spectrum
    Hω : FRF magnitude of response amplitude operator
    """

    S_R⁺ = Hω .^ 2 .* waveEnergySpectrum

    return S_R⁺
end