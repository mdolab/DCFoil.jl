# --- Julia 1.9---
"""
@File    :   OceanWaves.jl
@Time    :   2024/01/22
@Author  :   Galen Ng
@Desc    :   Contains ocean wave loads on a submerged hydrofoil
"""


module OceanWaves

using ..Unsteady: compute_sears
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE, GRAV

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
    h - depth
    span - full span of hydrofoil
    claVec - lift slope
    """

    ampDist = compute_AWave(freqspan, ωpk, waveamp)

    nStrip = length(chordLengths)
    fAey = zeros(size(freqspan), nStrip)
    mAey = zeros(size(freqspan), nStrip)

    # figure()
    # plot(freqv, ampDist); xlim([0 1]); hold on
    # xlabel("Frequency [Hz]")
    # ylabel("Wave amplitude [m]")
    # l1 = xline(w_e,'--b','DIsplayName',['\omega_e = ' num2str(w_e) 'Hz']); legend(l1);
    # title(["Wave spectrum \omega_{wave}=" num2str(w_0)])
    bi = 0.5 * chordLengths

    for (ii, ω) in enumerate(freqspan)

        kw = ω^2 / GRAV # [1/m] wave number

        kf = (ω * bi / Uinf) # reduced freq using wave encounter

        _, S0k = compute_sears.(kf)

        Aω = ampDist[ii]

        coeff = ω * Aω * exp(-kw * h)

        # ---------------------------
        #   Sectional lift loads
        # ---------------------------
        # Circulatory
        Lc = 0.5 * ϱ * Uinf * chordLengths .* claVec .* S0k * coeff

        # Noncirculatory (added mass type)
        Lnc = 1im * ϱ * π * bi .^ 2 * w_e * coeff

        fAey[ii, :] = (Lnc .+ Lc) .* stripWidths# [N]

        mAey[ii, :] = 0.25 * Lc .* stripWidths .* chordLengths # [N-m]

        if abs(ω) < MEPSLARGE
            fAey[ii] = 0
            mAey[ii] = 0
        end
    end

    # figure()
    # tt = tiledlayout(1,2); nexttile;
    # plot(freqv, fAey); xlim([0 1])
    # xlabel("Frequency [Hz]")
    # ylabel("Wave force [N]")
    # l1 = xline(w_e,'--b','DIsplayName',['\omega_e = ' num2str(w_e) 'Hz']);
    # nexttile;
    # plot(freqv, mAey); xlim([0 1])
    # xlabel("Frequency [Hz]")
    # ylabel("Wave moment [N-m]")
    # l1 = xline(w_e,'--b','DIsplayName',['\omega_e = ' num2str(w_e) 'Hz']); legend(l1);
    # string = ["Wave loads with \omega_0=" num2str(w_0) " Hz"];
    # title(tt, string);

    return fAey, mAey

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
    pζ = ωRange / ωe .* exp(-0.5 * (ωRange / ωe) .^ 2) # [-] normalized version

    ampDist = waveamp * pζ # [m] distribution

    return ampDist
end

end # end module