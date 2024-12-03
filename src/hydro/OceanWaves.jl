# --- Julia 1.9---
"""
@File    :   OceanWaves.jl
@Time    :   2024/01/22
@Author  :   Galen Ng
@Desc    :   Contains ocean wave loads on a submerged hydrofoil
"""


module OceanWaves

function PMwave_spectrum(Vwind, w)
    """
    Parameters
    ----------
    Vwind : 
        wind speed at height 19.4 m
    w :
        list of angular frequencies, for example: w=0.01:0.01:4.
    """
    g = 9.81
    A = 8.1e-3 * g^2
    B = 0.74 * (g / Vwind)^4
    S = A * w .^ (-5) .* exp(-B ./ (w .^ 4)) # PM wave spectrum
    w_m = (4 * B / 5)^0.25 # modal frequency
    l_m = 2 * pi * Vwind^2 / g # modal wavelength
    var = 2.74e-3 * Vwind^4 / g^2 # variance of the wave elevation
    H_sig = 0.21 * Vwind^2 / g # significant wave height
    return S, w_m, l_m, var, H_sig
end

function [fAey, mAey] = waveloads(cbar, Uo, rho_f, w_e, w_0, freqv, waveamp, h, span)
    # freqspan - frequency sweep
    # cbar - mean chord [m]
    # Py - power spectrum
    # FT - frequency spectrum
    # w_e - encounter frequency
    # h - depth
    
    # TODO: FINISH DEBUGGING
    rayleighDist = (freqv / (w_e)) .* exp(-0.5 * (freqv / (w_e)).^2); # [-] normalized version
    ampDist = waveamp * rayleighDist; # [m] distribution
    
    k = w_0^2/9.81; # [1/m]
    fAey = zeros(size(freqv));
    mAey = zeros(size(freqv));
    
    # figure()
    # plot(freqv, ampDist); xlim([0 1]); hold on
    # xlabel("Frequency [Hz]")
    # ylabel("Wave amplitude [m]")
    # l1 = xline(w_e,'--b','DIsplayName',['\omega_e = ' num2str(w_e) 'Hz']); legend(l1);
    # title(["Wave spectrum \omega_{wave}=" num2str(w_0)])
    
    ii = 1;
    for w = freqv
        kf = (0.5*w*cbar/Uo); # reduced freq using wave encounter
        CK = besselh(1, 2, kf) / (besselh(1, 2, kf) + 1i * besselh(0, 2, kf));
        fAey(ii) = abs(0.25* rho_f* (pi*cbar^2*span * 1i * w_0 * w_e * ampDist(ii) * exp(-k*h)) ...
            + rho_f * pi * Uo * cbar * CK * span * w_0 * ampDist(ii) * exp(-k*h));
        mAey(ii) = fAey(ii) * 0.25 * cbar; #[N-m]
        
        if w == 0
            fAey(ii) = 0; mAey(ii) = 0;
        end
        ii = ii + 1;
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
    
end

end # end module