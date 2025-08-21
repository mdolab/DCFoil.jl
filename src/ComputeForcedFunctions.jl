# --- Julia 1.11---
"""
@File          :   ComputeForcedFunctions.jl
@Date created  :   2024/11/22
@Last modified :   2025/02/10
@Author        :   Galen Ng
@Desc          :   Compute cost functions from the forced vibration response
"""

function compute_PSDArea(Zω, fSweep, meanChord, waveEnergySpectrum)
    """

    Compute the area under the PSD curve nondimensionalized
    This is effectively the zeroth spectral moment
    The value is really small

    Inputs
    ------
    Zω : 
        Vibration solution object containing the deflection transfer function
    fSweep :


    """

    df = fSweep[2] - fSweep[1] # [Hz] frequency increments

    f_char = √(GRAV / (0.5 * meanChord)) # [Hz] characteristic frequency for nondimensionalization

    # Hωsquare = real.(Zω).^2 .+ imag.(Zω).^2 # transfer function for the deflection response

    Zωbend = Zω[:, WIND:NDOF:end] # bending shape
    Zωtwist = Zω[:, ΘIND:NDOF:end] # twisting shape

    gvib_bend = 0.0 # bending vibration func
    gvib_twist = 0.0 # twisting vibration func
    for (ii, Zωbendi) in enumerate(eachcol(Zωbend)) # loop over space

        # PSD : [m^2 - sec] Power Spectral Density of the deformations in response to the wave energy spectrum
        PSDbendi = compute_responseSpectralDensityFunc(vec(Zωbendi), waveEnergySpectrum)
        PSDtwisti = compute_responseSpectralDensityFunc(Zωtwist[:, ii], waveEnergySpectrum)

        # m0 = sum(PSDbendi * df) # zeroth spectral moment
        # println("m0: $m0")
        # m1 = sum(PSDbendi .* fSweep * df) # first spectral moment
        # println("m1: $m1")

        gvib_bend += sum(PSDbendi .* fSweep * df) / ((0.5 * meanChord)^2 * f_char)
        gvib_twist += sum(PSDtwisti .* fSweep * df) / (f_char)
    end
    # Zωsumbend = vec(sum(Zωbend, dims=2)) # bending shape
    # Zωsumtwist = vec(sum(Zωtwist, dims=2)) # twisting shape

    # # PSD : [m^2 - sec] Power Spectral Density of the deformations in response to the wave energy spectrum
    # PSDbend = compute_responseSpectralDensityFunc(Zωsumbend, waveEnergySpectrum)
    # PSDtwist = compute_responseSpectralDensityFunc(Zωsumtwist, waveEnergySpectrum)

    # gvib_bend = sum(PSDbend .* fSweep * df) / (meanChord^2 * f_char)
    # gvib_twist = sum(PSDtwist .* fSweep * df) / (f_char)

    scaler = 1e9

    return gvib_bend * scaler, gvib_twist * scaler
end

function compute_dynDeflectionPk(dynStructStates, solverOptions)
    """
    Average the dynamic deflection amplitudes over the frequency sweep, then find the maximum of that
    Done for both bending and twisting shapes respectively
    """

    dynH = dynStructStates[:, WIND:NDOF:end]
    dynTheta = dynStructStates[:, ΘIND:NDOF:end]

    dynBending = real.(dynH) .^ 2 + imag(dynH) .^ 2 # bending shape squared
    dynTwisting = real.(dynTheta) .^ 2 + imag.(dynTheta) .^ 2 # twisting shape squared

    ksTmpBend = zeros(size(dynBending, 1))
    ksTmpBend_z = Zygote.Buffer(ksTmpBend)
    ksTmpTwist = zeros(size(dynBending, 1))
    ksTmpTwist_z = Zygote.Buffer(ksTmpTwist)

    for (ifreq, dynBω) in enumerate(eachrow(dynBending)) # loop over frequencies

        ksbend = compute_KS(dynBω, solverOptions["rhoKS"])
        kstwist = compute_KS(dynTwisting[ifreq, :], solverOptions["rhoKS"])
        # Find the maximum of that
        ksTmpBend_z[ifreq] = ksbend
        ksTmpTwist_z[ifreq] = kstwist

    end

    ksTmpBend = copy(ksTmpBend_z)
    ksTmpTwist = copy(ksTmpTwist_z)

    ksbend = compute_KS(ksTmpBend, solverOptions["rhoKS"])
    kstwist = compute_KS(ksTmpTwist, solverOptions["rhoKS"])

    return ksbend, kstwist
end
