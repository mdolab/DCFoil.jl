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

    ω_char = √(GRAV / (0.5 * meanChord)) # [Hz] characteristic frequency for nondimensionalization

    # Hωsquare = real.(Zω).^2 .+ imag.(Zω).^2 # transfer function for the deflection response

    Zωbend = Zω[:, WIND:NDOF:end] # bending shape
    Zωtwist = Zω[:, ΘIND:NDOF:end] # twisting shape

    # sum over space
    Zωsumbend = vec(sum(Zωbend, dims=2)) # bending shape
    Zωsumtwist = vec(sum(Zωtwist, dims=2)) # twisting shape

    # PSD : [m^2 - sec] Power Spectral Density of the deformations in response to the wave energy spectrum
    PSDbend = compute_responseSpectralDensityFunc(Zωsumbend, waveEnergySpectrum)
    PSDtwist = compute_responseSpectralDensityFunc(Zωsumtwist, waveEnergySpectrum)

    # p1 = plot(fSweep, PSDbend, label="Bending PSD")
    # p2 = plot(fSweep, PSDtwist, label="Twisting PSD")
    # p3 = plot(fSweep, waveEnergySpectrum, label="Wave Energy Spectrum", xlabel="Frequency [Hz]", ylabel="Power Spectral Density [m^2 - sec]")
    # p4 = plot(fSweep, Hωsumbend, label="Bending Transfer Function", xlabel="Frequency [Hz]", ylabel="Transfer Function Magnitude")
    # plot(p1, p2, p3, p4, layout=(2, 2), xlabel="Frequency [Hz]")
    # savefig("PSDbend_twist.png")

    gvib_bend = (2π)^2 * sum(PSDbend .* fSweep .* df) / (meanChord^2 * ω_char)
    gvib_twist = (2π)^2 * sum(PSDtwist .* fSweep .* df) / (ω_char)

    return gvib_bend, gvib_twist
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

    ksbend = compute_KS(dynBending, solverOptions["rhoKS"])
    kstwist = compute_KS(dynTwisting, solverOptions["rhoKS"])

    return ksbend, kstwist
end
