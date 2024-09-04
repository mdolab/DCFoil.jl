# --- Julia 1.7---
"""
@File          :   make_airfoil.jl
@Date created  :   2024/09/04
@Last modified :   2024/09/04
@Author        :   Galen Ng
@Desc          :   Makes an airfoil to be used by the VPM. This reads a file and resamples it.
"""

include("../../dcfoil/mach.jl")

hydrofoil = uppercase("H105")
hydrofoil = uppercase("E1127")

fname1 = "$(pwd())/INPUT/PROFILES/$(hydrofoil)_pts"
fname2 = "$(pwd())/INPUT/PROFILES/$(hydrofoil)_ctrl_pts"

filefoil = "$(pwd())/INPUT/PROFILES/$(hydrofoil).dat"
rawCoords = prefoil.utils.readCoordFile(filefoil)
Foil = prefoil.Airfoil(rawCoords)

Foil.normalizeChord()

ptsCoords = Foil.getSampledPts(
    nPts=80,
    spacingFunc=prefoil.sampling.conical,
    func_args=Dict("coeff" => 1),
    TE_knot=false
)
ctrlCoords = (ptsCoords[1:end-1, :] .+ ptsCoords[2:end, :]) .* 0.5


# --- Write airfoil to dat ---
Foil.writeCoords(fname1, coords=ptsCoords, file_format="dat")
Foil.writeCoords(fname2, coords=ctrlCoords, file_format="dat")