# --- Julia 1.11---
"""
@File          :   make_airfoil.jl
@Date created  :   2024/09/04
@Last modified :   2024/09/08
@Author        :   Galen Ng
@Desc          :   Makes an airfoil to be used by the VPM. 
                   This reads a file (get from airfoiltools.com) and resamples it.
                   Run from root directory for this to work
"""

include("../dcfoil/mach.jl")

# ==============================================================================
#                         SETTINGS
# ==============================================================================
NPTS = 80


hydrofoil = uppercase("H105")
# hydrofoil = uppercase("E1127")
hydrofoil = uppercase("NACA0012")

# ==============================================================================
#                         MAIN
# ==============================================================================
fname1 = "$(pwd())/INPUT/PROFILES/$(hydrofoil)_pts"
fname2 = "$(pwd())/INPUT/PROFILES/$(hydrofoil)_ctrl_pts"

# --- Read airfoils ---
filefoil = "$(pwd())/INPUT/PROFILES/$(hydrofoil).dat"
rawCoords = PREFOIL.utils.readCoordFile(filefoil)

Foil = PREFOIL.Airfoil(rawCoords)

Foil.normalizeChord()

ptsCoords = Foil.getSampledPts(
    nPts=NPTS,
    spacingFunc=PREFOIL.sampling.conical,
    func_args=Dict("coeff" => 1),
    TE_knot=false
)
ctrlCoords = (ptsCoords[1:end-1, :] .+ ptsCoords[2:end, :]) .* 0.5


# --- Write airfoil to dat ---
Foil.writeCoords(fname1, coords=ptsCoords, file_format="dat")
Foil.writeCoords(fname2, coords=ctrlCoords, file_format="dat")