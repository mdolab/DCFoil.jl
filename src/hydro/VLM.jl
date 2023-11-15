# --- Julia 1.9---
"""
@File    :   VLM.jl
@Time    :   2023/11/15
@Author  :   Galen Ng
@Desc    :   This is based on the Propeller Vortex Lattice (PVL) open source codes from Justin Kerwin (from Julie Young's hydrofoils course)
"""

module VLM

function init(dvdict::Dict, solverOptions::Dict)
    """
    Initialize the VLM module

    Inputs
    ------
    dvDict - These are the current entries of the dict:
        'MT': number of panels radially
        'ITER': number of wake alignment iterations
        'RHV' : hub vortex radius
        'NBLADE': number of blades
        'J': advance coefficient
    solverOptions - 
    """

end

function forces()
    """
    NBLADE,MCP,ADVCO,WAKE,RV,RC,TANBC,UASTAR,UTSTAR,VA,CHORD,CD,G,RHV,IHUB
    """

    CD_LD = 1 # Default: Input CD interpreted as viscous drag coefficient
    if (CD[1] > 1.0)
        CD_LD = 0 # CD(1)>1 signals that input is L/D
    end

    CT = 0.0
    CQ = 0.0

    # ************************************************
    #     Loop over propeller radius 'panels'
    # ************************************************
    # TODO: redo this to loop over the struct like this
    # for (mm, PropSec) in enumerate(PropSections)
    for M = 1:MCP
        DR = RV[M+1] - RV[M]
        VTSTAR = VA[M] / TANBC[M] + UTSTAR[M]
        VASTAR = VA[M] + UASTAR[M]
        VSTRSQ = VTSTAR^2 + VASTAR^2
        VSTAR = sqrt(VSTRSQ)
        if (CD_LD == 1) # Interpret CD as viscous drag coefficient, Cd
            DVISC = (VSTRSQ * CHORD[M] * CD[M]) / (2 * pi)
        else # Interpret CD as the lift/drag ratio L/D
            FKJ = VSTAR * G[M]
            DVISC = FKJ / CD[M]
        end
        CT = CT + (VTSTAR * G[M] - DVISC * VASTAR / VSTAR) * DR
        CQ = CQ + (VASTAR * G[M] + DVISC * VTSTAR / VSTAR) * RC[M] * DR
    end

    # ************************************************
    #     Add hub vortex drag if hub image is present
    # ************************************************
    if (IHUB != 0)
        CTH = 0.5 * (log(1.0 / RHV) + 3.0) * (NBLADE * G[1])^2
    else
        CTH = 0.0
    end

    CT = CT * 4 * NBLADE - CTH
    CQ = CQ * 4 * NBLADE
    CP = CQ * pi / ADVCO
    KT = CT * ADVCO^2 * pi / 8
    KQ = CQ * ADVCO^2 * pi / 16
    EFFY = CT * WAKE / CP

    return CT, CQ, CP, KT, KQ, EFFY, CTH
end
end
