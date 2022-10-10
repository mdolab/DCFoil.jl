# --- Julia---
"""
@File    :   SolveDynamic.jl
@Time    :   2022/10/07
@Author  :   Galen Ng
@Desc    :   Similar to SolveSteady.jl but now it is a second order dynamical system!
"""

module SolveDynamic
    """
    Frequency domain hydroelastic solver
    """

    # --- Public functions ---
    export solve

    # --- Libraries ---
    using FLOWMath: linear, akima
    using LinearAlgebra, Statistics
    using JSON
    using Zygote

    # --- DCFoil modules ---
    # First include them
    include("InitModel.jl")
    include("Struct.jl")
    include("struct/FiniteElements.jl")
    include("Hydro.jl")
    # then use them
    using .InitModel, .Hydro, .StructProp
    using .FEMMethods

    function solve(DVDict, outputDir::String)
        """
        Solve (-ω²[M]-jω[C]+[K]){ũ} = {f̃}
        """
        # ---------------------------
        #   Initialize
        # ---------------------------
        global FOIL = InitModel.init_dynamic(fSweep, DVDict)
        nElem = FOIL.neval - 1
        constitutive = FOIL.constitutive

        # ************************************************
        #     Assemble structural matrices
        # ************************************************
        elemType = "BT2"
        
        structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)
        globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, constitutive)

        # --- Initialize states ---
        u = copy(globalF)

        # ************************************************
        #     Assemble hydrodynamic matrices
        # ************************************************
    end

    


end # end module