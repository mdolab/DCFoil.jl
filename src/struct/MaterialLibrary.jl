# --- Julia 1.9---
"""
@File    :   MaterialLibrary.jl
@Time    :   2023/03/04
@Author  :   Galen Ng
@Desc    :   Based on material string name, return the material properties
"""

module MaterialLibrary

# --- Public functions ---
export return_constitutive

function return_constitutive(materialName::String)
    """
    Based on material string name, return the material properties
    SI units
    """
    ρₛ = 0.0
    E₁ = 0.0
    E₂ = 0.0
    G₁₂ = 0.0
    ν₁₂ = 0.0

    if (materialName == "cfrp") # carbon-fiber reinforced plastic UD
        ρₛ = 1590.0
        E₁ = 117.8e9
        E₂ = 13.4e9
        G₁₂ = 3.9e9
        ν₁₂ = 0.25
        constitutive = "orthotropic"

    elseif (materialName == "pmc")
        ρₛ = 1800.0
        E₁ = 39.3e9
        E₂ = 4.47e9
        G₁₂ = 1.3e9
        ν₁₂ = 0.25
        constitutive = "orthotropic"
    elseif (materialName == "gfrp")
        ρₛ = 1830.0
        E₁ = 13.09e9
        E₂ = 1.49e9
        G₁₂ = 0.43e9
        ν₁₂ = 0.25
        constitutive = "orthotropic"
    elseif (materialName == "test-comp")
        # TODO: will need to retrain these tests
        ρₛ = 1590.0
        E₁ = 100.0
        E₂ = 50.0
        G₁₂ = 10.0
        ν₁₂ = 0.25
        constitutive = "orthotropic"

    elseif (materialName == "ss") # stainless-steel
        ρₛ = 7900.0
        E₁ = 193e9
        E₂ = 193e9
        G₁₂ = 77.2e9
        ν₁₂ = 0.3
        constitutive = "isotropic"
    elseif (materialName == "al6061") # aluminum 6061
        ρₛ = 2700.0
        E₁ = 71e9
        E₂ = E₁
        ν₁₂ = 0.33
        G₁₂ = E₁ / (2 * (1 + ν₁₂))
        constitutive = "isotropic"
    elseif (materialName == "pvc") # polyvinyl chloride described in Ward et al. 2018
        ρₛ = 1300.0
        E₁ = 3.36e9
        E₂ = E₁
        ν₁₂ = 0.16
        G₁₂ = 1.45e9 # shear modulus reported in Ward et al. 2018 which agrees with the isotropic relation
        constitutive = "isotropic"
    elseif (materialName == "rigid") # unrealistic rigid material
        ρₛ = 7900.0
        E₁ = 193e12
        E₂ = 193e12
        G₁₂ = 77.2e12
        ν₁₂ = 0.3
        constitutive = "isotropic"
    elseif (materialName == "eirikurPl") # unrealistic rigid material
        ρₛ = 2800.0
        E₁ = 70e9
        E₂ = 70e9
        ν₁₂ = 0.3
        G₁₂ = E₁ / 2 / (1 + ν₁₂)
        constitutive = "isotropic"
    elseif (materialName == "test-iso")
        ρₛ = 1590.0
        E₁ = 1.0
        E₂ = 1.0
        G₁₂ = 1.0
        ν₁₂ = 0.25
        # constitutive = "isotropic"
        constitutive = "orthotropic" # NOTE: Need to use this because the isotropic case uses an ellipse for GJ
    elseif (materialName == "test-iso3d")
        ρₛ = 1000.0
        E₁ = 1e9
        E₂ = 1e9
        G₁₂ = 1e9
        ν₁₂ = 0.25
        constitutive = "isotropic"
    elseif (materialName == "ud-wov-ud") # unidirectional-woven-unidirectional
        ρₛ = 1570.0
        E₁ = 98.0e9
        E₂ = 25.2e9
        G₁₂ = 4.2e9
        ν₁₂ = 0.20
        constitutive = "orthotropic"
    elseif (materialName == "wov-ud-wov") # woven-unidirectional-woven
        ρₛ = 1560.0
        E₁ = 77.1e9
        E₂ = 39.7e9
        G₁₂ = 4.6e9
        ν₁₂ = 0.15
        constitutive = "orthotropic"
    elseif (materialName == "IM6-epoxy")
        ρₛ = 1590.0
        E₁ = 203.1e9
        E₂ = 11.2e9
        G₁₂ = 8.4e9
        ν₁₂ = 0.32
        constitutive = "orthotropic"
    elseif (materialName == "new composite material ")
        println("not done yet")
    end # if

    return ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive
end # function return_constitutive

end # module MaterialLibrary