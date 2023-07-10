# --- Julia 1.7---
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
    """
    if (materialName == "cfrp") # carbon-fiber reinforced plastic
        ρₛ = 1590.0
        E₁ = 117.8e9
        E₂ = 13.4e9
        G₁₂ = 3.9e9
        ν₁₂ = 0.25
        constitutive = "orthotropic"
    elseif (materialName == "test-comp")
        ρₛ = 1590.0
        E₁ = 1
        E₂ = 1
        G₁₂ = 1
        ν₁₂ = 0.25
        constitutive = "orthotropic"
    elseif (materialName == "ss") # stainless-steel
        ρₛ = 7900
        E₁ = 193e9
        E₂ = 193e9
        G₁₂ = 77.2e9
        ν₁₂ = 0.3
        constitutive = "isotropic"
    elseif (materialName == "rigid") # unrealistic rigid material
        ρₛ = 7900
        E₁ = 193e12
        E₂ = 193e12
        G₁₂ = 77.2e12
        ν₁₂ = 0.3
        constitutive = "isotropic"
    elseif (materialName == "eirikurPl") # unrealistic rigid material
        ρₛ = 2800
        E₁ = 70e9
        E₂ = 70e9
        ν₁₂ = 0.3
        G₁₂ = E₁ / 2 / (1 + ν₁₂)
        constitutive = "isotropic"
    elseif (materialName == "test-iso")
        ρₛ = 1590.0
        E₁ = 1
        E₂ = 1
        G₁₂ = 1
        ν₁₂ = 0.25
        # constitutive = "isotropic"
        constitutive = "orthotropic" # NOTE: Need to use this because the isotropic case uses an ellipse for GJ
    elseif (materialName == "test-iso3d")
        ρₛ = 10.0
        E₁ = 1
        E₂ = 1
        G₁₂ = 1
        ν₁₂ = 0.25
        constitutive = "isotropic"
    end # if

    return ρₛ, E₁, E₂, G₁₂, ν₁₂, constitutive
end # function return_constitutive

end # module MaterialLibrary