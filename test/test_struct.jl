# Unit test 
# Check Fig 4.1 from
# Deniz Tolga Akcabaya, Yin Lu Young "Steady and dynamic hydroelastic behavior of composite lifting surfaces" 

include("../src/Struct.jl")

using .StructProp
using Plots

# Inputs
c = 0.1
t = 0.012
ab = 0.0
ρₛ = 1590.0
E₁ = 117.8e9
E₂ = 13.4e9
G₁₂ = 3.9e9
ν₁₂ = 0.25
θ = pi / 6

N = 100
θₐ = range(-pi / 2, stop=pi / 2, length=N)
EIₛₐ = zeros(Float64, N)
Kₛₐ = zeros(Float64, N)
GJₛₐ = zeros(Float64, N)
Sₛₐ = zeros(Float64, N)

for i in 1:N
    θₗ = θₐ[i]
    section = StructProp.section_property(c, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, θₗ)
    EIₛ, Kₛ, GJₛ, Sₛ, Iₛ, mₛ = StructProp.compute_section_property(section)

    EIₛₐ[i] = EIₛ
    Kₛₐ[i] = Kₛ
    GJₛₐ[i] = GJₛ
    Sₛₐ[i] = Sₛ

end

plot(θₐ, EIₛₐ, show=true)
plot!(θₐ, Kₛₐ, show=true)
plot!(θₐ, GJₛₐ, show=true)
