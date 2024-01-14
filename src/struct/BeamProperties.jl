"""
Computes the sectional properties of composite beams
"""
module StructProp

# --- Public functions ---
export compute_section_property

struct section_property

	"""
	Inputs:
	c: chord length
	t: thickness
	ab: 
	ρₛ: density
	E₁: Young's modulus in-plane fiber longitudinal direction (x)
	E₂: Young's modulus in-plane fiber normal direction (y)
	G₁₂: In-plane Shear modulus 
	ν₁₂: Poisson ratio
	θ: global frame orientation
	"""
	c::Any
	t::Any
	ab::Any
	ρₛ::Any
	E₁::Any
	E₂::Any
	G₁₂::Any
	ν₁₂::Any
	θ::Any

end


function compute_section_property(section::section_property, constitutive)
	"""
	Orthotropic material uses classic laminate theory (CLT) for composite cross section property computation.

	Outputs
	-------
		EIₛ: scalar
			bending stiffness OOP [N - m²]
		EIₛIP: scalar
			in-plane (IP) bending stiffness of the element [N - m²]
		Kₛ: scalar
			bend-twist coupling [N - m²]
		GJₛ: scalar
			torsion stiffness [N - m²]
		Sₛ: scalar
			warping resistance [N - m⁴]
		EAₛ: scalar
            axial stiffness [N]
        Iₛ: scalar
            mass moment of inertia per unit length [kg-m²/m]
        mₛ: scalar
            mass per unit length [kg/m]

	NOTE:
	Axes convention is 1 along fiber, 2 transverse in-plane, 3 transverse out-of-plane

	This follows the formulation in 
	'Steady and dynamic hydroelastic behavior of composite lifting surfaces' 
	by Deniz Tolga Akcabaya & Yin Lu Young
	"""

	c = section.c
	t = section.t
	ab = section.ab
	ρₛ = section.ρₛ
	E₁ = section.E₁
	E₂ = section.E₂
	G₁₂ = section.G₁₂
	ν₁₂ = section.ν₁₂
	θ = section.θ

	# Compute nu_21 by E2 * nu12 = E1 * nu12
	ν₂₁ = (E₂ / E₁) * ν₁₂

	# Compute sectional properties output
	mₛ = ρₛ * c * t # [kg/m]
	Iₛ = ρₛ * (c * t^3 / 12 + c^3 * t / 12) # [kg-m^2/m]
	EIₛ = 0.0
	EIₛIP = 0.0
	EAₛ = 0.0
	Kₛ = 0.0
	GJₛ = 0.0
	if (constitutive == "orthotropic")
		# Fiber frame
		divPoissonsRatio = 1 / (1 - ν₁₂ * ν₂₁)
		q₁₁ = E₁ * divPoissonsRatio
		q₂₂ = E₂ * divPoissonsRatio
		q₁₂ = q₂₂ * ν₁₂
		q₆₆ = G₁₂

		# Convert to physical frame
		m = cos(θ)
		n = sin(θ)
		q₁₁ₚ = q₁₁ * m^4 + q₂₂ * n^4 + 2 * (q₁₂ + 2 * q₆₆) * m^2 * n^2
		q₂₂ₚ = q₁₁ * n^4 + q₂₂ * m^4 + 2 * (q₁₂ + 2 * q₆₆) * m^2 * n^2
		q₁₂ₚ = (q₁₁ + q₂₂ - 4 * q₆₆) * m^2 * n^2 + q₁₂ * (m^4 + n^4)
		q₁₆ₚ = m * n * (q₁₁ * m^2 - q₂₂ * n^2 - (q₁₂ + 2 * q₆₆) * (m^2 - n^2))
		q₂₆ₚ = m * n * (q₁₁ * n^2 - q₂₂ * m^2 + (q₁₂ + 2 * q₆₆) * (m^2 - n^2))
		q₆₆ₚ = (q₁₁ + q₂₂ - 2 * q₁₂) * m^2 * n^2 + q₆₆ * (m^2 - n^2)^2

		# Flexural stiffnesses D_ij for single layer laminate (M_i = D_ij k_i)
		d₁₁ = q₁₁ₚ / 12
		d₂₂ = q₂₂ₚ / 12
		d₁₂ = q₁₂ₚ / 12
		d₁₆ = q₁₆ₚ / 12
		d₂₆ = q₂₆ₚ / 12
		d₆₆ = q₆₆ₚ / 12

		# Weisshaar and Foist 1985 for a zero chordwise moment beam via composite plate theory (t^3 comes here for computational speedup)
		EIₛ = (d₁₁ - d₁₂^2 / d₂₂) * c * t^3
		Kₛ = 2 * (d₁₆ - d₁₂ * d₂₆ / d₂₂) * c * t^3
		GJₛ = 4 * (d₆₆ - d₂₆^2 / d₂₂) * c * t^3

		# TODO: make these parts more accurate later
		EIₛIP = E₁ * c^3 * t / 12 # in-plane EI for a rectangle
		EAₛ = E₁ * c * t # EA for a rectangle
		
	elseif (constitutive == "isotropic")
		EIₛ = E₁ * c * t^3 / 12 # EI for a rectangle
		EIₛIP = E₁ * c^3 * t / 12 # EI for a rectangle
		EAₛ = E₁ * c * t # EA for a rectangle
		GJₛ = G₁₂ * π * c^3 * t^3 / (c^2 + t^2) # GJ for an ellipse
		GJₛ = G₁₂ * c^3 * t * 0.333  # GJ for a rectangle
	end
	Sₛ = EIₛ * ((0.5 * ab)^2 + (c^2 / 12.0))

	# # Bullshit factor to remove spurious modes
	# EIₛIP *= 1e2
	# EAₛ *= 1e2

	# if (Kₛ < 1e-5)
	#     Kₛ = Kₛ + 1e-5
	# end

	# println("bend stiff: ", EIₛ)
	# println("BTC: ", Kₛ)
	# println("torsion stiff: ", GJₛ)
	# println("warp res: ", Sₛ)
	# println("ext. stiff: ", EAₛ)

	return EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ

end

end # end of module
