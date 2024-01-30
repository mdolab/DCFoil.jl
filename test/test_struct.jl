# Unit test
# Check Fig 4.1 from
# Deniz Tolga Akcabaya, Yin Lu Young "Steady and dynamic hydroelastic behavior of composite lifting surfaces"

include("../src/struct/BeamProperties.jl")

using LinearAlgebra
using .BeamProperties
using Plots, DelimitedFiles

# ==============================================================================
#                         BEAM PROPERTIES
# ==============================================================================
function test_struct()
	"""
	Test the constitutive relations (the beam stiffnesses)
	"""
	# ************************************************
	#     Setup the test problem
	# ************************************************
	c = 0.1
	t = 0.012
	ab = 0.0
	ρₛ = 1590.0
	E₁ = 117.8e9
	E₂ = 13.4e9
	G₁₂ = 3.9e9
	ν₁₂ = 0.25
	# θ = pi / 6

	N = 100
	θₐ = range(-pi / 2, stop = pi / 2, length = N)
	EIₛₐ = zeros(Float64, N)
	EIIPₛₐ = zeros(Float64, N)
	EAₛₐ = zeros(Float64, N)
	Kₛₐ = zeros(Float64, N)
	GJₛₐ = zeros(Float64, N)
	Sₛₐ = zeros(Float64, N)

	for i in 1:N
		θₗ = θₐ[i]
		section = BeamProperties.section_property(c, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, θₗ)
		EIₛ, EIIP, Kₛ, GJₛ, Sₛ, EAₛ, _, _ = BeamProperties.compute_section_property(section, "orthotropic")

		EIₛₐ[i] = EIₛ
		EIIPₛₐ[i] = EIIP
		EAₛₐ[i] = EAₛ
		Kₛₐ[i] = Kₛ
		GJₛₐ[i] = GJₛ
		Sₛₐ[i] = Sₛ

	end

	# # --- Uncomment to visualize ---
	# plot(θₐ, EIₛₐ, show=true)
	# plot!(θₐ, Kₛₐ, show=true)
	# plot!(θₐ, GJₛₐ, show=true)

	# Reference values
	EIₛref_sol = vec(
		[192.960000000000 192.799684814088 192.319904623194 191.524225463090 190.418827023785 189.012868357573 187.319022878197 185.354206788453 183.140533585080 180.706537000260 178.088715867676 175.333467081596 172.499487014635 169.660737128840 166.910085172596 164.363747523630 162.166667693359 160.498965275934 159.583569834954 159.695101569992 161.169954487455 164.417348880274 169.930808957395 178.299040633559 190.214484517500 206.476867702419 227.987896931683 255.731964562205 290.736724144565 334.007281154564 386.429510878274 448.642792354674 520.890993825552 602.872291705434 693.620382683019 791.456203019547 894.043772897086 998.562699791943 1101.97720110910 1201.34955371324 1294.12966841576 1378.36042743800 1452.76647625616 1516.72862672858 1570.17249156711 1613.41105476725 1646.97782940337 1671.47629680222 1687.45900311087 1695.34014992862 1695.34014992862 1687.45900311087 1671.47629680222 1646.97782940337 1613.41105476725 1570.17249156711 1516.72862672858 1452.76647625616 1378.36042743800 1294.12966841576 1201.34955371324 1101.97720110910 998.562699791943 894.043772897086 791.456203019547 693.620382683019 602.872291705434 520.890993825552 448.642792354674 386.429510878274 334.007281154564 290.736724144565 255.731964562205 227.987896931683 206.476867702419 190.214484517500 178.299040633559 169.930808957395 164.417348880274 161.169954487455 159.695101569992 159.583569834954 160.498965275934 162.166667693359 164.363747523630 166.910085172596 169.660737128840 172.499487014635 175.333467081596 178.088715867676 180.706537000260 183.140533585080 185.354206788453 187.319022878197 189.012868357573 190.418827023785 191.524225463090 192.319904623194 192.799684814088 192.960000000000],
	)
	GJₛref_sol = vec(
		[224.640000000000 225.362114363454 227.532631236126 231.164084143938 236.277400595295 242.901951254463 251.075601384646 260.844745107605 272.264294355498 285.397583062970 300.316132201380 317.099201494559 335.833027599201 356.609614403475 379.524896837641 404.676042988744 432.157588309451 462.056009038752 494.442240317506 529.361532647711 566.819930554533 606.766574606235 649.071017837568 693.494886017419 739.657614999496 786.996831757905 834.725410995890 881.789532574745 926.835267697813 968.195099432228 1003.90949806182 1031.80050764042 1049.61166399931 1055.21868848129 1046.89707577916 1023.60868726266 985.248034071342 932.781998094614 868.233555594841 794.499463870064 715.039501536755 633.509922310265 553.421570969655 477.883554574417 409.459176675770 350.128038048921 301.327242008653 264.037599832452 238.883863054271 226.225874980410 226.225874980410 238.883863054271 264.037599832452 301.327242008653 350.128038048921 409.459176675770 477.883554574417 553.421570969655 633.509922310265 715.039501536755 794.499463870064 868.233555594841 932.781998094614 985.248034071342 1023.60868726266 1046.89707577916 1055.21868848129 1049.61166399931 1031.80050764042 1003.90949806182 968.195099432228 926.835267697813 881.789532574745 834.725410995890 786.996831757905 739.657614999496 693.494886017419 649.071017837568 606.766574606235 566.819930554533 529.361532647711 494.442240317506 462.056009038752 432.157588309451 404.676042988744 379.524896837641 356.609614403475 335.833027599201 317.099201494559 300.316132201380 285.397583062970 272.264294355498 260.844745107605 251.075601384646 242.901951254463 236.277400595295 231.164084143938 227.532631236126 225.362114363454 224.640000000000],
	)
	Kₛref_sol = vec(
		[1.00000000094844e-05 4.90461396538536 9.74536625367033 14.4571979597781 18.9726314301506 23.2205000404101 27.1246047018449 30.6022724836161 33.5627930821587 35.9057100305332 37.5189461691877 38.2767479639919 38.0374421766705 36.6410132181870 33.9065331689898 29.6295130220845 23.5792988013365 15.4967172939570 5.09229276333805 -7.95448033349170 -23.9941004713133 -43.4032565544733 -66.5768914945866 -93.9143379810164 -125.797298874114 -162.557116431449 -204.428762703819 -251.489628268612 -303.582961058275 -360.229239745193 -420.534208333200 -483.109477483209 -546.029007573466 -606.849235365196 -662.717569942935 -710.579509527916 -747.468685172584 -770.833587066186 -778.832668480705 -770.529023355333 -745.940999595771 -705.946525389821 -652.078452057551 -586.270055712359 -510.609093758511 -427.141685803409 -337.744776673831 -244.067063037643 -147.527005842863 -49.3525111745736 49.3525111745736 147.527005842863 244.067063037643 337.744776673831 427.141685803409 510.609093758511 586.270055712359 652.078452057551 705.946525389821 745.940999595771 770.529023355333 778.832668480705 770.833587066186 747.468685172584 710.579509527916 662.717569942935 606.849235365196 546.029007573466 483.109477483209 420.534208333200 360.229239745193 303.582961058275 251.489628268612 204.428762703819 162.557116431449 125.797298874114 93.9143379810164 66.5768914945866 43.4032565544733 23.9941004713133 7.95448033349170 -5.09229276333805 -15.4967172939570 -23.5792988013365 -29.6295130220845 -33.9065331689898 -36.6410132181870 -38.0374421766705 -38.2767479639919 -37.5189461691877 -35.9057100305332 -33.5627930821587 -30.6022724836161 -27.1246047018449 -23.2205000404101 -18.9726314301506 -14.4571979597781 -9.74536625367033 -4.90461396538536 9.99999999051562e-06],
	)
	Sₛref_sol = vec(
		[0.160800000000000 0.160666404011740 0.160266587185995 0.159603521219242 0.158682355853154 0.157510723631311 0.156099185731830 0.154461838990377 0.152617111320900 0.150588780833550 0.148407263223063 0.146111222567996 0.143749572512196 0.141383947607367 0.139091737643830 0.136969789603025 0.135138889744466 0.133749137729945 0.132986308195795 0.133079251308326 0.134308295406213 0.137014457400228 0.141609007464496 0.148582533861299 0.158512070431250 0.172064056418683 0.189989914109736 0.213109970468504 0.242280603453804 0.278339400962137 0.322024592398562 0.373868993628895 0.434075828187960 0.502393576421195 0.578016985569182 0.659546835849622 0.745036477414238 0.832135583159952 0.918314334257584 1.00112462809437 1.07844139034647 1.14863368953167 1.21063873021346 1.26394052227382 1.30847707630593 1.34450921230604 1.37248152450281 1.39289691400185 1.40621583592572 1.41278345827385 1.41278345827385 1.40621583592572 1.39289691400185 1.37248152450281 1.34450921230604 1.30847707630593 1.26394052227382 1.21063873021346 1.14863368953167 1.07844139034647 1.00112462809437 0.918314334257584 0.832135583159952 0.745036477414238 0.659546835849622 0.578016985569182 0.502393576421195 0.434075828187960 0.373868993628895 0.322024592398562 0.278339400962137 0.242280603453804 0.213109970468504 0.189989914109736 0.172064056418683 0.158512070431250 0.148582533861299 0.141609007464496 0.137014457400228 0.134308295406213 0.133079251308326 0.132986308195795 0.133749137729945 0.135138889744466 0.136969789603025 0.139091737643830 0.141383947607367 0.143749572512196 0.146111222567996 0.148407263223063 0.150588780833550 0.152617111320900 0.154461838990377 0.156099185731830 0.157510723631311 0.158682355853154 0.159603521219242 0.160266587185995 0.160666404011740 0.160800000000000],
	)

	# Relative error
	rel_err1 = LinearAlgebra.norm(EIₛₐ - EIₛref_sol, 2) / LinearAlgebra.norm(EIₛref_sol, 2)
	rel_err2 = LinearAlgebra.norm(GJₛₐ - GJₛref_sol, 2) / LinearAlgebra.norm(GJₛref_sol, 2)
	rel_err3 = LinearAlgebra.norm(Kₛₐ - Kₛref_sol, 2) / LinearAlgebra.norm(Kₛref_sol, 2)
	rel_err4 = LinearAlgebra.norm(Sₛₐ - Sₛref_sol, 2) / LinearAlgebra.norm(Sₛref_sol, 2)
	rel_err = max(rel_err1, rel_err2, rel_err3, rel_err4) # just call it the max of all of them

	return rel_err
end

# ==============================================================================
#                         FINITE ELEMENT
# ==============================================================================
"""
unit tests to verify the beam bend and bend-twist element
"""

include("../src/InitModel.jl")
include("../src/struct/FEMMethods.jl")
include("../src/solvers/SolverRoutines.jl")
using .FEMMethods, .InitModel
using .LinearBeamElem, .SolverRoutines

# ==============================================================================
#                         Test elemental matrices
# ==============================================================================
function test_BT2_stiff()
	"""
	Test the second order beam matrix with unit values
	"""
	constitutive = "orthotropic"
	# ************************************************
	#     BT2 element stiff
	# ************************************************
	elemType = "BT2"

	Ktest = LinearBeamElem.compute_elem_stiff(2, 0.0, 8, 4, 8 / 3, 0.0, 2, 1, elemType, constitutive)
	# show(stdout, "text/plain", Ktest[1:end, 1:end])

	# --- Reference value ---
	# These were obtained from the matlab symbolic script plugging 1 for flexural stiffnesses and 2 for the chord
	ref_sol = vec(
		[
			3.0000 3.0000 -3.0000 -5.0000 -3.0000 3.0000 3.0000 -1.0000
			3.0000 4.0000 -1.0000 -6.0000 -3.0000 2.0000 1.0000 0
			-3.0000 -1.0000 8.8000 4.8000 3.0000 -5.0000 -8.8000 4.8000
			-5.0000 -6.0000 4.8000 11.4667 5.0000 -4.0000 -4.8000 2.1333
			-3.0000 -3.0000 3.0000 5.0000 3.0000 -3.0000 -3.0000 1.0000
			3.0000 2.0000 -5.0000 -4.0000 -3.0000 4.0000 5.0000 -2.0000
			3.0000 1.0000 -8.8000 -4.8000 -3.0000 5.0000 8.8000 -4.8000
			-1.0000 0 4.8000 2.1333 1.0000 -2.0000 -4.8000 3.4667
		],
	)

	# # --- Relative error ---
	answers = vec(Ktest) # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	if det(Ktest) >= 1e-16
		print("Your stiffness matrix is not singular...it's wrong")
		rel_err += 1 # make test fail
	end

	if Ktest' != Ktest
		print("Your stiffness matrix is not symmetric...it's wrong")
		rel_err += 1 # make test fail
	end

	return rel_err
end

# test_BT2_stiff()

function test_BT2_mass()
	"""
	Test the second order beam matrix with unit values
	"""
	# ************************************************
	#     BT2 element mass
	# ************************************************
	elemType = "BT2"
	# INPUTS
	ms = 4
	isea = 16 / 3
	le = 2
	xalphab = -1
	# Test
	Mtest = LinearBeamElem.compute_elem_mass(ms, isea, le, xalphab, elemType)
	# show(stdout, "text/plain", Mtest[5:end, 5:end])

	totalMass = ms * le
	totalInertia = isea * le

	# --- Sum values over translational DOFs ---
	massSum = sum(Mtest[1, 1]) + sum(Mtest[1, 5]) + sum(Mtest[5, 1]) + sum(Mtest[5, 5])
	momentSum = sum(Mtest[3, 3]) + sum(Mtest[3, 7]) + sum(Mtest[7, 3]) + sum(Mtest[7, 7])

	# --- Reference value ---
	# These were obtained from the matlab symbolic script plugging 2 for rho, 2 for the chord, and 1 for everything else
	ref_sol = vec(
		[
			2.9714 0.8381 -2.9714 -0.8381 1.0286 -0.4952 -1.0286 0.4952
			0.8381 0.3048 -0.8381 -0.3048 0.4952 -0.2286 -0.4952 0.2286
			-2.9714 -0.8381 3.9619 1.1175 -1.0286 0.4952 1.3714 -0.6603
			-0.8381 -0.3048 1.1175 0.4063 -0.4952 0.2286 0.6603 -0.3048
			1.0286 0.4952 -1.0286 -0.4952 2.9714 -0.8381 -2.9714 0.8381
			-0.4952 -0.2286 0.4952 0.2286 -0.8381 0.3048 0.8381 -0.3048
			-1.0286 -0.4952 1.3714 0.6603 -2.9714 0.8381 3.9619 -1.1175
			0.4952 0.2286 -0.6603 -0.3048 0.8381 -0.3048 -1.1175 0.4063
		],
	)

	# # --- Relative error ---
	answers = vec(Mtest) # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	if minimum(eigvals(Mtest)) < 0.0
		print("Your mass matrix is not positive definite...it's wrong")
		rel_err += 1 # make test fail
	end

	return rel_err
end

function test_COMP2_stiff()
	"""
	Test the second order beam matrix with unit values
	"""
	constitutive = "orthotropic"
	# ************************************************
	#     COMP2 element stiff
	# ************************************************
	elemType = "COMP2"

	Ktest = LinearBeamElem.compute_elem_stiff(2, 0.0, 8, 4, 8 / 3, 0.0, 2, 1, elemType, constitutive)
	# show(stdout, "text/plain", Ktest[1:end, 1:end])

	# # Reduce the full size to what we actually test
	# blankingList = vcat([1,2,6,9], [1,2,6,9] .+ 9)
	# Ktest = Ktest[1:end.∉[blankingList], 1:end.∉[blankingList]]

	# Rearrange stuff so we can compare to the matlab training solution
	switch = [3, 5, 8, 4, 7]
	switchid = vcat(switch, switch .+ 9)
	Ktest = hcat(
		Ktest[3, switchid],
		Ktest[5, switchid],
		Ktest[8, switchid],
		Ktest[4, switchid],
		Ktest[7, switchid],
		Ktest[12, switchid],
		Ktest[14, switchid],
		Ktest[17, switchid],
		Ktest[13, switchid],
		Ktest[16, switchid],
	)
	println("Ktest")
	show(stdout, "text/plain", Ktest)
	println()

	# --- Reference value ---
	# These were obtained from the matlab symbolic script plugging 1 for flexural stiffnesses and 2 for the chord
	ref_sol = vec(
		[
			4.2857    4.2857    0.4286   -3.0000   -5.0000   -4.2857    4.2857   -0.4286    3.0000   -1.0000
			4.2857    5.4857    0.6286   -0.6000   -5.6000   -4.2857    3.0857   -0.2286    0.6000    0.4000
			0.4286    0.6286    0.3429    0.4000    0.4000   -0.4286    0.2286    0.0571   -0.4000    0.4000
			-3.0000   -0.6000    0.4000    8.8000    4.8000    3.0000   -5.4000    0.4000   -8.8000    4.8000
			-5.0000   -5.6000    0.4000    4.8000   11.4667    5.0000   -4.4000    0.4000   -4.8000    2.1333
			-4.2857   -4.2857   -0.4286    3.0000    5.0000    4.2857   -4.2857    0.4286   -3.0000    1.0000
			4.2857    3.0857    0.2286   -5.4000   -4.4000   -4.2857    5.4857   -0.6286    5.4000   -2.4000
			-0.4286   -0.2286    0.0571    0.4000    0.4000    0.4286   -0.6286    0.3429   -0.4000    0.4000
			3.0000    0.6000   -0.4000   -8.8000   -4.8000   -3.0000    5.4000   -0.4000    8.8000   -4.8000
			-1.0000    0.4000    0.4000    4.8000    2.1333    1.0000   -2.4000    0.4000   -4.8000    3.4667
		],
	)

	# --- Relative error ---
	answers = vec(Ktest) # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	if det(Ktest) >= 1e-16
		println("Your stiffness matrix is not singular...it's wrong")
		rel_err += 1 # make test fail
	end

	if Ktest' != Ktest
		println("Your stiffness matrix is not symmetric...it's wrong")
		rel_err += 1 # make test fail
	end

	return rel_err
end

function test_COMP2_mass()
	"""
	Test the second order beam matrix with unit values

	To debug this, you'll need my overleaf document with the matrix derivations
	"""
	# ************************************************
	#     COMP2 element mass
	# ************************************************
	elemType = "COMP2"
	# INPUTS
	ms = 4
	isea = 16 / 3
	le = 2
	xalphab = -1.0 # you can't just change xalpha without changing isea btw
	# Test
	Mtest = LinearBeamElem.compute_elem_mass(ms, isea, le, xalphab, elemType)
	# show(stdout, "text/plain", Mtest[5:end, 5:end])

	totalMass = ms * le
	totalInertia = isea * le

	# # Reduce the full size to what we actually test
	# blankingList = vcat([1,2,6,9], [1,2,6,9] .+ 9)
	# Mtest = Mtest[1:end.∉[blankingList], 1:end.∉[blankingList]]

	# Now rearrange stuff so we can compare to the matlab training solution
	# Rearrange stuff so we can compare to the matlab training solution
	switch = [3, 5, 8, 4, 7]
	switchid = vcat(switch, switch .+ 9)
	Mtest = hcat(
		Mtest[3, switchid],
		Mtest[5, switchid],
		Mtest[8, switchid],
		Mtest[4, switchid],
		Mtest[7, switchid],
		Mtest[12, switchid],
		Mtest[14, switchid],
		Mtest[17, switchid],
		Mtest[13, switchid],
		Mtest[16, switchid],
	)
	println("Mtest")
	show(stdout, "text/plain", Mtest)
	println()

	# --- Sum over translational and rotational DOFs ---
	massSum = sum(Mtest[1, 1]) + sum(Mtest[1, 6]) + sum(Mtest[6, 1]) + sum(Mtest[6, 6])
	momentSum = sum(Mtest[4, 4]) + sum(Mtest[4, 9]) + sum(Mtest[9, 4]) + sum(Mtest[9, 9])


	# --- Reference value ---
	# These were obtained from the matlab symbolic script plugging 2 for rho, 2 for the chord, and 1 for everything else
	ref_sol = vec(
		[
			 3.1342    1.0771    0.1622   -3.0476   -0.8571    0.8658   -0.5229    0.1045   -0.9524    0.4762
			 1.0771    0.4802    0.0797   -1.0476   -0.3810    0.5229   -0.3071    0.0600   -0.5524    0.2667
			 0.1622    0.0797    0.0139   -0.1587   -0.0635    0.1045   -0.0600    0.0115   -0.1079    0.0508
			-3.0476   -1.0476   -0.1587    3.9619    1.1175   -0.9524    0.5524   -0.1079    1.3714   -0.6603
			-0.8571   -0.3810   -0.0635    1.1175    0.4063   -0.4762    0.2667   -0.0508    0.6603   -0.3048
			 0.8658    0.5229    0.1045   -0.9524   -0.4762    3.1342   -1.0771    0.1622   -3.0476    0.8571
			-0.5229   -0.3071   -0.0600    0.5524    0.2667   -1.0771    0.4802   -0.0797    1.0476   -0.3810
			 0.1045    0.0600    0.0115   -0.1079   -0.0508    0.1622   -0.0797    0.0139   -0.1587    0.0635
			-0.9524   -0.5524   -0.1079    1.3714    0.6603   -3.0476    1.0476   -0.1587    3.9619   -1.1175
			 0.4762    0.2667    0.0508   -0.6603   -0.3048    0.8571   -0.3810    0.0635   -1.1175    0.4063
		],
	)

	# --- Relative error ---
	answers = vec(Mtest) # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	if Mtest' != Mtest
		println("Your mass matrix is not symmetric...it's wrong")
		rel_err += 1 # make test fail
	end

	if minimum(eigvals(Mtest)) < 0.0
		print("Your mass matrix is not positive definite...it's wrong")
		rel_err += 1 # make test fail
	end

	return rel_err
end

# ans = test_BT3_mass()
# ans = test_BT3_stiff()
# ans1 = test_COMP2_stiff()
# ans2 = test_COMP2_mass()
# ==============================================================================
#                         Test finite element solver with unit loads
# ==============================================================================
function test_FiniteElementIso()
	"""
	Test the finite elements with unit loads, thickness, length, and structural moduli
	"""
	nNodes = 30
	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => 30.0 * π / 180, # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 1.0, # semispan [m]
		"ab" => zeros(nNodes), # dist from midchord to EA [m]
		"toc" => 1, # thickness-to-chord ratio
		"x_αb" => zeros(nNodes), # static imbalance [m]
		"θ" => 0 * π / 180, # fiber angle global [rad]
	)
	solverOptions = Dict(
		"nNodes" => nNodes,
		"material" => "test-iso", # preselect from material library
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
	)
	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	constitutive = "orthotropic" # NOTE: using this because the isotropic code uses an ellipse for computing GJ
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])
	# ************************************************
	#     bend element
	# ************************************************
	elemType = "bend"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType)

	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-1] = 1.0 # 1 Newton tip force NOTE: FIX LATER bend
	u = copy(globalF)

	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q1 = FEMMethods.solve_structure(K, M, F)

	# ************************************************
	#     bend-twist
	# ************************************************
	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "bend-twist"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-2] = 1.0 # 0 Newton tip force
	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q2 = FEMMethods.solve_structure(K, M, F)

	# ---------------------------
	#   Tip torque only
	# ---------------------------
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end] = 1.0 # 0 Newton tip force
	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q3 = FEMMethods.solve_structure(K, M, F)

	# --- Reference value ---
	# the tip deformations should be 4m for pure bending with tip force and 3 radians for tip torque
	ref_sol = [4, 4, 3]

	# --- Relative error ---
	answers = [q1[end-1], q2[end-2], q3[end]] # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	return rel_err
end

function test_FiniteElementComp()
	"""
	Test the finite elements with unit loads, thickness, length, and structural moduli
	"""
	nNodes = 40
	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => 0.0 * π / 180, # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 1.0, # semispan [m]
		"ab" => zeros(nNodes), # dist from midchord to EA [m]
		"toc" => 1, # thickness-to-chord ratio
		"x_αb" => zeros(nNodes), # static imbalance [m]
		"θ" => deg2rad(15), # fiber angle global [rad]
	)
	solverOptions = Dict(
		"material" => "test-comp", # preselect from material library
		"nNodes" => nNodes,
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
	)
	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	constitutive = FOIL.constitutive
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])


	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "BT2"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-3] = 1.0 # 0 Newton tip force
	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)
	writedlm("DebugK-BT2.csv", K, ",")
	writedlm("DebugF-BT2.csv", F, ",")

	q3 = FEMMethods.solve_structure(K, M, F)
	BT2_Ftip_wtip = q3[end-3]
	BT2_Ftip_psitip = q3[end-1]

	# ---------------------------
	#   Tip torque only
	# ---------------------------
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-1] = 1.0 # 0 Newton tip force
	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q4 = FEMMethods.solve_structure(K, M, F)
	BT2_Ttip_wtip = q4[end-3]
	BT2_Ttip_psitip = q4[end-1]

	# --- Print these out if something does not make sense ---
	println("BT2_Ftip_wtip = ", BT2_Ftip_wtip, " [m]")
	println("BT2_Ftip_psitip = ", rad2deg(BT2_Ftip_psitip), " [deg]")
	println("BT2_Ttip_wtip = ", BT2_Ttip_wtip, " [m]")
	println("BT2_Ttip_psitip = ", rad2deg(BT2_Ttip_psitip), " [deg]")

	# --- Reference value ---
	# the tip deformations should be 4m for pure bending with tip force and 3 radians for tip torque
	# Of course, the tip torque for BT2 will be smaller since we prescribe the zero twist derivative BC at the root
	ref_sol = [4.0, 0.0, 0.0, 2.56699]

	# --- Relative error ---
	answers = [BT2_Ftip_wtip, BT2_Ftip_psitip, BT2_Ttip_wtip, BT2_Ttip_psitip] # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	omegaSquared, modeShapes = SolverRoutines.compute_eigsolve(K, M, 3)

	println("f_n = ", sqrt.(omegaSquared) / (2 * pi))

	return rel_err
end

# test_FiniteElementComp()

using DelimitedFiles, LinearAlgebra

function test_FiniteElementIso3D()
	"""
	Test the finite elements with unit loads, thickness, length, and structural moduli
	"""
	nNodes = 15
	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => deg2rad(0.0), # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 10, # semispan [m]
		"ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
		"toc" => 1.0, # thickness-to-chord ratio
		"x_αb" => 0 * ones(nNodes), # static imbalance [m]
		"θ" => deg2rad(0.0), # fiber angle global [rad]
		"s_strut" => 0.4, # from Yingqian
	)
	solverOptions = Dict(
		# ---------------------------
		#   I/O
		# ---------------------------
		"name" => "t-foil",
		"debug" => false,
		# ---------------------------
		#   General appendage options
		# ---------------------------
		"config" => "wing",
		# "config" => "t-foil",
		"nNodes" => nNodes, # number of nodes on foil half wing
		"nNodeStrut" => 10, # nodes on strut
		"rotation" => 0.0, # deg
		"gravityVector" => [0.0, 0.0, -9.81],
		"use_tipMass" => false,
		# ---------------------------
		#   Flow
		# ---------------------------
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
		"use_freeSurface" => false,
		"use_cavitation" => false,
		"use_ventilation" => false,
		# ---------------------------
		#   Structure
		# ---------------------------
		"material" => "test-iso3d", # preselect from material library
		# ---------------------------
		#   Solver modes
		# ---------------------------
		# --- Static solve ---
		"run_static" => true,
		# --- Forced solve ---
		"run_forced" => false,
		"fSweep" => 1:2,
		"tipForceMag" => 1,
		# --- p-k (Eigen) solve ---
		"run_modal" => false,
		"run_flutter" => false,
		"nModes" => 1,
		"uRange" => [1, 2],
		"maxQIter" => 100, # that didn't fix the slow run time...
		"rhoKS" => 80.0,
	)
	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])

	# ************************************************
	#     BT2
	# ************************************************
	dim = 3
	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "BEAM3D"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped", dim)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive, dim)
	T = [ # transform -90 deg about z
		0 1 0
		-1 0 0
		0 0 1
	]
	transMat = [
		T zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) T zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) T zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) T
	]
	FEMMethods.apply_tip_load!(globalF, elemType, transMat, "force")
	FEMMethods.apply_tip_load!(globalF, elemType, transMat, "torque")
	u = copy(globalF)

	# println("Global matrices")
	# println("globalK = ")
	# show(stdout, "text/plain", globalK)
	# println()
	# println("globalM = ")
	# show(stdout, "text/plain", globalM)
	# println()
	# println("globalF = ")
	# println(globalF)

	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	# println("Blanked matrices")
	# println("K = ")
	# show(stdout, "text/plain", K)
	# println()
	# # println("globalM = ")
	# # show(stdout, "text/plain", globalM)
	# # println()
	# println("F = ")
	# println(F)
	# # return K, F
	q1 = FEMMethods.solve_structure(K, M, F)

	# writedlm("DebugKGlobMatrix.csv", globalK, ',')
	# writedlm("DebugMGlobMatrix.csv", globalM, ',')

	# ************************************************
	#     Natural frequencies
	# ************************************************
	omegaSquared, modeShapes = SolverRoutines.compute_eigsolve(K, M, nNodes)

	return q1, K, M, F, omegaSquared, modeShapes

end


# q1, K, M, F, omegaSquared, modeShapes = test_FiniteElementIso3D()
# println("tip deflection = ", q1[end-3], " m")
# println("tip rotation about y axis = ", q1[end-1], " rad")
# natFreqs = sqrt.(Complex.(omegaSquared)) / (2 * pi)
# println("Natural frequencies = ", real.(natFreqs), " Hz")

function test_FiniteElementBend()
	"""
	Test the finite elements with unit loads, thickness, length, and structural moduli
	"""
	nNodes = 3
	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => deg2rad(0.0), # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1.0 * ones(nNodes), # chord length [m]
		"s" => 10, # semispan [m]
		"ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
		"toc" => 1.0, # thickness-to-chord ratio
		"x_αb" => 0 * ones(nNodes), # static imbalance [m]
		"θ" => deg2rad(0.0), # fiber angle global [rad]
		"s_strut" => 0.4, # from Yingqian
	)
	solverOptions = Dict(
		# ---------------------------
		#   I/O
		# ---------------------------
		"name" => "t-foil",
		"debug" => false,
		# ---------------------------
		#   General appendage options
		# ---------------------------
		"config" => "wing",
		# "config" => "t-foil",
		"nNodes" => nNodes, # number of nodes on foil half wing
		"nNodeStrut" => 10, # nodes on strut
		"rotation" => 0.0, # deg
		"gravityVector" => [0.0, 0.0, -9.81],
		"use_tipMass" => false,
		# ---------------------------
		#   Flow
		# ---------------------------
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
		"use_freeSurface" => false,
		"use_cavitation" => false,
		"use_ventilation" => false,
		# ---------------------------
		#   Structure
		# ---------------------------
		"material" => "test-iso3d", # preselect from material library
		# ---------------------------
		#   Solver modes
		# ---------------------------
		# --- Static solve ---
		"run_static" => true,
		# --- Forced solve ---
		"run_forced" => false,
		"fSweep" => 1:2,
		"tipForceMag" => 1,
		# --- p-k (Eigen) solve ---
		"run_modal" => false,
		"run_flutter" => false,
		"nModes" => 1,
		"uRange" => [1, 2],
		"maxQIter" => 100, # that didn't fix the slow run time...
		"rhoKS" => 80.0,
	)
	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])

	# ************************************************
	#     BT2
	# ************************************************
	dim = 3
	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "bend"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType, "clamped", dim)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive, dim)
	T = [ # transform -90 deg about z
		0 1 0
		-1 0 0
		0 0 1
	]
	transMat = [
		T[3, 1] T[3, 2] T[3, 3] zeros(1, 3) zeros(1, 3) zeros(1, 3)
		zeros(1, 3) T[2, 1] T[2, 2] T[2, 3] zeros(1, 3) zeros(1, 3)
		zeros(1, 3) zeros(1, 3) T[3, 1] T[3, 2] T[3, 3] zeros(1, 3)
		zeros(1, 3) zeros(1, 3) zeros(1, 3) T[2, 1] T[2, 2] T[2, 3]
	]
	# transMat = Matrix(I, 4, 4)
	FEMMethods.apply_tip_load!(globalF, elemType, transMat, "force")
	u = copy(globalF)

	# println("Global matrices")
	# println("globalK = ")
	# show(stdout, "text/plain", globalK)
	# println()
	# println("globalM = ")
	# show(stdout, "text/plain", globalM)
	# println()
	# println("globalF = ")
	# println(globalF)
	# writedlm("DebugFGlob.csv", globalF, ',')
	# writedlm("DebugKGlobMatrix.csv", globalK, ',')
	# writedlm("DebugMGlobMatrix.csv", globalM, ',')

	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	# println("Blanked matrices")
	# println("K = ")
	# show(stdout, "text/plain", K)
	# println()
	# # println("globalM = ")
	# # show(stdout, "text/plain", globalM)
	# # println()
	# println("F = ")
	# println(F)
	# # return K, F
	q1 = FEMMethods.solve_structure(K, M, F)


	# ************************************************
	#     Natural frequencies
	# ************************************************
	omegaSquared, modeShapes = SolverRoutines.compute_eigsolve(K, M, 3)

	return q1, K, M, F, omegaSquared, modeShapes

end


# q1, K, M, F, omegaSquared, modeShapes = test_FiniteElementBend()
# println("tip deflection = ", q1[end-1], " m")
# # println("tip rotation about y axis = ", q1[end-1], " rad")
# natFreqs = sqrt.(Complex.(omegaSquared)) / (2 * pi)

function test_FEBT3()
	"""
	Test the finite elements with unit loads, thickness, length, and structural moduli
	"""
	nNodes = 20
	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => 0.0 * π / 180, # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 1.0, # semispan [m]
		"ab" => zeros(nNodes), # dist from midchord to EA [m]
		"toc" => 1, # thickness-to-chord ratio
		"x_αb" => zeros(nNodes), # static imbalance [m]
		"θ" => 0 * π / 180, # fiber angle global [rad]
	)
	solverOptions = Dict(
		"material" => "test-comp", # preselect from material library
		"nNodes" => nNodes,
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
	)
	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	constitutive = FOIL.constitutive
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])

	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "BT3"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-4] = 1.0 # 0 Newton tip force
	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q3 = FEMMethods.solve_structure(K, M, F)
	BT2_Ftip_wtip = q3[end-4]
	BT2_Ftip_psitip = q3[end-1]

	# writedlm("DebugKGlobMatrix.csv", globalK, ',')
	# writedlm("DebugMGlobMatrix.csv", globalM, ',')
	# ---------------------------
	#   Tip torque only
	# ---------------------------
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	globalF[end-1] = 1.0 # 0 Newton tip force
	u = copy(globalF)

	# return q3

	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	q4 = FEMMethods.solve_structure(K, M, F)
	BT2_Ttip_wtip = q4[end-4]
	BT2_Ttip_psitip = q4[end-1]

	# --- Print these out if something does not make sense ---
	# println("bt_Ftip_wtip = ", bt_Ftip_wtip, " [m]")
	# println("bt_Ftip_psitip" = ", bt_Ftip_psitip", " [rad]")
	# println("bt_Ttip_wtip = ", bt_Ttip_wtip, " [m]")
	# println("bt_Ttip_psitip" = ", bt_Ttip_psitip", " [rad]")
	# println("BT2_Ftip_wtip = ", BT2_Ftip_wtip, " [m]")
	# println("BT2_Ftip_psitip = ", BT2_Ftip_psitip, " [rad]")
	# println("BT2_Ttip_wtip = ", BT2_Ttip_wtip, " [m]")
	# println("BT2_Ttip_psitip = ", BT2_Ttip_psitip, " [rad]")

	# --- Reference value ---
	# the tip deformations should be 4m for pure bending with tip force and 3 radians for tip torque
	# Of course, the tip torque for BT2 will be smaller since we prescribe the zero twist derivative BC at the root
	ref_sol = [4.0, 0.0, 0.0, 2.56699]

	# --- Relative error ---
	answers = [BT2_Ftip_wtip, BT2_Ftip_psitip, BT2_Ttip_wtip, BT2_Ttip_psitip] # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	omegaSquared, modeShapes = SolverRoutines.compute_eigsolve(K, M, 3)

	# println("f_n = ", sqrt.(omegaSquared)/(2*pi))

	return rel_err
end

# test_FiniteElementComp()
# ans = test_FEBT3()

function test_FECOMP2()
	"""
	Test simple tip force and torque on a composite beam
	"""
	nNodes = 40
	testAngle = 0.0 # [deg] angle of rotation from default (NONZERO WORKS  )
	angleDefault = deg2rad(-90) # default angle of rotation of the axes to match beam

	axisDefault = "z"

	DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => 0.0 * π / 180, # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 1.0, # semispan [m]
		"ab" => zeros(nNodes), # dist from midchord to EA [m]
		"toc" => 1, # thickness-to-chord ratio
		"x_αb" => zeros(nNodes), # static imbalance [m]
		"θ" => deg2rad(15), # fiber angle global [rad]
	)
	solverOptions = Dict(
		"material" => "test-comp", # preselect from material library
		"nNodes" => nNodes,
		"U∞" => 5.0, # free stream velocity [m/s]
		"ρ_f" => 1000.0, # fluid density [kg/m³]
	)

	FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)

	nElem = nNodes - 1
	constitutive = FOIL.constitutive
	structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"], rotation = testAngle)

	# ---------------------------
	#   Tip force only
	# ---------------------------
	elemType = "COMP2"
	globalDOFBlankingList = FEMMethods.get_fixed_dofs(elemType)
	abVec = DVDict["ab"]
	x_αbVec = DVDict["x_αb"]
	chordVec = DVDict["c"]
	ebVec = 0.25 * chordVec .+ abVec
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	# T1 = SolverRoutines.get_rotate3dMat(angleDefault, axis=axisDefault) 
	# T = T1
	T = I(3)
	transMatL2G = [
		T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T
	]
	FEMMethods.apply_tip_load!(globalF, elemType, transMatL2G, "force")

	u = copy(globalF)


	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

	u3 = FEMMethods.solve_structure(K, M, F)
	# transform back to local coordinates
	# T1 = SolverRoutines.get_rotate3dMat(angleDefault, axis=axisDefault) # question remaining why this is not the negative angle
	# T = T1
	transMatG2L = [
		T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3)
		zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T
	]
	q3 = transMatG2L[1:9, 1:9] * u3[end-8:end] # transform back to local coordinates
	BT2_Ftip_wtip = q3[end-6]
	BT2_Ftip_psitip = q3[end-5]
	println("tip displacements:", q3)

	# writedlm("DebugKGlobMatrix.csv", globalK, ',')
	# writedlm("DebugMGlobMatrix.csv", globalM, ',')
	# ---------------------------
	#   Tip torque only
	# ---------------------------
	globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
	FEMMethods.apply_tip_load!(globalF, elemType, transMatL2G, "torque")
	u = copy(globalF)

	K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)
	# writedlm("DebugK-COMP2.csv", K,",")
	# writedlm("DebugF-COMP2.csv", F,",")

	u4 = FEMMethods.solve_structure(K, M, F)
	q4 = transMatG2L[1:9, 1:9] * u4[end-8:end] # transform back to local coordinates
	BT2_Ttip_wtip = q4[end-6]
	BT2_Ttip_psitip = q4[end-5]
	println("tip displacements:", q4)

	# --- Print these out if something does not make sense ---
	println("COMP2_Ftip_wtip = ", BT2_Ftip_wtip, " [m]")
	println("COMP2_Ftip_psitip = ", rad2deg(BT2_Ftip_psitip), " [deg]")
	println("COMP2_Ttip_wtip = ", BT2_Ttip_wtip, " [m]")
	println("COMP2_Ttip_psitip = ", rad2deg(BT2_Ttip_psitip), " [deg]")

	# --- Reference value ---
	# the tip deformations should be 4m for pure bending with tip force and 3 radians for tip torque
	# Of course, the tip torque for BT2 will be smaller since we prescribe the zero twist derivative BC at the root
	ref_sol = [4.0, 0.0, 0.0, 2.56699]

	# --- Relative error ---
	answers = [BT2_Ftip_wtip, BT2_Ftip_psitip, BT2_Ttip_wtip, BT2_Ttip_psitip] # put computed solutions here
	rel_err = LinearAlgebra.norm(answers - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)

	omegaSquared, modeShapes = SolverRoutines.compute_eigsolve(K, M, 6)

	println("f_n = ", sqrt.(omegaSquared) / (2 * pi))

	return rel_err
end

# test_FECOMP2()
