# --- Julia 1.9 ---
"""
@File    :   LiftingLine.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   Modern lifting line from Phillips and Snyder 2000, Reid 2020 appendix
             The major weakness is the discontinuity in the locus of aerodynamic centers (LAC)
             for a highly swept wing at the root AND the mathematical requirement that the LAC 
             be locally perpendicular to the trailing vortex (TV). Reid 2020 overcame this by
             using a blending function at the wing root and a jointed TV

             Note: The origin is at the midchord of the root airfoil.
             x is positive streamwise
             y is positive to stbd
             z is positive in the vertical direction

             KNOWN BUGS:
             If there are NaNs, check the TV influence functions
             If there are NaNs in the derivatives, check the sweep angles
"""

module LiftingLine

# --- PACKAGES ---
using FLOWMath: abs_cs_safe, atan_cs_safe, norm_cs_safe
using Plots
using LinearAlgebra
using Statistics
using AbstractDifferentiation: AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @ignore_derivatives
using Zygote, ReverseDiff, ForwardDiff
using FiniteDifferences
using PythonCall
using Printf

# using Debugger
using DelimitedFiles

# --- DCFoil modules ---
for headerName in [
    "../constants/DataTypes",
    "../constants/SolutionConstants",
    "../adrules/CustomRules",
    "../constants/DesignConstants",
    "../utils/Utilities",
    "../utils/Interpolation",
    "../utils/Rotations",
    "../struct/EBBeam",
    "../utils/Preprocessing",
    "../hydro/Unsteady",
    "../hydro/VPM",
    "../solvers/NewtonRaphson",
    "../ComputeHydroFunctions",
]
    include(headerName * ".jl")
end

const Δα = 1e-3 # [rad] Finite difference step for lift slope calculations
const NPT_WING = 40 # this is known to be accurate
# const NPT_WING = 20

export LiftingLineNLParams, XDIM, YDIM, ZDIM, compute_LLresiduals, compute_LLresJacobian,
    compute_KS, GRAV

# ==============================================================================
#                         Structs
# ==============================================================================
struct LiftingLineMesh{T<:Number,TF<:Number,TI,TA<:AbstractVector,TM<:AbstractMatrix,TH<:AbstractArray}
    """
    Only geometry and mesh information
    Coordinates use the midchord root airfoil as the origin
    """
    nodePts::TM # LL node points
    collocationPts::TM # Control points [3 x npt]
    jointPts::AbstractMatrix # TV joint points
    npt_wing::TI # Number of wing points
    localChords::AbstractVector # Local chord lengths of the panel edges [m]
    localChordsCtrl::AbstractVector # Local chord lengths of the control points [m]
    sectionVectors::TM # Nondimensional section vectors, "dζi" in paper
    sectionLengths::TA # Section lengths
    sectionAreas::TA # Section areas
    # HydroProperties # Hydro properties at the cross sections
    npt_airfoil::TI # Number of airfoil points
    span::TF # Span of one wing
    # planformArea::TF
    # SRef::TF # Reference area [m^2]
    # AR::TF # Aspect ratio
    # span::Number # Span of one wing
    planformArea::Number
    SRef::Number # Reference area [m^2]
    AR::Number # Aspect ratio
    rootChord::Number # Root chord [m]
    # sweepAng::TF # Wing sweep angle [rad]
    sweepAng::Number # Wing sweep angle [rad]
    rc::T # Finite-core vortex radius (viscous correction) [m]
    wing_xyz_eff::TH # Effective wing LAC coordinates per control point
    wing_joint_xyz_eff::TH # Effective TV joint locations per control point
    local_sweeps::AbstractVector
    local_sweeps_eff::TM
    local_sweeps_ctrl::AbstractVector
end

struct LiftingLineHydro{TF,TM<:AbstractMatrix{TF}}
    """
    Hydro section properties
    """
    airfoil_CLa::TF # Airfoil lift slope ∂cl/∂α [1/rad]
    airfoil_aL0::TF # Airfoil zero-lift angle of attack [rad]
    airfoil_xy::TM # Airfoil coordinates
    airfoil_ctrl_xy::TM # Airfoil control points
end

struct FlowConditions{TF,TC,TA<:AbstractVector}
    Uinfvec::Vector # Freestream velocity [m/s] [U, V, W]
    Uinf::TC # Freestream velocity magnitude [m/s]
    uvec::TA # Freestream velocity unit vector
    alpha::TC # Angle of attack [rad]
    beta::Number
    rhof::TF # Freestream density [kg/m^3]
    depth::TF
end

struct LiftingLineOutputs{TF,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    """
    Nondimensional and dimensional outputs of interest
    Redimensionalize with the reference area and velocity
    """
    Fdist::TM # Loads distribution vector (Dimensional) [N] [3 x npt]
    Γdist::TA # Converged circulation distribution (Γᵢ) [m^2/s]
    cla::TA # Spanwise lift slopes [1/rad]
    cl::TA # Spanwise lift coefficients
    # F::TA # Total integrated loads vector [Fx, Fy, Fz] [N]
    F::TA # Total integrated loads vector perpendicular to flow directions [Fdrag, Fside, Flift] [N]
    CL::TF # Lift coefficient (perpendicular to freestream in symmetry plane)
    CDi::TF # Induced drag coefficient (aligned w/ freestream)
    CS::TF # Side force coefficient
end

struct LiftingLineNLParams
    """
    Parameters needed in the nonlinear solve of the LL
    """
    TV_influence
    LLSystem # LiftingLineMesh
    LLHydro # LiftingLineHydro
    FlowCond # Flow conditions
    # Stuff for the 2D VPM solve
    Airfoils #
    AirfoilInfluences #
end

function initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)


    # --- NACA 0012 ---
    airfoilX = [1.00000000e+00, 9.98993338e-01, 9.95977406e-01, 9.90964349e-01, 9.83974351e-01, 9.75035559e-01, 9.64183967e-01, 9.51463269e-01, 9.36924689e-01, 9.20626766e-01, 9.02635129e-01, 8.83022222e-01, 8.61867019e-01, 8.39254706e-01, 8.15276334e-01, 7.90028455e-01, 7.63612734e-01, 7.36135537e-01, 7.07707507e-01, 6.78443111e-01, 6.48460188e-01, 6.17879468e-01, 5.86824089e-01, 5.55419100e-01, 5.23790958e-01, 4.92067018e-01, 4.60375022e-01, 4.28842581e-01, 3.97596666e-01, 3.66763093e-01, 3.36466018e-01, 3.06827437e-01, 2.77966694e-01, 2.50000000e-01, 2.23039968e-01, 1.97195156e-01, 1.72569633e-01, 1.49262556e-01, 1.27367775e-01, 1.06973453e-01, 8.81617093e-02, 7.10082934e-02, 5.55822757e-02, 4.19457713e-02, 3.01536896e-02, 2.02535132e-02, 1.22851066e-02, 6.28055566e-03, 2.26403871e-03, 2.51728808e-04, 2.51728808e-04, 2.26403871e-03, 6.28055566e-03, 1.22851066e-02, 2.02535132e-02, 3.01536896e-02, 4.19457713e-02, 5.55822757e-02, 7.10082934e-02, 8.81617093e-02, 1.06973453e-01, 1.27367775e-01, 1.49262556e-01, 1.72569633e-01, 1.97195156e-01, 2.23039968e-01, 2.50000000e-01, 2.77966694e-01, 3.06827437e-01, 3.36466018e-01, 3.66763093e-01, 3.97596666e-01, 4.28842581e-01, 4.60375022e-01, 4.92067018e-01, 5.23790958e-01, 5.55419100e-01, 5.86824089e-01, 6.17879468e-01, 6.48460188e-01, 6.78443111e-01, 7.07707507e-01, 7.36135537e-01, 7.63612734e-01, 7.90028455e-01, 8.15276334e-01, 8.39254706e-01, 8.61867019e-01, 8.83022222e-01, 9.02635129e-01, 9.20626766e-01, 9.36924689e-01, 9.51463269e-01, 9.64183967e-01, 9.75035559e-01, 9.83974351e-01, 9.90964349e-01, 9.95977406e-01, 9.98993338e-01, 1.00000000e+00]
    airfoilY = [1.33226763e-17, -1.41200438e-04, -5.63343432e-04, -1.26208774e-03,
        -2.23030811e-03, -3.45825298e-03, -4.93375074e-03, -6.64245132e-03,
        -8.56808791e-03, -1.06927428e-02, -1.29971013e-02, -1.54606806e-02,
        -1.80620201e-02, -2.07788284e-02, -2.35880799e-02, -2.64660655e-02,
        -2.93884006e-02, -3.23300020e-02, -3.52650469e-02, -3.81669313e-02,
        -4.10082448e-02, -4.37607812e-02, -4.63956016e-02, -4.88831639e-02,
        -5.11935307e-02, -5.32966591e-02, -5.51627743e-02, -5.67628192e-02,
        -5.80689678e-02, -5.90551853e-02, -5.96978089e-02, -5.99761253e-02,
        -5.98729133e-02, -5.93749219e-02, -5.84732545e-02, -5.71636340e-02,
        -5.54465251e-02, -5.33271006e-02, -5.08150416e-02, -4.79241724e-02,
        -4.46719377e-02, -4.10787401e-02, -3.71671601e-02, -3.29610920e-02,
        -2.84848303e-02, -2.37621474e-02, -1.88154040e-02, -1.36647314e-02,
        -8.32732576e-03, -2.81688492e-03, 2.81688492e-03, 8.32732576e-03,
        1.36647314e-02, 1.88154040e-02, 2.37621474e-02, 2.84848303e-02,
        3.29610920e-02, 3.71671601e-02, 4.10787401e-02, 4.46719377e-02,
        4.79241724e-02, 5.08150416e-02, 5.33271006e-02, 5.54465251e-02,
        5.71636340e-02, 5.84732545e-02, 5.93749219e-02, 5.98729133e-02,
        5.99761253e-02, 5.96978089e-02, 5.90551853e-02, 5.80689678e-02,
        5.67628192e-02, 5.51627743e-02, 5.32966591e-02, 5.11935307e-02,
        4.88831639e-02, 4.63956016e-02, 4.37607812e-02, 4.10082448e-02,
        3.81669313e-02, 3.52650469e-02, 3.23300020e-02, 2.93884006e-02,
        2.64660655e-02, 2.35880799e-02, 2.07788284e-02, 1.80620201e-02,
        1.54606806e-02, 1.29971013e-02, 1.06927428e-02, 8.56808791e-03,
        6.64245132e-03, 4.93375074e-03, 3.45825298e-03, 2.23030811e-03,
        1.26208774e-03, 5.63343432e-04, 1.41200438e-04, -1.33226763e-17]
    airfoilCtrlX = [9.99496669e-01, 9.97485372e-01, 9.93470878e-01, 9.87469350e-01,
        9.79504955e-01, 9.69609763e-01, 9.57823618e-01, 9.44193979e-01,
        9.28775727e-01, 9.11630948e-01, 8.92828675e-01, 8.72444620e-01,
        8.50560862e-01, 8.27265520e-01, 8.02652394e-01, 7.76820594e-01,
        7.49874136e-01, 7.21921522e-01, 6.93075309e-01, 6.63451649e-01,
        6.33169828e-01, 6.02351778e-01, 5.71121594e-01, 5.39605029e-01,
        5.07928988e-01, 4.76221020e-01, 4.44608801e-01, 4.13219623e-01,
        3.82179880e-01, 3.51614556e-01, 3.21646728e-01, 2.92397065e-01,
        2.63983347e-01, 2.36519984e-01, 2.10117562e-01, 1.84882395e-01,
        1.60916095e-01, 1.38315166e-01, 1.17170614e-01, 9.75675810e-02,
        7.95850013e-02, 6.32952845e-02, 4.87640235e-02, 3.60497304e-02,
        2.52036014e-02, 1.62693099e-02, 9.28283111e-03, 4.27229719e-03,
        1.25788376e-03, 2.51728808e-04, 1.25788376e-03, 4.27229719e-03,
        9.28283111e-03, 1.62693099e-02, 2.52036014e-02, 3.60497304e-02,
        4.87640235e-02, 6.32952845e-02, 7.95850013e-02, 9.75675810e-02,
        1.17170614e-01, 1.38315166e-01, 1.60916095e-01, 1.84882395e-01,
        2.10117562e-01, 2.36519984e-01, 2.63983347e-01, 2.92397065e-01,
        3.21646728e-01, 3.51614556e-01, 3.82179880e-01, 4.13219623e-01,
        4.44608801e-01, 4.76221020e-01, 5.07928988e-01, 5.39605029e-01,
        5.71121594e-01, 6.02351778e-01, 6.33169828e-01, 6.63451649e-01,
        6.93075309e-01, 7.21921522e-01, 7.49874136e-01, 7.76820594e-01,
        8.02652394e-01, 8.27265520e-01, 8.50560862e-01, 8.72444620e-01,
        8.92828675e-01, 9.11630948e-01, 9.28775727e-01, 9.44193979e-01,
        9.57823618e-01, 9.69609763e-01, 9.79504955e-01, 9.87469350e-01,
        9.93470878e-01, 9.97485372e-01, 9.99496669e-01]
    airfoilCtrlY = [-7.06002188e-05, -3.52271935e-04, -9.12715585e-04, -1.74619792e-03,
        -2.84428054e-03, -4.19600186e-03, -5.78810103e-03, -7.60526962e-03,
        -9.63041534e-03, -1.18449221e-02, -1.42288910e-02, -1.67613504e-02,
        -1.94204243e-02, -2.21834542e-02, -2.50270727e-02, -2.79272331e-02,
        -3.08592013e-02, -3.37975244e-02, -3.67159891e-02, -3.95875880e-02,
        -4.23845130e-02, -4.50781914e-02, -4.76393828e-02, -5.00383473e-02,
        -5.22450949e-02, -5.42297167e-02, -5.59627967e-02, -5.74158935e-02,
        -5.85620766e-02, -5.93764971e-02, -5.98369671e-02, -5.99245193e-02,
        -5.96239176e-02, -5.89240882e-02, -5.78184443e-02, -5.63050796e-02,
        -5.43868129e-02, -5.20710711e-02, -4.93696070e-02, -4.62980550e-02,
        -4.28753389e-02, -3.91229501e-02, -3.50641261e-02, -3.07229612e-02,
        -2.61234889e-02, -2.12887757e-02, -1.62400677e-02, -1.09960286e-02,
        -5.57210534e-03, 0.00000000e+00, 5.57210534e-03, 1.09960286e-02,
        1.62400677e-02, 2.12887757e-02, 2.61234889e-02, 3.07229612e-02,
        3.50641261e-02, 3.91229501e-02, 4.28753389e-02, 4.62980550e-02,
        4.93696070e-02, 5.20710711e-02, 5.43868129e-02, 5.63050796e-02,
        5.78184443e-02, 5.89240882e-02, 5.96239176e-02, 5.99245193e-02,
        5.98369671e-02, 5.93764971e-02, 5.85620766e-02, 5.74158935e-02,
        5.59627967e-02, 5.42297167e-02, 5.22450949e-02, 5.00383473e-02,
        4.76393828e-02, 4.50781914e-02, 4.23845130e-02, 3.95875880e-02,
        3.67159891e-02, 3.37975244e-02, 3.08592013e-02, 2.79272331e-02,
        2.50270727e-02, 2.21834542e-02, 1.94204243e-02, 1.67613504e-02,
        1.42288910e-02, 1.18449221e-02, 9.63041534e-03, 7.60526962e-03,
        5.78810103e-03, 4.19600186e-03, 2.84428054e-03, 1.74619792e-03,
        9.12715585e-04, 3.52271935e-04, 7.06002188e-05]

    # # --- NACA 0015 ---
    # airfoilX = [1.00000000e+00, 9.98993338e-01, 9.95977406e-01, 9.90964349e-01, 9.83974351e-01, 9.75035559e-01, 9.64183967e-01, 9.51463269e-01, 9.36924689e-01, 9.20626766e-01, 9.02635129e-01, 8.83022222e-01, 8.61867019e-01, 8.39254706e-01, 8.15276334e-01, 7.90028455e-01, 7.63612734e-01, 7.36135537e-01, 7.07707507e-01, 6.78443111e-01, 6.48460188e-01, 6.17879468e-01, 5.86824089e-01, 5.55419100e-01, 5.23790958e-01, 4.92067018e-01, 4.60375022e-01, 4.28842581e-01, 3.97596666e-01, 3.66763093e-01, 3.36466018e-01, 3.06827437e-01, 2.77966694e-01, 2.50000000e-01, 2.23039968e-01, 1.97195156e-01, 1.72569633e-01, 1.49262556e-01, 1.27367775e-01, 1.06973453e-01, 8.81617093e-02, 7.10082934e-02, 5.55822757e-02, 4.19457713e-02, 3.01536896e-02, 2.02535132e-02, 1.22851066e-02, 6.28055566e-03, 2.26403871e-03, 2.51728808e-04, 2.51728808e-04, 2.26403871e-03, 6.28055566e-03, 1.22851066e-02, 2.02535132e-02, 3.01536896e-02, 4.19457713e-02, 5.55822757e-02, 7.10082934e-02, 8.81617093e-02, 1.06973453e-01, 1.27367775e-01, 1.49262556e-01, 1.72569633e-01, 1.97195156e-01, 2.23039968e-01, 2.50000000e-01, 2.77966694e-01, 3.06827437e-01, 3.36466018e-01, 3.66763093e-01, 3.97596666e-01, 4.28842581e-01, 4.60375022e-01, 4.92067018e-01, 5.23790958e-01, 5.55419100e-01, 5.86824089e-01, 6.17879468e-01, 6.48460188e-01, 6.78443111e-01, 7.07707507e-01, 7.36135537e-01, 7.63612734e-01, 7.90028455e-01, 8.15276334e-01, 8.39254706e-01, 8.61867019e-01, 8.83022222e-01, 9.02635129e-01, 9.20626766e-01, 9.36924689e-01, 9.51463269e-01, 9.64183967e-01, 9.75035559e-01, 9.83974351e-01, 9.90964349e-01, 9.95977406e-01, 9.98993338e-01, 1.00000000e+00]
    # airfoilY = [1.66533454e-17, -1.76500547e-04, -7.04179290e-04, -1.57760967e-03, -2.78788513e-03, -4.32281622e-03, -6.16718843e-03, -8.30306415e-03, -1.07101099e-02, -1.33659285e-02, -1.62463767e-02, -1.93258507e-02, -2.25775252e-02, -2.59735355e-02, -2.94850999e-02, -3.30825819e-02, -3.67355008e-02, -4.04125025e-02, -4.40813086e-02, -4.77086641e-02, -5.12603060e-02, -5.47009765e-02, -5.79945020e-02, -6.11039549e-02, -6.39919134e-02, -6.66208239e-02, -6.89534679e-02, -7.09535239e-02, -7.25862098e-02, -7.38189816e-02, -7.46222611e-02, -7.49701566e-02, -7.48411416e-02, -7.42186523e-02, -7.30915682e-02, -7.14545425e-02, -6.93081564e-02, -6.66588758e-02, -6.35188020e-02, -5.99052154e-02, -5.58399221e-02, -5.13484251e-02, -4.64589502e-02, -4.12013650e-02, -3.56060379e-02, -2.97026843e-02, -2.35192550e-02, -1.70809143e-02, -1.04091572e-02, -3.52110615e-03, 3.52110615e-03, 1.04091572e-02, 1.70809143e-02, 2.35192550e-02, 2.97026843e-02, 3.56060379e-02, 4.12013650e-02, 4.64589502e-02, 5.13484251e-02, 5.58399221e-02, 5.99052154e-02, 6.35188020e-02, 6.66588758e-02, 6.93081564e-02, 7.14545425e-02, 7.30915682e-02, 7.42186523e-02, 7.48411416e-02, 7.49701566e-02, 7.46222611e-02, 7.38189816e-02, 7.25862098e-02, 7.09535239e-02, 6.89534679e-02, 6.66208239e-02, 6.39919134e-02, 6.11039549e-02, 5.79945020e-02, 5.47009765e-02, 5.12603060e-02, 4.77086641e-02, 4.40813086e-02, 4.04125025e-02, 3.67355008e-02, 3.30825819e-02, 2.94850999e-02, 2.59735355e-02, 2.25775252e-02, 1.93258507e-02, 1.62463767e-02, 1.33659285e-02, 1.07101099e-02, 8.30306415e-03, 6.16718843e-03, 4.32281622e-03, 2.78788513e-03, 1.57760967e-03, 7.04179290e-04, 1.76500547e-04, -1.66533454e-17]
    # airfoilCtrlX = [9.99496669e-01, 9.97485372e-01, 9.93470878e-01, 9.87469350e-01, 9.79504955e-01, 9.69609763e-01, 9.57823618e-01, 9.44193979e-01, 9.28775727e-01, 9.11630948e-01, 8.92828675e-01, 8.72444620e-01, 8.50560862e-01, 8.27265520e-01, 8.02652394e-01, 7.76820594e-01, 7.49874136e-01, 7.21921522e-01, 6.93075309e-01, 6.63451649e-01, 6.33169828e-01, 6.02351778e-01, 5.71121594e-01, 5.39605029e-01, 5.07928988e-01, 4.76221020e-01, 4.44608801e-01, 4.13219623e-01, 3.82179880e-01, 3.51614556e-01, 3.21646728e-01, 2.92397065e-01, 2.63983347e-01, 2.36519984e-01, 2.10117562e-01, 1.84882395e-01, 1.60916095e-01, 1.38315166e-01, 1.17170614e-01, 9.75675810e-02, 7.95850013e-02, 6.32952845e-02, 4.87640235e-02, 3.60497304e-02, 2.52036014e-02, 1.62693099e-02, 9.28283111e-03, 4.27229719e-03,
    #     1.25788376e-03, 2.51728808e-04, 1.25788376e-03, 4.27229719e-03, 9.28283111e-03, 1.62693099e-02, 2.52036014e-02, 3.60497304e-02, 4.87640235e-02, 6.32952845e-02, 7.95850013e-02, 9.75675810e-02, 1.17170614e-01, 1.38315166e-01, 1.60916095e-01, 1.84882395e-01, 2.10117562e-01, 2.36519984e-01, 2.63983347e-01, 2.92397065e-01, 3.21646728e-01, 3.51614556e-01, 3.82179880e-01, 4.13219623e-01, 4.44608801e-01, 4.76221020e-01, 5.07928988e-01, 5.39605029e-01, 5.71121594e-01, 6.02351778e-01, 6.33169828e-01, 6.63451649e-01, 6.93075309e-01, 7.21921522e-01, 7.49874136e-01, 7.76820594e-01, 8.02652394e-01, 8.27265520e-01, 8.50560862e-01, 8.72444620e-01, 8.92828675e-01, 9.11630948e-01, 9.28775727e-01, 9.44193979e-01, 9.57823618e-01, 9.69609763e-01, 9.79504955e-01, 9.87469350e-01, 9.93470878e-01, 9.97485372e-01, 9.99496669e-01]
    # airfoilCtrlY = [-8.82502736e-05, -4.40339919e-04, -1.14089448e-03, -2.18274740e-03, -3.55535068e-03, -5.24500233e-03, -7.23512629e-03, -9.50658702e-03, -1.20380192e-02, -1.48061526e-02, -1.77861137e-02, -2.09516879e-02, -2.42755303e-02, -2.77293177e-02, -3.12838409e-02, -3.49090413e-02, -3.85740016e-02, -4.22469055e-02, -4.58949864e-02, -4.94844850e-02, -5.29806412e-02, -5.63477392e-02, -5.95492284e-02, -6.25479342e-02, -6.53063686e-02, -6.77871459e-02, -6.99534959e-02, -7.17698669e-02, -7.32025957e-02, -7.42206214e-02, -7.47962088e-02, -7.49056491e-02, -7.45298970e-02, -7.36551103e-02, -7.22730553e-02, -7.03813495e-02, -6.79835161e-02, -6.50888389e-02, -6.17120087e-02, -5.78725688e-02, -5.35941736e-02, -4.89036876e-02, -4.38301576e-02, -3.84037014e-02, -3.26543611e-02, -2.66109696e-02, -2.03000846e-02, -1.37450357e-02, -6.96513168e-03, 0.00000000e+00, 6.96513168e-03, 1.37450357e-02, 2.03000846e-02, 2.66109696e-02, 3.26543611e-02, 3.84037014e-02, 4.38301576e-02, 4.89036876e-02, 5.35941736e-02, 5.78725688e-02, 6.17120087e-02, 6.50888389e-02, 6.79835161e-02, 7.03813495e-02, 7.22730553e-02, 7.36551103e-02, 7.45298970e-02, 7.49056491e-02, 7.47962088e-02, 7.42206214e-02, 7.32025957e-02, 7.17698669e-02, 6.99534959e-02, 6.77871459e-02, 6.53063686e-02, 6.25479342e-02, 5.95492284e-02, 5.63477392e-02, 5.29806412e-02, 4.94844850e-02, 4.58949864e-02, 4.22469055e-02, 3.85740016e-02, 3.49090413e-02, 3.12838409e-02, 2.77293177e-02, 2.42755303e-02, 2.09516879e-02, 1.77861137e-02, 1.48061526e-02, 1.20380192e-02, 9.50658702e-03, 7.23512629e-03, 5.24500233e-03, 3.55535068e-03, 2.18274740e-03, 1.14089448e-03, 4.40339919e-04, 8.82502736e-05]

    airfoilXY = copy(transpose(hcat(airfoilX, airfoilY)))
    airfoilCtrlXY = copy(transpose(hcat(airfoilCtrlX, airfoilCtrlY)))
    npt_wing = NPT_WING
    npt_airfoil = 99

    rootChord = chordVec[1]
    TR = chordVec[end] / rootChord

    Uvec = [cos(deg2rad(α0)), 0.0, sin(deg2rad(α0))] * solverOptions["Uinf"]

    # Rotate by RH rule by leeway angle
    Tz = get_rotate3dMat(deg2rad(β0), "z")
    Uvec = Tz * Uvec

    if solverOptions["use_freeSurface"]
        depth = depth0
    else
        depth = depth0 * 100
    end

    options = Dict(
        "translation" => vec([appendageOptions["xMount"], 0, 0]), # of the midchord
        "debug" => true,
        "depth" => depth,
        "is_antisymmetry" => false,
    )
    return airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options
end

function setup(Uvec, sweepAng, rootChord, taperRatio, midchords, displacements::AbstractMatrix, preTwist;
    npt_wing=99, npt_airfoil=199, blend=0.25, δ=0.15, rc=0.0, rhof=1025.0,
    airfoil_xy=nothing, airfoil_ctrl_xy=nothing, airfoilCoordFile=nothing, options=nothing)
    """
    Initialize and setup the lifting line model for one wing

    Inputs:
    -------
    displacements : array
        The displacements of the wing collocation nodes [m] size 6 x npt_wing.
        This modifies the collocation nodes
    preTwist : array
        The pre-twist angles [rad] same size as midchords array.
    wingSpan : scalar
        The span of the wing [m] (after sweep is applied, so this is not the structural span!)
    sweepAng : scalar
        The wing sweep angle in rad.
    blend : scalar , optional
        The normalized blending distance, used to calculate the
        effective loci of aerodynamic centers.
    δ : scalar 0.15, optional
        The fraction of the local chord the vortex segment portion of the
        TV extends from the LAC.
    rc : scalar 0.0, optional
        The finite-core vortex radius (viscous correction) [m]
    airfoilCoordFile : filename, optional
        The filename of the airfoil coordinates to use. If not
        provided, the airfoil_xy and airfoil_ctrl_xy arrays are used
    options : dict, optional
        Dictionary of options to pass to the lifting line model regarding debug stuff
    """

    # ************************************************
    #     Airfoil hydro properties
    # ************************************************
    if !isnothing(airfoilCoordFile) && isnothing(airfoil_xy) && isnothing(airfoil_ctrl_xy)

        println("Reading airfoil coordinates from $(airfoilCoordFile) and using MACH...")

        PREFOIL = pyimport("prefoil")

        rawCoords = PREFOIL.utils.readCoordFile(airfoilCoordFile)

        Foil = PREFOIL.Airfoil(rawCoords)

        Foil.normalizeChord()

        airfoil_pts = Foil.getSampledPts(
            nPts=npt_airfoil + 1, # one more to delete the TE knot
            spacingFunc=PREFOIL.sampling.conical,
            func_args=Dict("coeff" => 1),
            TE_knot=false # weird stuff going on with a trailing knot
        )

        airfoil_ctrl_pts = (airfoil_pts[1:end-2, :] .+ airfoil_pts[2:end-1, :]) .* 0.5

        # --- Transpose and reverse since PreFoil is different ---
        airfoil_xy = reverse(transpose(airfoil_pts[1:end-1, :]), dims=2)
        airfoil_ctrl_xy = reverse(transpose(airfoil_ctrl_pts), dims=2)

        # elseif !isnothing(airfoil_xy) && !isnothing(airfoil_ctrl_xy)
        #     println("Using provided airfoil coordinates")
    end
    # The initial hydro properties use zero sweep
    LLHydro, Airfoil, Airfoil_influences = compute_hydroProperties(0.0, airfoil_xy, airfoil_ctrl_xy)

    # ************************************************
    #     Preproc stuff
    # ************************************************
    # --- Structural span is not the same as aero span ---
    idxTip = get_tipnode(midchords)
    aeroWingSpan = compute_aeroSpan(midchords, idxTip)
    # println("Aero span: $(aeroWingSpan) m")

    # wingSpan = span * cos(sweepAng) #no

    # Blending parameter for the LAC
    σ = 4 * cos(sweepAng)^2 / (blend^2 * aeroWingSpan^2)

    alpha, beta, Uinf = compute_cartAnglesFromVector(Uvec)
    uvec = Uvec / Uinf

    # Wing area
    SRef = rootChord * aeroWingSpan * (1 + taperRatio) * 0.5
    AR = aeroWingSpan^2 / SRef

    # ************************************************
    #     Make wing coordinates
    # ************************************************
    # ---------------------------
    #   Y coords (span)
    # ---------------------------
    start = -aeroWingSpan * 0.5
    stop = aeroWingSpan * 0.5

    # # --- Even spacing ---
    # θ_bound = LinRange(start, stop, npt_wing * 2 + 1)
    # wing_xyz_ycomp = reshape(θ_bound[1:2:end], 1, npt_wing + 1)
    # wing_ctrl_xyz_ycomp = reshape(θ_bound[2:2:end], 1, npt_wing)

    # --- Cosine spacing ---
    # if abs_cs_safe(sweepAng) > 0.0 # actually I think this introduces discontinuity wrt sweep angle derivative
    # θ_bound = PREFOIL.sampling.cosine(start, stop, npt_wing * 2 + 1, 2π)
    # println("θ_bound: $(θ_bound)")
    θ_bound = LinRange(0.0, 2π, npt_wing * 2 + 1)
    wing_xyz_ycomp = reshape([sign(θ - π) * 0.25 * aeroWingSpan * (1 + cos(θ)) for θ in θ_bound[1:2:end]], 1, npt_wing + 1)
    wing_ctrl_xyz_ycomp = reshape([sign(θ - π) * 0.25 * aeroWingSpan * (1 + cos(θ)) for θ in θ_bound[2:2:end]], 1, npt_wing)
    # end

    Zeros = zeros(1, npt_wing + 1)
    ZerosCtrl = zeros(1, npt_wing)
    wing_xyz = cat(Zeros, wing_xyz_ycomp, Zeros, dims=1)
    wing_ctrl_xyz = cat(ZerosCtrl, wing_ctrl_xyz_ycomp, ZerosCtrl, dims=1)

    # Interpolate the pre-twist angles to the collocation points
    idxSort = sortperm(midchords[YDIM, :])
    preTwistCtrl = do_linear_interp(midchords[YDIM, idxSort], preTwist[idxSort], wing_ctrl_xyz_ycomp)

    # ---------------------------
    #   X coords (chord dist)
    # ---------------------------
    iTR = 1.0 - taperRatio

    local_chords = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_xyz_ycomp[1, :]) / aeroWingSpan)
    local_chords_ctrl = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_ctrl_xyz_ycomp[1, :]) / aeroWingSpan)

    # ∂c/∂y
    local_dchords = 2.0 * rootChord * (-iTR) * sign.(wing_xyz_ycomp[1, :]) / aeroWingSpan
    local_dchords_ctrl = 2.0 * rootChord * (-iTR) * sign.(wing_ctrl_xyz_ycomp[1, :]) / aeroWingSpan

    # ---------------------------
    #   Shift collocation points
    # ---------------------------
    # --- Handle displacements of collocation nodes ---
    size(displacements) == (6, npt_wing) || error("Displacements must be 6 x $(npt_wing). Size is $(size(displacements))")
    translatDisplCtrl = cat(displacements[1:2, :], zeros(1, length(displacements[1, :])), dims=1) # Ignore any effect of dihedral on the wing
    rotationDisplacementsCtrl = displacements[4:end, :]
    # For the displacements on the panel edges, we want to use the extrapolated edge values for the tips and the average for the inner vals
    # However, this is not differentiable so we just use the edge values
    averages = translatDisplCtrl[:, 1:end-1] .+ translatDisplCtrl[:, 2:end] * 0.5
    midVals = cat(averages[:, 1:npt_wing÷2-1], zeros(3, 1), averages[:, npt_wing÷2+1:end], dims=2)
    portTip = translatDisplCtrl[:, 1]# + portSlope * halfDist

    stbdTip = translatDisplCtrl[:, end] #+ stbdSlope * halfDist

    # println("Stbd tip: $(stbdTip)")
    translatDispl = cat(portTip, midVals, stbdTip, dims=2)
    if options["is_antisymmetry"]
        # TODO GGGGGGGGG
        translatDisplCtrl[:,1:npt_wing÷2] *= -1.0 # flip 
        rotationDisplacementsCtrl[:,1:npt_wing÷2] *= -1.0 # flip
        # swap twists and displacements
        midVals = cat(
            -averages[:, 1:npt_wing÷2-1],
            zeros(3, 1),
            averages[:, npt_wing÷2+1:end],
            dims=2)
        translatDispl = cat(-portTip, midVals, stbdTip, dims=2)
        println("Using antisymmetry conditions")
        show(stdout, "text/plain", translatDispl)
    end

    # --- x shift setup ---
    # Apply translation wrt using the root airfoil midchord as origin 
    # This origin is different from Reid 2020 who used the root airfoil LE as the origin
    rootsemichord = 0.5 * rootChord
    translationVec = zeros(3) - vec([rootsemichord, 0.0, 0.0])
    if !isnothing(options) && haskey(options, "translation")
        translationVec = options["translation"] - vec([rootsemichord, 0.0, 0.0]) # [m] 3d translation of the wing
    end
    translatMatCtrl = repeat(reshape(translationVec, size(translationVec)..., 1), 1, npt_wing) + translatDisplCtrl # [3, npt_wing]
    translatMat = repeat(reshape(translationVec, size(translationVec)..., 1), 1, npt_wing + 1) + translatDispl # [3, npt_wing + 1]


    # --- Locus of aerodynamic centers (LAC) ---
    LAC = compute_LAC(AR, LLHydro, wing_xyz_ycomp[1, :], local_chords, rootChord, sweepAng, aeroWingSpan)
    wing_xyz = cat(reshape(LAC, 1, size(LAC)...), wing_xyz_ycomp, Zeros, dims=1) .+ translatMat

    LAC_ctrl = compute_LAC(AR, LLHydro, wing_ctrl_xyz_ycomp[1, :], local_chords_ctrl, rootChord, sweepAng, aeroWingSpan)
    wing_ctrl_xyz = cat(reshape(LAC_ctrl, 1, size(LAC_ctrl)...), wing_ctrl_xyz_ycomp, ZerosCtrl, dims=1) .+ translatMatCtrl

    # Need a mess of LAC's for each control point
    LACeff = compute_LACeffective(AR, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, aeroWingSpan)
    # This is a 3D array of a shape
    # wing_xyz_eff = zeros(3, npt_wing, npt_wing + 1)
    # The idea is that for the 'npt_wing' control points
    # Make sure to also add the translation vector to the effective locus of aerodynamic centers
    wing_xyz_eff_xcomp = reshape(LACeff, 1, size(LACeff)...) .+ translationVec[XDIM]
    wing_xyz_eff_ycomp = reshape(repeat(transpose(wing_xyz[YDIM, :]), npt_wing, 1), 1, npt_wing, npt_wing + 1)
    wing_xyz_eff_zcomp = reshape(repeat(transpose(wing_xyz[ZDIM, :]), npt_wing, 1), 1, npt_wing, npt_wing + 1)
    wing_xyz_eff = cat(
        wing_xyz_eff_xcomp,
        wing_xyz_eff_ycomp,
        wing_xyz_eff_zcomp,
        dims=1)

    # --- Compute local sweeps ---
    # Vectors containing local sweep at each coordinate location in wing_xyz
    fprime = compute_dLACds(AR, LLHydro, wing_xyz[YDIM, :], local_chords, local_dchords, sweepAng, aeroWingSpan)
    localSweeps = -atan_cs_safe.(fprime, ones(size(fprime)))

    fprimeCtrl = compute_dLACds(AR, LLHydro, wing_ctrl_xyz[YDIM, :], local_chords_ctrl, local_dchords_ctrl, sweepAng, aeroWingSpan)
    localSweepsCtrl = -atan_cs_safe.(fprimeCtrl, ones(size(fprimeCtrl)))

    fprimeEff = compute_dLACdseffective(AR, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, aeroWingSpan)
    localSweepEff = -atan_cs_safe.(fprimeEff, ones(size(fprimeEff)))

    # --- Compute local dihedrals ---
    # TODO: need to do this eventually to generalize the code.

    # --- Other section properties ---
    sectionVectors = wing_xyz[:, 1:end-1] - wing_xyz[:, 2:end] # dℓᵢ

    sectionLengths = .√(sectionVectors[XDIM, :] .^ 2 + sectionVectors[YDIM, :] .^ 2 + sectionVectors[ZDIM, :] .^ 2) # ℓᵢ
    sectionAreas = 0.5 * (local_chords[1:end-1] + local_chords[2:end]) .* abs_cs_safe.(wing_xyz[YDIM, 1:end-1] - wing_xyz[YDIM, 2:end]) # dAᵢ

    ζ = sectionVectors ./ reshape(sectionAreas, 1, size(sectionAreas)...) # Normalized section vectors, [3, npt_wing]

    # ---------------------------
    #   Aero section properties
    # ---------------------------
    # Where the 2D VPM comes into play
    Airfoils = Vector(undef, npt_wing)
    AirfoilInfluences = Vector(undef, npt_wing)
    Airfoils_z = Zygote.Buffer(Airfoils)
    AirfoilInfluences_z = Zygote.Buffer(AirfoilInfluences)
    for (ii, sweep) in enumerate(localSweepsCtrl)

        twistAngle = rotationDisplacementsCtrl[YDIM, ii] + preTwistCtrl[ii]

        # The airfoils are rotated by the negative twist angle to be consistent with the wing
        ryMat = get_rotate3dMat(-twistAngle, "z")[1:2, 1:2]
        airfoil_xy_rot = ryMat * airfoil_xy
        airfoil_ctrl_xy_rot = ryMat * airfoil_ctrl_xy

        # p1 = plot(airfoil_xy_rot[XDIM, :], airfoil_xy_rot[YDIM, :], aspect_ratio=:equal, label="$(twistAngle)")
        # p2 = plot(airfoil_ctrl_xy_rot[XDIM, :], airfoil_ctrl_xy_rot[YDIM, :], aspect_ratio=:equal)
        # plot(p1, p2, layout=(1, 2), size=(1200, 400))

        # savefig("airfoil_rotated_$(ii).png")

        # Pass in copies because this routine was modifying the input
        Airfoil, Airfoil_influences = setup_VPM(copy(airfoil_xy_rot[XDIM, :]), copy(airfoil_xy_rot[YDIM, :]), copy(airfoil_ctrl_xy_rot), sweep)
        Airfoils_z[ii] = Airfoil
        AirfoilInfluences_z[ii] = Airfoil_influences
        # Airfoils[ii] = Airfoil
        # AirfoilInfluences[ii] = Airfoil_influences
    end
    Airfoils = copy(Airfoils_z)
    AirfoilInfluences = copy(AirfoilInfluences_z)

    # # List comprehension version
    # Airfoils, AirfoilInfluences = [ setup_VPM(copy(airfoil_xy[XDIM, :]), copy(airfoil_xy[YDIM, :]), copy(airfoil_ctrl_xy), sweep) for sweep in localSweepsCtrl]

    # ---------------------------
    #   TV joint locations
    # ---------------------------
    # These are where the bound vortex lines kink and then bend to follow the freestream direction

    local_chords_colmat = reshape(local_chords, 1, size(local_chords)...)

    wing_joint_xyz_xcomp = reshape(wing_xyz[XDIM, :] + δ * local_chords .* cos.(localSweeps), 1, npt_wing + 1)
    wing_joint_xyz_eff_xcomp = reshape(wing_xyz_eff[XDIM, :, :] + δ * local_chords_colmat .* cos.(localSweepEff), 1, npt_wing, npt_wing + 1)

    wing_joint_xyz_ycomp = reshape(wing_xyz[YDIM, :] + δ * local_chords .* sin.(localSweeps), 1, npt_wing + 1)
    wing_joint_xyz_eff_ycomp = reshape(transpose(wing_xyz[YDIM, :]) .+ δ * local_chords_colmat .* sin.(localSweepEff), 1, npt_wing, npt_wing + 1)

    wing_joint_xyz_zcomp = reshape(wing_xyz[ZDIM, :], 1, npt_wing + 1)
    wing_joint_xyz_eff_zcomp = reshape(wing_xyz_eff[ZDIM, :, :], 1, npt_wing, npt_wing + 1)

    wing_joint_xyz = cat(wing_joint_xyz_xcomp, wing_joint_xyz_ycomp, wing_joint_xyz_zcomp, dims=1)
    wing_joint_xyz_eff = cat(wing_joint_xyz_eff_xcomp, wing_joint_xyz_eff_ycomp, wing_joint_xyz_eff_zcomp, dims=1)

    # Store all computed quantities here
    LLMesh = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, wing_joint_xyz, npt_wing, local_chords, local_chords_ctrl, ζ, sectionLengths, sectionAreas,
        npt_airfoil, aeroWingSpan, SRef, SRef, AR, rootChord, sweepAng, rc, wing_xyz_eff, wing_joint_xyz_eff,
        localSweeps, localSweepEff, localSweepsCtrl)

    submergence = 20.0
    if !isnothing(options) && haskey(options, "depth")
        submergence = options["depth"]
    end

    FlowCond = FlowConditions(Uvec, Uinf, uvec, alpha, beta, rhof, submergence)

    return LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences
end

function compute_LAC(AR, LLHydro, y, c, cr, Λ, span; model="kuechemann")
    """
    Compute the locus of aerodynamic centers (LAC) for the wing

    Küchemann's 1956 method for the LAC of a constant swept wing with AR effects

    Parameters
    ----------
    y : spanwise coordinate [m]
    c : chord length at location y [m]
    cr : chord length at the root [m]
    Λ : global sweep of the wing [rad]
    span : full span of the wing [m]

    Returns
    -------
    x : location of the aerodynamic center at location y [m]
    """

    if model == "kuechemann"
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        # if Λ == 0
        #     fs1 = 0.25 * cr .- c * (1.0 - 1.0 / K) / 4.0
        #     fs = fs1
        # else
        tanl = vec(2π * tan(Λₖ) ./ (Λₖ * c))
        lam = .√(1.0 .+ (tanl .* y) .^ 2) .-
              tanl .* abs_cs_safe.(y) .-
              .√(1.0 .+ (tanl .* (0.5 * span .- abs_cs_safe.(y))) .^ 2) .+
              tanl .* (0.5 * span .- abs_cs_safe.(y))

        fs = 0.25 * cr .+
             tan(Λ) .* abs_cs_safe.(y) .-
             c .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25

        # as sweep goes to zero
        # fs → 0.25 * cr .-  c .* (1.0 .- (1.0 ) / K) * 0.25
        # end

        # Λtr = 0.0 # when to switch between models
        # λ = 1e-5 # offset to switch models (want close to zero)
        # sig = compute_sigmoid(Λ, Λtr, λ, 20)
        # fs = fs1 + (fs2 - fs1) * sig

    else
        println("Model not implemented yet")
    end

    return fs
end

function compute_LACeffective(AR, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
    """
    The effective LAC, based on Küchemann's equation .

    Parameters
    ----------
    y : spanwise coordinate
    y0 : control point location
    c : chord length at position y
    c_y0 : chord length at control point z0
    dc : change in chord length at location y, dc/dy
    dc_y0 : change in chord length at control point y0 , dc/dy
    σ : blend strength factor
    Λ : global sweep of the wing [rad]
    cr : chord length at the root
    span : full span of the wing
    model : LAC model to blend

    Returns
    -------
    x : location of the effective aerodynamic center at point y
    """

    # This is a matrix
    ywork = reshape(y, 1, size(y)...)
    y0work = reshape(y0, size(y0)..., 1)
    blend = exp.(-σ * (y0work .- ywork) .^ 2)

    if model == "kuechemann"

        LAC = compute_LAC(AR, LLHydro, ywork[1, :], c, cr, Λ, span)
        LACwork = reshape(LAC, 1, size(LAC)...)

        LAC0 = compute_LAC(AR, LLHydro, y0work[:, 1], c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, size(LAC0)..., 1)

        fprime0 = compute_dLACds(AR, LLHydro, y0work[:, 1], c_y0, dc_y0, Λ, span)

        LACeff = (1.0 .- blend) .* LACwork .+
                 blend .* (fprime0 .* (ywork .- y0work) .+ LAC0work)

        return LACeff
    else
        println("Model not implemented yet")
    end
end

function compute_dLACds(AR, LLHydro, y, c, ∂c∂y, Λ, span; model="kuechemann")
    """
    Compute the derivative of the LAC curve wrt the spanwise coordinate
    f'(s)
    Parameters
    ----------
    y : spanwise coordinate
    c : chord length at location y
    ∂c∂y : change in chord length at location y
    Λ : global sweep of the wing (rad)
    span : full span of the wing

    Returns
    -------
    dx : change in the location of the aerodynamic center at location y
    """

    if model == "kuechemann"
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        # if Λ == 0
        #     dx1 = -∂c∂y * (1.0 - 1.0 / K) * 0.25
        #     dx = dx1
        # else
        tanl = vec(2π * tan(Λₖ) ./ (Λₖ * c))
        lam = .√(1.0 .+ (tanl .* y) .^ 2) .-
              tanl .* abs_cs_safe.(y) .-
              .√(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2) .+
              tanl .* (span / 2.0 .- abs_cs_safe.(y))

        lamp = ((tanl .^ 2 .* (y .* c .- y .^ 2 .* ∂c∂y) ./ c) ./ .√(1.0 .+ (tanl .* y) .^ 2) -
                tanl .* (sign.(y) .* c .- abs_cs_safe.(y) .* ∂c∂y) ./ c +
                ((tanl .^ 2 .* (sign.(y) .* (span / 2.0 .- abs_cs_safe.(y)) .* c .+ ∂c∂y .* (span / 2.0 .- abs.(y)) .^ 2) ./ c) ./ .√(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2)) -
                tanl .* (sign.(y) .* c .+ (span / 2.0 .- abs_cs_safe.(y)) .* ∂c∂y) ./ c)

        dx = tan(Λ) * sign.(y) .+
             lamp * Λₖ .* c / (2π * K) .-
             ∂c∂y .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25

        # as sweep approaches zero
        # dx2 → ∂c∂y .* (1.0 .- (1.0) / K) * 0.25
        # which is the same as the above
        # end

        # Λtr = 0.0 # when to switch between models
        # λ = 1e-5 # offset to switch models (want close to zero)
        # sig = compute_sigmoid(Λ, Λtr, λ, 20)
        # dx = dx1 + (dx2 - dx1) * sig

    else
        println("Model not implemented yet")
    end

    return dx
end

function compute_dLACdseffective(AR, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
    """
    The derivative of the effective LAC , based on Kuchemann 's equation .

    Parameters
    ----------
    y : spanwise coordinate
    y0 : control point location
    c : chord length at position y
    c_y0 : chord length at control point z0
    dc : change in chord length at location y, dc/dy
    dc_y0 : change in chord length at control point y0 , dc/dy
    σ : blend strength factor
    Λ : global sweep of the wing [rad]
    cr : chord length at the root
    span : full span of the wing
    model : LAC model to blend

    Returns
    -------
    x : change in location of the effective aerodynamic center at point y
    """

    # This is a matrix
    ywork = reshape(y, 1, length(y))
    y0work = reshape(y0, length(y0), 1)
    blend = exp.(-σ * (y0work .- ywork) .^ 2)

    if model == "kuechemann"

        LAC = compute_LAC(AR, LLHydro, y, c, cr, Λ, span)
        LACwork = reshape(LAC, 1, length(LAC))
        fprime = compute_dLACds(AR, LLHydro, y, c, dc, Λ, span)
        fprimework = reshape(fprime, 1, length(fprime))
        LAC0 = compute_LAC(AR, LLHydro, y0, c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, length(LAC0), 1)
        fprime0 = compute_dLACds(AR, LLHydro, y0, c_y0, dc_y0, Λ, span)
        fprime0work = reshape(fprime0, length(fprime0), 1)

        return fprimework .+
               blend .*
               (fprime0work .- fprimework .-
                2 * σ * (y0work .- ywork) .*
                (fprime0work .* (y0work .- ywork) .-
                 (LAC0work .- LACwork)))
    else
        println("Model not implemented yet")
    end
end

function solve(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=true)
    """
    Execute LL algorithm.
    Top level wrapper to interface with. 
    Taking derivatives is trickier and done analytically

    Inputs:
    -------
    LiftingSystem : LiftingLineSystem
        Lifting line system struct with all necessary parameters

    LLHydro : LiftingLineHydro
        Section properties at the root airfoil
    Returns:
    --------
    LLResults : LiftingLineResults
        Lifting line results struct with all necessary parameters
    """

    # --- Unpack data structs ---
    Uinf = FlowCond.Uinf
    α = FlowCond.alpha
    β = FlowCond.beta
    rhof = FlowCond.rhof
    DimForces, Γdist, ∂cl∂α, cl, IntegratedForces, CL, CDi, CS = compute_solution(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=is_verbose)

    # --- Pack back up  ---
    LLResults = LiftingLineOutputs(DimForces, Γdist, ∂cl∂α, cl, IntegratedForces, CL, CDi, CS)

    return LLResults
end

function compute_solution(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=true)

    ∂α = FlowCond.alpha + Δα # FD

    ∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
    ∂Uinf = norm_cs_safe(∂Uinfvec)
    ∂uvec = ∂Uinfvec / FlowCond.Uinf
    ∂FlowCond = FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof, FlowCond.depth)

    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    TV_influence = compute_TVinfluences(FlowCond, LLMesh)

    ∂TV_influence = compute_TVinfluences(∂FlowCond, LLMesh)

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess using root properties
    c_r = LLMesh.rootChord
    clα = LLHydro.airfoil_CLa
    αL0 = LLHydro.airfoil_aL0
    Λ = LLMesh.sweepAng
    # Ux, _, Uz = FlowCond.Uinfvec
    ux, uy, uz = FlowCond.uvec
    ∂ux, ∂uy, ∂uz = ∂FlowCond.uvec
    span = LLMesh.span
    ctrl_pts = LLMesh.collocationPts
    ζi = LLMesh.sectionVectors
    dAi = reshape(LLMesh.sectionAreas, 1, size(LLMesh.sectionAreas)...)
    g0 = 0.5 * c_r * clα * cos(Λ) *
         (uz / ux - αL0) *
         ((1.0 .- (2.0 * ctrl_pts[YDIM, :] / span) .^ 4) .^ 2) .^ (0.5)


    # --- Pack up parameters for the NL solve ---
    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    # --- Nonlinear solve for circulation distribution ---
    Gconv, _, _ = do_newton_raphson(compute_LLresiduals, compute_LLresJacobian, g0, nothing;
        solverParams=LLNLParams, is_verbose=is_verbose,
        mode="FiDi" # this is the fastest
    )
    ∂Gconv, _, _ = do_newton_raphson(compute_LLresiduals, compute_LLresJacobian, g0, nothing;
        solverParams=∂LLNLParams, is_verbose=is_verbose,
        mode="FiDi" # this is the fastest
    )

    DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, LLNLParams)

    # --- Compute the lift curve slope ---
    # ∂G∂α = imag(∂Gconv) / Δα # CS
    ∂G∂α = (∂Gconv .- Gconv) / Δα # Forward Difference
    ∂cl∂α = 2 * ∂G∂α ./ LLMesh.localChordsCtrl

    return DimForces, Γdist, ∂cl∂α, clvec, IntegratedForces, CL, CDi, CS

end

function compute_liftslopes(Gconv::AbstractVector, ∂Gconv::AbstractVector, LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences, appendageOptions, solverOptions)
    """
    Compute the lift curve slope of the wing at the converged solution
    """

    # ************************************************
    #     Method 1
    # ************************************************
    # --- Compute the lift curve slope ---
    ∂G∂α = (∂Gconv .- Gconv) / Δα # Forward Difference
    ∂cl∂α = 2 * ∂G∂α ./ LLMesh.localChordsCtrl

    # # TODO: perturb beta as well to get the lift curve slope wrt to sideslip angle for the strut sections

    # # ************************************************
    # #     Method 2
    # # ************************************************
    # ∂α = FlowCond.alpha + Δα # FD

    # ∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
    # ∂Uinf = norm_cs_safe(∂Uinfvec)
    # ∂uvec = ∂Uinfvec / FlowCond.Uinf
    # ∂FlowCond = FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof, FlowCond.depth)

    # # ---------------------------
    # #   Calculate influence matrix
    # # ---------------------------
    # ∂TV_influence = compute_TVinfluences(∂FlowCond, LLMesh)

    # # ---------------------------
    # #   Solve for circulation
    # # ---------------------------
    # # --- Pack up parameters for the NL solve ---
    # ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    # # --- Nonlinear solve for circulation distribution ---
    # ∂Gconv, ∂residuals = do_newton_raphson(compute_LLresiduals, compute_LLresJacobian, Gconv, nothing;
    #     solverParams=∂LLNLParams, is_verbose=false,
    #     appendageOptions=appendageOptions, solverOptions=solverOptions,
    #     mode="FiDi"  # this is the fastest
    # )
    # # --- Compute the lift curve slope ---
    # ∂G∂α = (∂Gconv .- Gconv) / Δα # Forward Difference
    # ∂cl∂α = 2 * ∂G∂α ./ LLMesh.localChordsCtrl # this thing changes when you change the mesh

    return ∂cl∂α
end

function compute_dcladXpt(Gconv_i, Gconv_f, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FiDi")
    """
    Derivative of the lift slope wrt the design variables
    """

    dcldX_f = zeros(npt_wing, length(ptVec))
    dcldX_i = zeros(npt_wing, length(ptVec))
    appendageParams_da = copy(appendageParams)
    appendageParams_da["alfa0"] = appendageParams["alfa0"] + Δα

    if uppercase(mode) == "FIDI" # very different from what adjoint gives. I think it's because of edge nodes as those show the highest discrepancy in the derivatives

        dh = 1e-7
        idh = 1 / dh

        # ************************************************
        #     First time with current angle of attack
        # ************************************************
        LLOutputs_i, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)
        f_i = LLOutputs_i.cl
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

            f_f = LLOutputs_f.cl

            dcldX_i[:, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end

        # writedlm("dcldX_i-$(mode).csv", dcldX_i, ',')

        # ************************************************
        #     Second time with perturbed angle of attack
        # ************************************************

        LLOutputs_i, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions; return_all=true)
        f_i = LLOutputs_i.cl
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions; return_all=true)
            f_f = LLOutputs_f.cl
            dcldX_f[:, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end
        # writedlm("dcldX_f-$(mode).csv", dcldX_f, ',')


    elseif uppercase(mode) == "IMPLICIT"
        function compute_directMatrix(∂r∂u, ∂r∂xPt)

            Φ = ∂r∂u \ ∂r∂xPt
            return Φ
        end
        # ************************************************
        #     First time with current converged solution
        # ************************************************

        # println("∂r∂Γ") # 2 sec, so it's fast
        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv_i, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        # println("∂r∂Xpt") # ACCELERATE THIS!!?
        # Takes about 4 sec in pure julia and about 20sec from python
        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv_i, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions;
            # mode="FAD",
            mode="FiDi", # fastest
        )

        ∂cl∂Γ = diagm(2 * LLOutputs_i.cl ./ LLOutputs_i.Γdist)
        ∂cl∂X = zeros(npt_wing, length(ptVec)) # There's no dependence

        Φ = compute_directMatrix(∂r∂Γ, ∂r∂xPt)
        dcldX_i = ∂cl∂X - ∂cl∂Γ * Φ


        # ************************************************
        #     Second time with perturbed angle of attack
        # ************************************************

        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv_f, ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions)

        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv_f, ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions)
        ∂cl∂Γ = diagm(2 * LLOutputs_f.cl ./ LLOutputs_f.Γdist)

        Φ = compute_directMatrix(∂r∂Γ, ∂r∂xPt)
        dcldX_f = ∂cl∂X - ∂cl∂Γ * Φ

    end

    dcladXpt = (dcldX_f - dcldX_i) ./ Δα

    return dcladXpt
end

function compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, LLNLParams)
    """
    """

    Gi = reshape(Gconv, 1, size(Gconv)...) # now it's a (1, npt) matrix
    Gjvji = TV_influence .* Gi
    Gjvjix = TV_influence[XDIM, :, :] * Gconv
    Gjvjiy = TV_influence[YDIM, :, :] * Gconv
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    Gjvjiz = -TV_influence[ZDIM, :, :] * Gconv
    Gjvji = cat(Gjvjix, Gjvjiy, Gjvjiz, dims=2)
    Gjvji = permutedims(Gjvji, [2, 1])
    u∞ = repeat(reshape(FlowCond.uvec, 3, 1), 1, LLMesh.npt_wing)

    ui = Gjvji .+ u∞ # Local velocities (nondimensional)

    ζi = LLMesh.sectionVectors
    dAi = reshape(LLMesh.sectionAreas, 1, size(LLMesh.sectionAreas)...)
    ux, uy, uz = FlowCond.uvec

    # This is the Biot--Savart law but nondimensional
    # fi = 2 | ( ui ) × ζi| Gi dAi / SRef
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    # uicrossζi = -cross.(eachcol(ui), eachcol(ζi))
    # uicrossζi = hcat(uicrossζi...) # now it's a (3, npt) matrix

    uicrossζi_z = Zygote.Buffer(zeros(Number, 3, LLMesh.npt_wing))
    for ii in 1:LLMesh.npt_wing
        uicrossζi_z[:, ii] = -myCrossProd(ui[:, ii], ζi[:, ii])
    end
    uicrossζi = copy(uicrossζi_z)

    coeff = 2.0 / LLMesh.SRef
    NondimForces = coeff * (uicrossζi .* Gi) .* dAi

    # Integrated = 2 Σ ( u∞ + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
    IntegratedNondimForces = vec(coeff * sum((uicrossζi .* Gi) .* dAi, dims=2))
    # These integrated forces are about the origin

    Γdist = Gconv * FlowCond.Uinf # dimensionalize the circulation distribution
    # Forces = NondimForces .* LLMesh.SRef * 0.5 * ϱ * FlowCond.Uinf^2 # dimensionalize the forces
    # println(Γdist)
    cmvec = compute_cm_LE(Gconv; solverParams=LLNLParams)

    # --- Dimensional forces ---
    Γi = Gi * FlowCond.Uinf
    Γjvji = TV_influence .* Γi
    Γjvjix = TV_influence[XDIM, :, :] * Γdist
    Γjvjiy = TV_influence[YDIM, :, :] * Γdist
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    Γjvjiz = -TV_influence[ZDIM, :, :] * Γdist
    Γjvji = cat(Γjvjix, Γjvjiy, Γjvjiz, dims=2)
    Γjvji = permutedims(Γjvji, [2, 1])
    U∞ = repeat(reshape(FlowCond.Uinfvec, 3, 1), 1, LLMesh.npt_wing)

    Ui = Γjvji .+ U∞ # Local velocities
    Uicrossdli = -cross.(eachcol(Ui), eachcol(ζi))
    Uicrossdli = hcat(Uicrossdli...) # now it's a (3, npt) matrix
    DimForces = FlowCond.rhof * (Uicrossdli .* Γi) .* dAi


    # --- Vortex core viscous correction ---
    if LLMesh.rc != 0
        println("Vortex core viscous correction not implemented yet")
    end

    # --- Final outputs ---
    # NOTE: the X force is the "drive force" in the chordwise direction
    CL = -IntegratedNondimForces[XDIM] * uz +
         IntegratedNondimForces[ZDIM] * ux / (ux^2 + uz^2)

    CDi = IntegratedNondimForces[XDIM] * ux +
          IntegratedNondimForces[YDIM] * uy +
          IntegratedNondimForces[ZDIM] * uz
    CS = (
        -IntegratedNondimForces[XDIM] * ux * uy -
        IntegratedNondimForces[ZDIM] * uz * uy +
        IntegratedNondimForces[YDIM] * (uz^2 + ux^2)
    ) / √(ux^2 * uy^2 + uz^2 * uy^2 + (uz^2 + ux^2)^2)

    # --- Compute the spanwise lift coefficients ---
    clvec = 2 * Gconv ./ LLMesh.localChordsCtrl

    # This would be the integrated forces in the x, y, z, directions, but we need drag
    IntegratedDimForcesXYZ = vec(sum(DimForces, dims=2))
    IntegratedDimForces = LLMesh.SRef * 0.5 * FlowCond.rhof * FlowCond.Uinf^2 * [CDi, CS, CL]   # [induced drag, sideforce, lift]

    return DimForces, Γdist, clvec, cmvec, IntegratedDimForces, CL, CDi, CS
end

function compute_TVinfluences(FlowCond, LLMesh)
    """
    Outputs
    -------
    TV_influence : array_like (3, npt_wing, npt_wing)
        Influence matrix for the lifting line system
    """
    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    uinf = reshape(FlowCond.uvec, 3, 1, 1)
    uinfMat = repeat(uinf, 1, LLMesh.npt_wing, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    P1 = LLMesh.wing_joint_xyz_eff[:, :, 2:end]
    P2 = LLMesh.wing_xyz_eff[:, :, 2:end]
    P3 = LLMesh.wing_xyz_eff[:, :, 1:end-1]
    P4 = LLMesh.wing_joint_xyz_eff[:, :, 1:end-1]

    ctrlPts = reshape(LLMesh.collocationPts, size(LLMesh.collocationPts)..., 1)
    ctrlPtMat = repeat(ctrlPts, 1, 1, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    # Mask for the bound segment (npt_wing x npt_wing)
    # This is a matrix of ones with a main diagonal of zeros
    bound_mask = ones(LLMesh.npt_wing, LLMesh.npt_wing) - diagm(ones(LLMesh.npt_wing))


    # --- TODO: these routines will eventually need to be generalized to work with dihedral wings without relying on a small dz approximation ---
    influence_semiinfa = compute_straightSemiinfinite(P1, uinfMat, ctrlPtMat, LLMesh.rc)
    influence_straightsega = compute_straightSegment(P1, P2, ctrlPtMat, LLMesh.rc)
    influence_straightsegb = compute_straightSegment(P2, P3, ctrlPtMat, LLMesh.rc) .* reshape(bound_mask, 1, size(bound_mask)...)
    influence_straightsegc = compute_straightSegment(P3, P4, ctrlPtMat, LLMesh.rc)
    influence_semiinfb = compute_straightSemiinfinite(P4, uinfMat, ctrlPtMat, LLMesh.rc)


    TV_influence = -influence_semiinfa +
                   influence_straightsega +
                   influence_straightsegb +
                   influence_straightsegc +
                   influence_semiinfb
    return TV_influence
end

function compute_LLresiduals(G; solverParams=nothing)
    """
    Nonlinear , nondimensional lifting - line equation .
    Parameters
    ----------
    G : vector
    Circulation distribution normalized by the freestream velocity
    magnitude.

    Returns
    -------
    R : array_like
    Array of the residuals between the lift values predicted from
    section properties and from circulation.
    """

    if isnothing(solverParams)
        println("WARNING: YOU NEED TO PASS IN SOLVER PARAMETERS")
    end

    TV_influence = solverParams.TV_influence
    LLSystem = solverParams.LLSystem
    Airfoils = solverParams.Airfoils
    AirfoilInfluences = solverParams.AirfoilInfluences
    FlowCond = solverParams.FlowCond
    ζi = LLSystem.sectionVectors


    # This is a (3 , npt, npt) × (npt,) multiplication
    # PYTHON: _Vi = TV_influence * G .+ transpose(LLSystem.uvec)
    uix = TV_influence[XDIM, :, :] * G .+ FlowCond.uvec[XDIM]
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    uiy = TV_influence[YDIM, :, :] * G .+ FlowCond.uvec[YDIM]
    uiz = -TV_influence[ZDIM, :, :] * G .+ FlowCond.uvec[ZDIM]
    ui = cat(uix, uiy, uiz, dims=2)
    ui = permutedims(ui, [2, 1])


    # Do a curve fit on aero props
    # if self._aero_approx:
    # _CL = self._lift_from_aero(*self._aero_properties, self.local_sweep_ctrl, self.Vinf * _Vi, self.Vinf)
    # else:
    # Actually solve VPM for each local velocity c
    Ui = FlowCond.Uinf * (ui) # dimensionalize the local velocities

    hcRatio = FlowCond.depth ./ LLSystem.localChordsCtrl

    c_l::Vector{Number} = [
        solve_VPM(Airfoils[ii], AirfoilInfluences[ii], V_local, 1.0, FlowCond.Uinf, hcRatio[ii])[1]
        for (ii, V_local) in enumerate(eachcol(Ui))
    ] # remember to only grab CL out of VPM solve

    ui_cross_ζi = cross.(eachcol(ui), eachcol(ζi)) # this gives a vector of vectors, not a matrix, so we need double indexing --> [][]
    ui_cross_ζi = hcat(ui_cross_ζi...) # now it's a (3, npt) matrix
    ui_cross_ζi_mag = .√(ui_cross_ζi[XDIM, :] .^ 2 + ui_cross_ζi[YDIM, :] .^ 2 + ui_cross_ζi[ZDIM, :] .^ 2)


    dFimag = 2.0 * ui_cross_ζi_mag .* G

    return dFimag - c_l
end

function compute_LLresJacobian(Gi; solverParams, mode="Analytic")
    """
    Compute the Jacobian of the nonlinear, nondimensional lifting line equation

    Inputs:
    -------
    Gi - Circulation distribution normalized by freestream velocity Γ / Uinf

    Returns:
    --------
    J - Jacobian matrix, matrix of partial derivatives ∂r/∂G for
        r(G) = Nondim LL eqn

    """

    if uppercase(mode) == "ANALYTIC" # After many hours of debugging, it matches Python but still doesn't converge...robustness issue

        TV_influence = solverParams.TV_influence
        LLSystem = solverParams.LLSystem
        LLHydro = solverParams.LLHydro
        # Airfoils = solverParams.Airfoils
        # AirfoilInfluences = solverParams.AirfoilInfluences
        FlowCond = solverParams.FlowCond
        # ζi = LLSystem.sectionVectors
        vji = TV_influence

        # (u∞ + Σ Gj vji)
        uix = -vji[XDIM, :, :] * Gi .+ FlowCond.uvec[XDIM] # negated...
        uiy = -vji[YDIM, :, :] * Gi .+ FlowCond.uvec[YDIM] # negated...
        uiz = -vji[ZDIM, :, :] * Gi .+ FlowCond.uvec[ZDIM] # negated...


        ui = cat(uix, uiy, uiz, dims=2)
        ui = permutedims(ui, [2, 1])

        ζ = LLSystem.sectionVectors
        # 3d array of ζ
        ζArr = repeat(reshape(ζ, size(ζ)..., 1), 1, 1, size(ζ, 2))
        #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
        uxy = -cross.(eachcol(ui), eachcol(ζ))
        uxy = hcat(uxy...) # now it's a (3, npt) matrix
        uxy_norm = .√(uxy[XDIM, :] .^ 2 + uxy[YDIM, :] .^ 2 + uxy[ZDIM, :] .^ 2)

        vxy = cross3D(vji, ζArr)

        # This is downwash contribution
        uxyvxy_xcomp = uxy[XDIM, :] .* vxy[XDIM, :, :]
        uxyvxy_ycomp = uxy[YDIM, :] .* vxy[YDIM, :, :]
        uxyvxy_zcomp = uxy[ZDIM, :] .* vxy[ZDIM, :, :]
        uxzdotvxz = uxyvxy_xcomp .+ uxyvxy_ycomp .+ uxyvxy_zcomp
        numerator = 2.0 * uxzdotvxz .* Gi
        J = numerator ./ uxy_norm .+ 2.0 * diagm(uxy_norm)

        # Along span
        Λ = LLSystem.local_sweeps_ctrl

        _Cs = cos.(Λ)
        _Ss = sin.(Λ)
        αs = atan_cs_safe.(uiz, uix)
        βs = atan_cs_safe.(uiy, uix)
        _aL = atan_cs_safe.(uiz, uix .* _Cs .+ uiy .* _Ss) # GOOD
        _aLMat = reshape(_aL, size(_aL)..., 1)
        _bL = βs .- Λ

        uixMat = reshape(uix, size(uix)..., 1)
        uiyMat = reshape(uiy, size(uiy)..., 1)
        uizMat = reshape(uiz, size(uiz)..., 1)
        _CsMat = reshape(_Cs, size(_Cs)..., 1)
        _SsMat = reshape(_Ss, size(_Ss)..., 1)
        uixviz = uixMat .* (-vji[ZDIM, :, :])
        uizvix = -(uizMat .* vji[XDIM, :, :])
        num_da = uixviz .- uizvix
        denom_da = uixMat .^ 2 .+ uizMat .^ 2
        _da = num_da ./ denom_da

        uixvy = uixMat .* (-vji[YDIM, :, :])
        uiyvx = -uiyMat .* vji[XDIM, :, :]
        num_db = uixvy .- uiyvx
        denom_db = uixMat .^ 2 + uiyMat .^ 2
        _db = num_db ./ denom_db

        uixcos = uixMat .* _CsMat
        uiysin = uiyMat .* _SsMat
        firstTerm_daL = (uixcos .+ uiysin) .* (-vji[ZDIM, :, :])
        uizvixcos = uizvix .* _CsMat
        uizviysin = uizMat .* (-vji[YDIM, :, :]) .* _SsMat
        secondTerm_daL = uizvixcos .+ uizviysin
        denom_daL = uixMat .^ 2 .+ (uixcos .+ uiysin) .^ 2
        _daL = (firstTerm_daL .- secondTerm_daL) ./ denom_daL # OK

        _Ca = cos.(αs)
        _Sa = sin.(αs)
        _Cb = cos.(βs)
        _Sb = sin.(βs)
        SaSquared = _Sa .^ 2
        SbSquared = _Sb .^ 2
        _CaL = cos.(_aL)
        _SaL = sin.(_aL)
        SaLSquared = _SaL .^ 2
        _CbL = cos.(_bL)
        _SbL = sin.(_bL)
        SbLSquared = _SbL .^ 2
        _Rn = .√(_Ca .^ 2 .* _CbL .^ 2 .+ SaSquared .* _Cb .^ 2)
        _Rd = .√(1.0 .- _Sa .^ 2 .* SbSquared)
        iRd = 1.0 ./ _Rd
        iRdSquared = iRd .^ 2
        _RLd = .√(1.0 .- SaLSquared .* SbLSquared)
        _R = _Rn .* iRd
        _RMat = reshape(_R, size(_R)..., 1)
        _RL = _CbL ./ _RLd
        _RLMat = reshape(_RL, size(_RL)..., 1)

        firstTermdR = reshape((_Sa .* _Ca .* (SbSquared .* _Rn .* iRdSquared .+ (_Cb .^ 2 .- _CbL .^ 2) ./ _Rn) ./ _Rd), size(_Sa)..., 1)
        secondTermdR = reshape((SaSquared .* _Sb .* _Cb .* _Rn .* iRdSquared .- (_Ca .^ 2 .* _SbL .* _CbL .+ SaSquared .* _Sb .* _Cb) ./ _Rn) .* iRd, size(_Sa)..., 1)

        _dR = firstTermdR .* _da .+ secondTermdR .* _db # OK

        firstTerm_dRL = reshape(_SaL .* _CaL .* _SbL .^ 2 .* _CbL ./ (_RLd .^ 3), size(_SaL)..., 1)
        secondTerm_dRL = reshape(_CaL .^ 2 .* _SbL ./ (_RLd .^ 3), size(_SaL)..., 1)

        _dRL = firstTerm_dRL .* _daL .- secondTerm_dRL .* _db # OK
        HydroProps = [compute_hydroProperties(sweep, LLHydro.airfoil_xy, LLHydro.airfoil_ctrl_xy)[1] for sweep in Λ]
        _CLa = [HydroProps[ii].airfoil_CLa for (ii, _) in enumerate(Λ)]
        _aL0 = [HydroProps[ii].airfoil_aL0 for (ii, _) in enumerate(Λ)]
        _CLaMat = reshape(_CLa, size(_CLa)..., 1)
        _aL0Mat = reshape(_aL0, size(_aL0)..., 1)

        _dCL = _dR .* _RLMat .* _CLaMat .* (_aLMat .- _aL0Mat) .+ _RMat .* _dRL .* _CLaMat .* (_aLMat .- _aL0Mat) .+ _RMat .* _RLMat .* _CLaMat .* _daL

        J = J .- _dCL
        # println("J: $(J[end,:])") # OK
        # println("\ndCL:")
        # show(stdout, "text/plain", J)
        # println(forceerror)
    elseif mode == "CS" # slow as hell but works
        dh = 1e-100
        ∂r∂G = zeros(DTYPE, length(Gi), length(Gi))

        GiCS = complex(copy(Gi))
        for ii in eachindex(Gi)
            GiCS[ii] += 1im * dh
            r_f = compute_LLresiduals(GiCS; solverParams=solverParams)
            GiCS[ii] -= 1im * dh
            ∂r∂G[:, ii] = imag(r_f) / dh
        end
        J = ∂r∂G
    elseif mode == "FiDi" # currently the best

        # backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
        # J, = AD.jacobian(backend, x -> compute_LLresiduals(x; solverParams=solverParams), Gi)

        dh = 1e-4
        ∂r∂G = zeros(DTYPE, length(Gi), length(Gi))
        ∂r∂G_z = Zygote.Buffer(∂r∂G)
        r_i = compute_LLresiduals(Gi; solverParams=solverParams)
        for ii in eachindex(Gi)
            ChainRulesCore.ignore_derivatives(Gi[ii] += dh)
            r_f = compute_LLresiduals(Gi; solverParams=solverParams)
            ChainRulesCore.ignore_derivatives(Gi[ii] -= dh)
            # ∂r∂G[:, ii] = (r_f - r_i) / dh
            ∂r∂G_z[:, ii] = (r_f - r_i) / dh
            # println("r:")
            # println(r_f)
            # println("r_i:")
            # println(r_i)
        end
        # J = ∂r∂G
        J = copy(∂r∂G_z)
    elseif mode == "RAD" # not working

        backend = AD.ZygoteBackend()
        J, = AD.jacobian(backend, x -> compute_LLresiduals(x; solverParams=solverParams), Gi)

    else
        println("Mode not implemented yet")
    end

    return J
end

function setup_solverparams(xPt, nodeConn, idxTip, displCol, appendageOptions, appendageParams, solverOptions)
    """
    This is a convenience function that sets up the solver parameters for the lifting line algorithm from xPt
    """

    LECoords, TECoords = repack_coords(xPt, 3, length(xPt) ÷ 3)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]

    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLSystem, FlowCond, LLHydro, Airfoils, AirfoilInfluences = setup(Uvec, sweepAng, rootChord, TR, midchords, displCol, pretwistDist;
        npt_wing=size(displCol, 2),
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=@ignore_derivatives(options),
    )

    TV_influence = compute_TVinfluences(FlowCond, LLSystem)

    # --- Pack up parameters for the NL solve ---
    solverParams = LiftingLineNLParams(TV_influence, LLSystem, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    return solverParams, FlowCond
end

function compute_∂I∂G(Gconv, LLMesh, FlowCond, LLNLParams, solverOptions, appendageParams; mode="FAD")

    NFORCES = 3
    NFORCECOEFFS = 3
    NQUANTS = 1
    outputVector = zeros(NFORCES + NFORCECOEFFS + 1 + 3 * NPT_WING * NQUANTS + NPT_WING)
    ∂I∂G = zeros(DTYPE, length(outputVector), length(Gconv))

    function compute_outputsFromGConv(Gconv)
        TV_influence = compute_TVinfluences(FlowCond, LLMesh)
        DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, LLNLParams)

        depth = appendageParams["depth0"] # FlowCond.depth is wrong
        Fnh = FlowCond.Uinf / √(GRAV * depth) # depth froude number TODO: make this a vectorized calculation
        clvent_incep = compute_cl_ventilation(Fnh, FlowCond.rhof, FlowCond.Uinf, PVAP)

        ventilationConstraint = clvec .- clvent_incep # subtract the ventilation inception lift coefficient, this must be less then zero!

        ksvent = compute_KS(ventilationConstraint, solverOptions["rhoKS"])

        # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
        outputvector = vcat(IntegratedForces[XDIM], IntegratedForces[YDIM], IntegratedForces[ZDIM], CL, CDi, CS, ksvent, vec(transpose(DimForces)), clvec)

        return outputvector
    end

    if uppercase(mode) == "FAD"
        # backend = AD.ReverseDiffBackend()
        backend = AD.ForwardDiffBackend()
        ∂I∂G, = AD.jacobian(backend, x -> compute_outputsFromGConv(x), Gconv)

    elseif uppercase(mode) == "FIDI"
        # Compares well with finite difference 
        backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
        ∂I∂G, = AD.jacobian(backend, x -> compute_outputsFromGConv(x), Gconv)
    elseif uppercase(mode) == "RAD" # with the number of outputs, this is not really worth it. Untested
        backend = AD.ReverseDiffBackend()
        backend = AD.ZygoteDiffBackend()
        ∂I∂G, = AD.jacobian(backend, x -> compute_outputsFromGConv(x), Gconv)
    end

    return ∂I∂G
end

function compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    LECoords, _ = repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = get_tipnode(LECoords)
    solverParams, FlowCond = setup_solverparams(ptVec, nodeConn, idxTip, appendageOptions, appendageParams, solverOptions)

    ∂r∂G = LiftingLine.compute_LLresJacobian(Gconv; solverParams=solverParams, mode="CS")
    ∂r∂Γ = ∂r∂G / FlowCond.Uinf

    return ∂r∂Γ
end

function compute_∂r∂Xpt(Gconv, ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")


    LECoords, _ = repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = get_tipnode(LECoords)

    function compute_resFromXpt(xPt, xDisplCol::AbstractVector)

        displCol_in = reshape(xDisplCol, size(displCol)...)

        solverParams, _ = setup_solverparams(xPt, nodeConn, idxTip, displCol_in, appendageOptions, appendageParams, solverOptions)

        resVec = compute_LLresiduals(Gconv; solverParams=solverParams)

        return resVec
    end

    displVec = vec(displCol)

    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        ∂r∂Xpt = zeros(DTYPE, length(Gconv), length(ptVec))
        ∂r∂Xdispl = zeros(DTYPE, length(Gconv), length(displCol))
        # dh = 1e-5
        dh = 1e-4 # standard
        # dh = 1e-3
        # dh = 1e-2

        resVec_i = compute_resFromXpt(ptVec, displVec) # initialize the solver

        # backend = AD.FiniteDifferencesBackend(central_fdm(3, 1))
        # ∂r∂Xpt, = AD.jacobian(backend, x -> compute_resFromXpt(x, displVec), ptVec)

        # @inbounds begin # no speedup
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            resVec_f = compute_resFromXpt(ptVec, displVec)

            ∂r∂Xpt[:, ii] = (resVec_f - resVec_i) / dh

            ptVec[ii] -= dh
        end

        for ii in eachindex(displVec)
            displVec[ii] += dh

            resVec_f = compute_resFromXpt(ptVec, displVec)

            displVec[ii] -= dh

            ∂r∂Xdispl[:, ii] = (resVec_f - resVec_i) / dh

        end
        # end
    elseif uppercase(mode) == "CS" # does not work
        dh = 1e-100

        ptVecCS = complex(copy(ptVec))

        for ii in eachindex(ptVec)
            ptVecCS[ii] += 1im * dh
            resVec_f = compute_resFromXpt(ptVecCS, displVec)
            ptVecCS[ii] -= 1im * dh
            ∂r∂Xpt[:, ii] = imag(resVec_f) / dh
        end
    elseif uppercase(mode) == "RAD" # It's broken

        backend = AD.ReverseDiffBackend() # stack overflow errors?
        # backend = AD.ZygoteBackend() # Broken and does not work without buffering over VPM solve
        ∂r∂Xpt, = AD.jacobian(backend, x -> compute_resFromXpt(x), ptVec)

    elseif uppercase(mode) == "FAD"

        backend = AD.ForwardDiffBackend()
        # ∂r∂Xpt, = AD.jacobian(backend, (xPt, xDisplCol) -> compute_resFromXpt(xPt, xDisplCol), ptVec, vec(displCol))
        ∂r∂Xpt = ForwardDiff.jacobian((xPt) -> compute_resFromXpt(xPt, displVec), ptVec)
        ∂r∂Xdispl = ForwardDiff.jacobian((xDisplCol) -> compute_resFromXpt(ptVec, xDisplCol), displVec)

    end

    return ∂r∂Xpt, ∂r∂Xdispl
end

function compute_∂I∂Xpt(Gconv::AbstractVector, ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    """
    Compute cost function Jacobian
    """

    NFORCES = 3
    NFORCECOEFFS = 3
    NQUANTS = 1
    npt_wing = size(displCol, 2)
    outputVector = zeros(NFORCES + NFORCECOEFFS + 1 + 3 * npt_wing * NQUANTS + npt_wing)
    LECoords, _ = repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = get_tipnode(LECoords)

    function compute_OutputFromXpt(xPt, xDisplCol::AbstractVector)

        displCol_in = transpose(reshape(xDisplCol, length(xDisplCol) ÷ 6, 6)) # this is the correct order

        solverParams, FlowCond = setup_solverparams(xPt, nodeConn, idxTip, displCol_in, appendageOptions, appendageParams, solverOptions)

        TV_influence = solverParams.TV_influence
        LLMesh = solverParams.LLSystem

        DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, solverParams)

        depth = appendageParams["depth0"] # FlowCond.depth is wrong
        Fnh = FlowCond.Uinf / √(GRAV * depth) # depth froude number TODO: make this a vectorized calculation
        clvent_incep = compute_cl_ventilation(Fnh, FlowCond.rhof, FlowCond.Uinf, PVAP)

        ventilationConstraint = clvec .- clvent_incep # subtract the ventilation inception lift coefficient, this must be less then zero!

        ksvent = compute_KS(ventilationConstraint, solverOptions["rhoKS"])

        # THIS ORDER MATTER. Check CostFuncsInOrder variable
        # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
        outputVector = vcat(IntegratedForces[XDIM], IntegratedForces[YDIM], IntegratedForces[ZDIM], CL, CDi, CS, ksvent, vec(transpose(DimForces)), clvec)

        return outputVector
    end

    # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
    # displCol is of shape (6, NPT_WING)
    # We need to make sure it is ordered such that we loop over NPT_WING first, then the 6 elements
    displVec = vec(transpose(displCol))

    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        ∂I∂Xpt = zeros(DTYPE, length(outputVector), length(ptVec))
        ∂I∂Xdispl = zeros(DTYPE, length(outputVector), length(displCol))
        dh = 1e-4

        f_i = compute_OutputFromXpt(ptVec, displVec)

        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            f_f = compute_OutputFromXpt(ptVec, displVec)

            ptVec[ii] -= dh

            ∂I∂Xpt[:, ii] = (f_f - f_i) / dh
        end
        for ii in eachindex(displVec)
            displVec[ii] += dh

            f_f = compute_OutputFromXpt(ptVec, displVec)

            displVec[ii] -= dh

            ∂I∂Xdispl[:, ii] = (f_f - f_i) / dh
        end
    elseif uppercase(mode) == "CS" # broken right now

        dh = 1e-100

        ptVecCS = complex(copy(ptVec))
        for ii in eachindex(ptVec)
            ptVecCS[ii] += 1im * dh

            f_f = compute_OutputFromXpt(ptVecCS)

            ptVecCS[ii] -= 1im * dh

            ∂I∂Xpt[:, ii] = imag(f_f) / dh
        end

    elseif uppercase(mode) == "RAD" # not working
        backend = AD.ReverseDiffBackend()
        # backend = AD.ZygoteBackend()
        ∂I∂Xpt, = AD.jacobian(backend, x -> compute_OutputFromXpt(x), ptVec)
        println("shape", size(∂I∂Xpt))
    elseif uppercase(mode) == "FAD"
        backend = AD.ForwardDiffBackend()


        ∂I∂Xpt, = AD.jacobian(backend, xPt -> compute_OutputFromXpt(xPt, displVec), ptVec)
        ∂I∂Xdispl, = AD.jacobian(backend, xDispl -> compute_OutputFromXpt(ptVec, xDispl), displVec)
    end

    # println("writing ∂I∂Xpt-$(mode).csv")
    # for ii in eachindex(outputVector)
    #     
    #     writedlm("∂I∂Xpt-$(ii)-$(mode).csv", ∂I∂Xpt[ii, :], ",")
    # end

    return ∂I∂Xpt, ∂I∂Xdispl
end

function compute_∂collocationPt∂Xpt(ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FAD")

    npt_wing = size(displCol, 2)
    ∂collocationPt∂Xpt = zeros(DTYPE, npt_wing * 3, length(ptVec))

    LECoords, _ = repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = get_tipnode(LECoords)

    function compute_collocationFromXpt(xPt)
        solverParams, _ = setup_solverparams(xPt, nodeConn, idxTip, displCol, appendageOptions, appendageParams, solverOptions)

        # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
        outputVec = vec(transpose(solverParams.LLSystem.collocationPts))

        return outputVec
    end

    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        dh = 1e-4 # do not use smaller finite difference steps

        resVec_i = compute_collocationFromXpt(ptVec) # initialize the solver

        # @inbounds begin # no speedup
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            resVec_f = compute_collocationFromXpt(ptVec)

            ptVec[ii] -= dh

            ∂collocationPt∂Xpt[:, ii] = (resVec_f - resVec_i) / dh
        end
        # end
    elseif uppercase(mode) == "CS"
        dh = 1e-100

        ptVecCS = complex(copy(ptVec))

        for ii in eachindex(ptVec)
            ptVecCS[ii] += 1im * dh
            resVec_f = compute_collocationFromXpt(ptVecCS)
            ptVecCS[ii] -= 1im * dh
            ∂collocationPt∂Xpt[:, ii] = imag(resVec_f) / dh
        end
    elseif uppercase(mode) == "RAD" # This takes nearly 15 seconds compared to a few sec in pure Fidi julia
        # backend = AD.ReverseDiffBackend()
        backend = AD.ZygoteBackend()
        ∂collocationPt∂Xpt, = AD.jacobian(backend, x -> compute_collocationFromXpt(x), ptVec)

    elseif uppercase(mode) == "FAD" # use this, same speed as Fidi
        backend = AD.ForwardDiffBackend()
        ∂collocationPt∂Xpt, = AD.jacobian(backend, x -> compute_collocationFromXpt(x), ptVec)

    end

    return ∂collocationPt∂Xpt
end

function compute_∂collocationPt∂displCol(ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FAD")

    npt_wing = size(displCol, 2)
    ∂collocationPt∂displCol = zeros(DTYPE, npt_wing * 3, length(displCol))

    LECoords, _ = repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = get_tipnode(LECoords)

    function compute_collocationFromdisplCol(xDispl)

        xdisplCol = transpose(reshape(xDispl, length(xDispl) ÷ 6, 6)) # this is the correct order
        # println("xdisplCol: $(xdisplCol)")
        solverParams, _ = setup_solverparams(ptVec, nodeConn, idxTip, xdisplCol, appendageOptions, appendageParams, solverOptions)

        outputVec = vec(transpose(solverParams.LLSystem.collocationPts))

        return outputVec
    end

    # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
    # displCol is of shape (6, NPT_WING)
    # We need to make sure it is ordered such that we loop over NPT_WING first, then the 6 elements
    displVec = vec(transpose(displCol))

    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        dh = 1e-4 # do not use smaller finite difference steps

        resVec_i = compute_collocationFromdisplCol(displVec) # initialize the solver

        # @inbounds begin # no speedup
        for ii in eachindex(displVec)

            # only do it for the first 3 elements
            # if mod(ii, 6) in [1, 2, 3]

            displVec[ii] += dh
            # println("displCol perturbed:")
            # show(stdout, "text/plain", transpose(reshape(displVec, length(displVec) ÷ 6, 6)))
            # println("")


            resVec_f = compute_collocationFromdisplCol(displVec)

            displVec[ii] -= dh

            ∂collocationPt∂displCol[:, ii] = (resVec_f - resVec_i) / dh
            # println("column of jacobian:")
            # println((resVec_f - resVec_i) / dh)
            # end

        end
        # end
    elseif uppercase(mode) == "RAD" # This takes nearly 15 seconds compared to a few sec in pure Fidi julia
        # backend = AD.ReverseDiffBackend()
        backend = AD.ZygoteBackend()
        ∂collocationPt∂displCol, = AD.jacobian(backend, x -> compute_collocationFromdisplCol(x), displVec)

    elseif uppercase(mode) == "FAD" # use this, same speed as Fidi
        backend = AD.ForwardDiffBackend()
        ∂collocationPt∂displCol, = AD.jacobian(backend, x -> compute_collocationFromdisplCol(x), displVec)

    elseif uppercase(mode) == "ANALYTIC"
        # for ii in 1:npt_wing*3
        #     ∂collocationPt∂displCol[ii, ii] = 1.0
        # end
        println("WARNING: ignoring dihedral effect in collocation point Jacobian (i.e., z deriv)")
        for ii in 1:npt_wing*2
            ∂collocationPt∂displCol[ii, ii] = 1.0
        end
    end

    return ∂collocationPt∂displCol
end

function compute_straightSemiinfinite(startpt, endvec, pt, rc)
    """
    Compute the influence of a straight semi-infinite vortex filament
    Inputs:
    -------
    startpt : ndarray
        Starting point of the semi-infinite vortex filament
        This should be on a panel edge
    endvec : Array{Float64, 3}
        Unit vector of the semi-infinite vortex filament
    pt : ndarray
        Point at which the influence is computed (field point).
        This should match where the collocation points are
    rc : scalar
        Vortex core radius (viscous correction)

    Returns:
    --------
    influence : ndarray
        Influence of the semi-infinite vortex filament at the field point. This is everything but the Γ in the induced velocity equation.
    """


    r1 = pt .- startpt # this is an array of shape (3, NPT_WING, NPT_WING) describing the vectors from the semi-inf vortex filament to the field points
    # println("r1")
    # show(stdout, "text/plain", r1[ZDIM, :, :]) # wrong
    # println("")
    # # --- Hack it so the z distances are zero (should generalize in the future) ---
    # # The assumption here is that collocation points and horseshoe vortices are all at the same 'z', which is only valid for small amounts of dihedral
    # r1[ZDIM, :, :] .= 0.0

    r1mag = .√(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    uinf = endvec

    r1dotuinf = r1[XDIM, :, :] .* uinf[XDIM, :, :] .+
                r1[YDIM, :, :] .* uinf[YDIM, :, :] .+
                r1[ZDIM, :, :] .* uinf[ZDIM, :, :]

    r1crossuinf = cross3D(r1, uinf)
    uinfcrossr1 = cross3D(uinf, r1)

    d = .√(r1crossuinf[XDIM, :, :] .^ 2 + r1crossuinf[YDIM, :, :] .^ 2 + r1crossuinf[ZDIM, :, :] .^ 2)
    d = ifelse.(real(r1dotuinf) .< 0.0, r1mag, d)

    # Reshape d to have a singleton dimension for correct broadcasting
    d = reshape(d, 1, size(d)...)

    numerator = uinfcrossr1 .*
                (d .^ 2 ./ .√(rc^4 .+ d .^ 4))

    denominator = (4π * r1mag .* (r1mag .- r1dotuinf))
    denominator = reshape(denominator, 1, size(denominator)...)

    influence = numerator ./ denominator

    # Replace NaNs and Infs with 0.0
    ChainRulesCore.ignore_derivatives() do
        influence = replace(influence, NaN => 0.0)
        influence = replace(influence, Inf => 0.0)
        influence = replace(influence, -Inf => 0.0)
    end


    return influence
end

function compute_straightSegment(startpt, endpt, pt, rc)
    """
    Compute the influence of a straight vortex filament segment on a point.

    Parameters
    ----------
    startpt : Array{Float64,3}
        The position vector of the beginning point of the vortex segment ,
        in three dimensions .
    endpt : Array{Float64,3}
        The position vector of the end point of the vortex segment ,
        in three dimensions .
    pt : Array{Float64,3}
        The position vector of the point at which the influence of the
        vortex segment is calculated , in three dimensions .
    rc : Float64
        The radius of the vortex finite core .
    Returns
    -------
    influence : Array{Float64,3}
        The influence of vortex segment at the point, in three dimensions .
    """

    r1 = pt .- startpt
    r1mag = .√(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    r2 = pt .- endpt

    # # --- Hack it so the z distances are zero (should generalize in the future) ---
    # # The assumption here is that collocation points and horseshoe vortices are all at the same 'z', which is only valid for small amounts of dihedral
    # r1[ZDIM, :, :] .= 0.0
    # r2[ZDIM, :, :] .= 0.0

    r2mag = .√(r2[XDIM, :, :] .^ 2 + r2[YDIM, :, :] .^ 2 + r2[ZDIM, :, :] .^ 2)
    r1r2 = r1 .- r2

    r1r2mag = .√(r1r2[XDIM, :, :] .^ 2 + r1r2[YDIM, :, :] .^ 2 + r1r2[ZDIM, :, :] .^ 2)

    r1dotr2 = r1[XDIM, :, :] .* r2[XDIM, :, :] + r1[YDIM, :, :] .* r2[YDIM, :, :] + r1[ZDIM, :, :] .* r2[ZDIM, :, :]
    r1dotr1r2 = r1[XDIM, :, :] .* r1r2[XDIM, :, :] + r1[YDIM, :, :] .* r1r2[YDIM, :, :] + r1[ZDIM, :, :] .* r1r2[ZDIM, :, :]
    r2dotr1r2 = r2[XDIM, :, :] .* r1r2[XDIM, :, :] + r2[YDIM, :, :] .* r1r2[YDIM, :, :] + r2[ZDIM, :, :] .* r1r2[ZDIM, :, :]

    r1crossr2 = cross3D(r1, r2)

    d = (r1crossr2[XDIM, :, :] .^ 2 + r1crossr2[YDIM, :, :] .^ 2 + r1crossr2[ZDIM, :, :] .^ 2) ./ r1r2mag
    d = ifelse.(r1dotr1r2 .< 0.0, r1mag, d)
    d = ifelse.(r2dotr1r2 .< 0.0, r2mag, d)

    # Reshape d to have a singleton dimension for correct broadcasting
    d = reshape(d, 1, size(d)...)

    numerator = reshape(r1mag .+ r2mag, 1, size(r1mag)...) .* r1crossr2
    numerator = numerator .* (d .^ 2) ./ .√(rc^4 .+ d .^ 4)
    denominator = (4π * r1mag .* r2mag .* (r1mag .* r2mag .+ r1dotr2))

    # Reshape the denominator to have the same dimensions as the influence    
    denominator = reshape(denominator, 1, size(denominator)...)

    influence = numerator ./ denominator

    # Replace NaNs and Infs with 0.0
    # Cannot keep in ignore derivatives block
    ChainRulesCore.ignore_derivatives() do
        influence = replace(influence, NaN => 0.0)
        influence = replace(influence, Inf => 0.0)
        influence = replace(influence, -Inf => 0.0)
    end

    return influence
end

function compute_hydroProperties(Λ, airfoil_xy_orig, airfoil_ctrl_xy_orig)
    """
    Determines the aerodynamic properties of a swept airfoil 

    Parameters
    ----------
    Λ : scalar , optional
        The local sweep angle of the effective airfoil [rad]
    airfoil_xy_orig : array_like
        The original airfoil coordinates
    airfoil_ctrl_xy_orig : array_like
        The original airfoil control points
    Returns
    -------
    CLa : scalar
        Effective lift slope of the swept airfoil [1/ rad].
    aL0 : scalar
        Effective zero - lift angle of attack of the swept airfoil [rad].
    """


    angles = [deg2rad(-5), 0.0, deg2rad(5)] # three angles to average properties over...
    V1 = compute_vectorFromAngles(angles[1], 0.0, 1.0)
    V2 = compute_vectorFromAngles(angles[2], 0.0, 1.0)
    V3 = compute_vectorFromAngles(angles[3], 0.0, 1.0)

    # println("airfoil x orig:", airfoil_xy_orig[XDIM, :])
    # println("airfoil y orig:", airfoil_xy_orig[YDIM, :])
    airfoil_xy, airfoil_ctrl_xy = compute_scaledAndSweptAirfoilCoords(Λ, airfoil_xy_orig, airfoil_ctrl_xy_orig)

    # --- VPM of airfoil ---
    Airfoil, Airfoil_influences = setup_VPM(airfoil_xy[XDIM, :], airfoil_xy[YDIM, :], airfoil_ctrl_xy, 0.0) # setup with no sweep
    # println("airfoil x:", airfoil_xy[XDIM, :]) #close enough
    # println("airfoil y:", airfoil_xy[YDIM, :]) #close enough
    _, cm1, Γ1, _ = solve_VPM(Airfoil, Airfoil_influences, V1)
    _, cm2, Γ2, _ = solve_VPM(Airfoil, Airfoil_influences, V2)
    _, cm3, Γ3, _ = solve_VPM(Airfoil, Airfoil_influences, V3)
    Γairfoils = [Γ1 Γ2 Γ3]
    Γbar = (Γ1 + Γ2 + Γ3) / 3.0

    airfoil_Γa = (angles[1] * (Γairfoils[1] - Γbar) +
                  angles[2] * (Γairfoils[2] - Γbar) +
                  angles[3] * (Γairfoils[3] - Γbar)) /
                 (angles[1]^2 + angles[2]^2 + angles[3]^2) # this should not be 0.0
    # println("Angles: $(angles)") # correct
    # println("Vectors: $(V1) $(V2) $(V3)")
    # println("Circulation values: $(Γairfoils) ") # close enough
    # println("Γbar: $(Γbar)") # close enough
    # println("Γa: $(airfoil_Γa)") #  close enough
    # println("cm1: $(cm1) cm2: $(cm2) cm3: $(cm3)")

    airfoil_aL0 = -Γbar / airfoil_Γa
    airfoil_CLa = 2.0 * airfoil_Γa / cos(Λ)

    LLHydro = LiftingLineHydro(airfoil_CLa, airfoil_aL0, airfoil_xy, airfoil_ctrl_xy)

    return LLHydro, Airfoil, Airfoil_influences
end

function compute_cm_LE(G; solverParams=nothing)
    """
    Nonlinear , nondimensional lifting - line equation .
    Parameters
    ----------
    G : vector
    Circulation distribution normalized by the freestream velocity
    magnitude.

    Returns
    -------
    R : array_like
    Array of the residuals between the lift values predicted from
    section properties and from circulation.
    """

    if isnothing(solverParams)
        println("WARNING: YOU NEED TO PASS IN SOLVER PARAMETERS")
    end

    TV_influence = solverParams.TV_influence
    LLSystem = solverParams.LLSystem
    Airfoils = solverParams.Airfoils
    AirfoilInfluences = solverParams.AirfoilInfluences
    FlowCond = solverParams.FlowCond
    ζi = LLSystem.sectionVectors


    # This is a (3 , npt, npt) × (npt,) multiplication
    # PYTHON: _Vi = TV_influence * G .+ transpose(LLSystem.uvec)
    uix = TV_influence[XDIM, :, :] * G .+ FlowCond.uvec[XDIM]
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    uiy = TV_influence[YDIM, :, :] * G .+ FlowCond.uvec[YDIM]
    uiz = -TV_influence[ZDIM, :, :] * G .+ FlowCond.uvec[ZDIM]
    ui = cat(uix, uiy, uiz, dims=2)
    ui = permutedims(ui, [2, 1])


    # Actually solve VPM for each local velocity c
    Ui = FlowCond.Uinf * (ui) # dimensionalize the local velocities

    hcRatio = FlowCond.depth ./ LLSystem.localChordsCtrl


    # c_m::AbstractVector{Number} = [
    #     solve_VPM(Airfoils[ii], AirfoilInfluences[ii], V_local, 1.0, FlowCond.Uinf, hcRatio[ii])[2]
    #     for (ii, V_local) in enumerate(eachcol(Ui))
    # ] # remember to only grab CM out of VPM solve

    c_m_z = Zygote.Buffer(zeros(size(Airfoils)))
    c_m_z = [
        solve_VPM(Airfoils[ii], AirfoilInfluences[ii], V_local, 1.0, FlowCond.Uinf, hcRatio[ii])[2]
        for (ii, V_local) in enumerate(eachcol(Ui))
    ] # remember to only grab CM out of VPM solve
    c_m = copy(c_m_z)

    return c_m
end

function compute_scaledAndSweptAirfoilCoords(Λ, airfoil_xy, airfoil_ctrl_xy, factor=1.0)
    """
    The effective swept airfoil geometry.
    Parameters
    ----------
    Λ : scalar , optional
        The sweep angle at which the effective airfoil coordinates will be
        calculated ( rad ).
    factor : scalar , optional
        The scaling by which the coordinates are multiplied to match the
        desired chord length .
    airfoil_xy : array_like
        The original airfoil coordinates
    airfoil_ctrl_xy : array_like
        The original airfoil control points
    """
    cosΛ = cos(Λ)

    return factor * transpose(hcat(airfoil_xy[XDIM, :] * cosΛ, airfoil_xy[YDIM, :])),
    factor * transpose(hcat(airfoil_ctrl_xy[XDIM, :] * cosΛ, airfoil_ctrl_xy[YDIM, :]))
end

end