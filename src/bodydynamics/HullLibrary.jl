# --- Julia 1.9---
"""
@File    :   HullLibrary.jl
@Time    :   2024/03/19
@Author  :   Galen Ng
@Desc    :   Store mass properties of hulls
"""


module HullLibrary

# --- Public functions ---
export return_hullprop

function return_hullprop(hullName::String)
    """
    SI units
    """
    mass::Float64 = 0.0
    length::Float64 = 0.0
    beam::Float64 = 0.0
    if (hullName == "moth")
        # https://sail1design.com/moth/#:~:text=Key%20Facts%3A,Unrestricted%20(~%2035%2D40%20Kg)
        mass = 40.0
        length = 3.3
        beam = 2.3
        # https://www.boatdesign.net/threads/moth-plans.9693/
        xcg = 1.405

        kxz = -0.274 #m
        ixz = kxz^2 * mass
    end
    Ixx, Iyy, Izz = estimate_massmoments(mass, length, beam)
    Ib = zeros(Float64, 3, 3)
    Ib[1, 1] = Ixx
    Ib[2, 2] = Iyy
    Ib[3, 3] = Izz

    return mass, length, beam, Ib
end

function estimate_massmoments(mass::Float64, loa::Float64, beam::Float64)
    """
    Use ITTC approximations to get mass and inertia properties
    These are mainly regressions for displacement hulls from section 2.3 Model Mass Properties in Seakeeping Experiments
        https://www.ittc.info/media/9705/75-02-07-021.pdf
    Actually Eggert has these radii of gyration for a moth
    """
    kxx = 0.3 * beam
    kyy = 0.25 * loa

    Ixx = kxx^2 * mass
    Iyy = kyy^2 * mass
    Izz = Iyy # assume yaw and pitch are the same
    return Ixx, Iyy, Izz
end

end