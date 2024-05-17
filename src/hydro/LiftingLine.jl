# --- Julia 1.9 ---
"""
@File    :   LiftingLine.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   Modern lifting line from Phillips and Snyder 2000, Reid 2020 appendix
The major weakness is the discontinuity in the locus of aerodynamic centers
for a highly swept wing at the root.
    Reid 2020 overcame this
"""

module LiftingLine

struct VortexElement{TF}
    alphaGeo::TF
    # collocationPts::
end


end