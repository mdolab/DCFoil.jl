# --- Julia 1.7---
"""
@File    :   geo.jl
@Time    :   2023/03/16
@Author  :   Galen Ng
@Desc    :   Wrapper to pygeo and pyspline
"""

using PyCall

pyspline = pyimport("pyspline")
pygeo = pyimport("pygeo")