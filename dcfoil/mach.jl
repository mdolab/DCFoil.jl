# --- Julia 1.7---
"""
@File          :   mach.jl
@Date created  :   2024/09/04
@Last modified :   2024/09/04
@Author        :   Galen Ng
@Desc          :   Interface to necessary python packages.
                   Be careful with calling this script because it creates the dependency
"""


using PyCall

PYSPLINE = pyimport("pyspline")
PYGEO = pyimport("pygeo")
PREFOIL = pyimport("prefoil")