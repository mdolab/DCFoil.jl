# --- Julia 1.7---
"""
@File          :   mach.jl
@Date created  :   2024/09/04
@Last modified :   2024/09/04
@Author        :   Galen Ng
@Desc          :   Interface to necessary python packages
"""


using PyCall

pyspline = pyimport("pyspline")
pygeo = pyimport("pygeo")
prefoil = pyimport("prefoil")