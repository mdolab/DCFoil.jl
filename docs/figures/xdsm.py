# --- Python 3.10 ---
"""
@File          :   xdsm.py
@Date created  :   2025/01/16
@Last modified :   2025/01/16
@Author        :   Galen Ng
@Desc          :   XDSM diagram
"""

from pyxdsm.XDSM import XDSM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem", action="store_true", default=False)
    args = parser.parse_args()
    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    if args.subsystem:
        opt = "Optimization"
        solver = "MDA"
        func = "Function"

        x = XDSM()

        # --- create systems ---
        # x.add_system("opt", opt, r"\text{Optimizer}")
        # x.add_system("geo", func, (r"\text{Geometry}", r"\text{parametrization}"))
        x.add_system("solver", solver, (r"\text{Static hydroelastic}", r"\text{MDA}"))
        x.add_system("febeam", func, (r"\text{Composite beam}", r"\text{solver}"))
        x.add_system("liftingline", func, (r"\text{Lifting line}", r"\text{solver}"))
        x.add_system("dynamicsolver", solver, (r"\text{Dynamic}", r"\text{solvers}"),stack=True)

        # --- draw data connection ---
        x.connect("febeam", "liftingline", r"\text{Displacements}")
        x.connect("solver","febeam",  r"\text{Surface loads}")
        x.connect("febeam","solver",  r"\mathbf{r}_s(\mathbf{u})")
        x.connect("liftingline","solver",  r"\mathbf{r}_f(\boldsymbol{\gamma})")
        x.connect("solver","dynamicsolver",  (r"\text{Static states and linearized quantities }",r"\text{about static equilibrium for dynamics analysis:}",r"c_{\ell_\alpha}'s, \mathbf{M}_{ss}, \mathbf{C}_{ss}, \mathbf{K}_{ss}, \text{etc.}"))
        # x.connect("opt", "geo", (r"\text{Geometric}", r"\text{variables}"))
        # x.connect("opt", "solver", (r"\text{Flow \&}", r"\text{structural}", r"\text{variables}"))

        # --- draw process connection ---
        x.add_process(["solver", "febeam"], arrow=True)
        x.add_process(["febeam", "liftingline"], arrow=True)
        x.add_process(["liftingline", "solver"], arrow=True)
        x.add_process(["solver", "dynamicsolver"], arrow=True)

        x.write("statichydroelastic")

    else:
        opt = "Optimization"
        solver = "MDA"
        func = "Function"

        x = XDSM()

        # --- create systems ---
        x.add_system("pre", func, r"\text{Pre-processing}")
        x.add_system("opt", opt, r"\text{Optimizer}")
        x.add_system("geo", func, (r"\text{Geometry}", r"\text{parametrization}"))
        x.add_system("solver", solver, (r"\text{Static hydroelastic}", r"\text{MDA}"))
        x.add_system("forcedsolver", solver, (r"\text{Forced vibration}", r"\text{MDA}"))
        x.add_system("fluttersolver", solver, (r"\text{Flutter}", r"\text{MDA}"))
        x.add_system("CAD", func, (r"\text{Adjoint}", r"\text{solver}"))
        x.add_system("RAD", func, (r"\text{Reverse AD}"))
        x.add_system("F", func, (r"\text{Objective \&}", r"\text{constraint}", r"\text{evaluation}"))
        x.add_system("G", func, (r"\text{Objective \&}", r"\text{constraint}", r"\text{derivatives}"))

        # # --- draw data connection ---
        x.connect("pre", "geo", r"\text{FFD points}")
        x.connect("solver", "CAD", r"\text{Static states}")
        x.connect("fluttersolver", "RAD", r"\text{Modal states}")
        x.connect("forcedsolver", "RAD", r"\text{Dynamic states}")
        x.connect("opt", "geo", (r"\text{Geometric}", r"\text{variables}"))
        x.connect("opt", "solver", (r"\text{Flow \&}", r"\text{structural}", r"\text{variables}"))
        x.connect("opt", "forcedsolver", (r"\text{Flow \&}", r"\text{structural}", r"\text{variables}"))
        x.connect("opt", "fluttersolver", (r"\text{Flow \&}", r"\text{structural}", r"\text{variables}"))
        x.connect("solver", "F", (r"\text{Static hydroelastic}", r"\text{functions}"))
        x.connect("fluttersolver", "F", (r"\text{Flutter}", r"\text{functions}"))
        x.connect("forcedsolver", "F", (r"\text{Forced vibration}", r"\text{functions}"))
        x.connect("CAD", "G", (r"\text{Static hydroelastic}", r"\text{function derivatives}"))
        x.connect("RAD", "G", (r"\text{Dynamic hydroelastic}", r"\text{function derivatives}"))
        x.connect("F", "opt", (r"\text{Objectives}", r"\text{\& constraints}"))
        x.connect("G", "opt", (r"\text{Derivatives of}", r"\text{objectives}", r"\text{\& constraints}"))

        # --- draw process connection ---
        x.add_process(["opt", "geo", "solver", "F"], arrow=True)
        x.add_process(["opt", "geo", "fluttersolver", "F"], arrow=True)
        x.add_process(["opt", "geo", "forcedsolver", "F"], arrow=True)
        x.add_process(["solver", "solver"], arrow=True)
        x.add_process(["solver", "CAD", "G"], arrow=True)
        x.add_process(["fluttersolver", "RAD", "G"], arrow=True)
        x.add_process(["forcedsolver", "RAD", "G"], arrow=True)
        x.add_output("opt", (r"\text{Optimal}", r"\text{design}"), side="left")
        x.add_output("solver", (r"\text{Final}", r"\text{states}"), side="left")
        x.add_output("forcedsolver", (r"\text{Final dynamic}", r"\text{states}"), side="left")
        x.add_output("fluttersolver", (r"\text{Flutter}", r"\text{mode evolution}"), side="left")


        x.write("dcfoil")
