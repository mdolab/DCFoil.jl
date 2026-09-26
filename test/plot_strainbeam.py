"""
@File    :   plot_strainbeam.py
@Time    :   2026/09/24
@Author  :   Galen Ng
@Desc    :   Spanwise figure of the StrainBeam CFRP NACA0009 cantilever from the CSV tables written by
             test_StrainBeam_NACA0009 (OUTPUT/strainbeam/). Fallback for environments where Plots.jl
             does not load. Usage: python3 test/plot_strainbeam.py [outputDir] [figurePath]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import niceplots

    plt.style.use(niceplots.get_style())
except Exception:
    pass


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(v) for v in row] for row in rows[1:]])
    return {name: data[:, j] for j, name in enumerate(header)}


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "OUTPUT", "strainbeam")
    figure_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo, "docs", "figures", "strainbeam_naca0009_spanwise.pdf")

    nodal_nl = read_csv(os.path.join(output_dir, "naca0009_nodal_nonlinear.csv"))
    nodal_lin = read_csv(os.path.join(output_dir, "naca0009_nodal_linear.csv"))
    elem_nl = read_csv(os.path.join(output_dir, "naca0009_elements_nonlinear.csv"))
    elem_lin = read_csv(os.path.join(output_dir, "naca0009_elements_linear.csv"))
    history = read_csv(os.path.join(output_dir, "naca0009_load_history.csv"))

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0, 0]
    for k, (key, label) in enumerate([("u1", "$u_1$ axial"), ("u2", "$u_2$ chordwise (to LE)"), ("u3", "$u_3$ flap")]):
        ax.plot(nodal_nl["s"], nodal_nl[key], color=colors[k], lw=2, label=f"{label} nonlinear")
        ax.plot(nodal_lin["s"], nodal_lin[key], color=colors[k], ls="--", label=f"{label} linear")
    ax.set_xlabel("span $s$ [m]")
    ax.set_ylabel("displacement [m]")
    ax.set_title("Deflections in the beam frame")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(nodal_nl["s"], np.degrees(nodal_nl["twist"]), color=colors[0], lw=2, label="nonlinear")
    ax.plot(nodal_lin["s"], np.degrees(nodal_lin["twist"]), color=colors[0], ls="--", label="linear")
    ax.set_xlabel("span $s$ [m]")
    ax.set_ylabel("twist [deg]")
    ax.set_title("Twist about the deformed axis (nose-down negative)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for k, key in enumerate(["kappa1", "kappa2", "kappa3"]):
        ax.plot(elem_nl["s"], elem_nl[key], color=colors[k], lw=2, label=rf"$\kappa_{k + 1}$ nonlinear")
        ax.plot(elem_lin["s"], elem_lin[key], color=colors[k], ls="--", label=rf"$\kappa_{k + 1}$ linear")
    ax.set_xlabel("span $s$ [m]")
    ax.set_ylabel("curvature [1/m]")
    ax.set_title("Curvatures")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(elem_nl["s"], elem_nl["gamma11"], color="k", lw=1, label=r"$\gamma_{11}$ nonlinear")
    ax2.set_ylabel(r"$\gamma_{11}$")
    ax2.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    for k, (key, label) in enumerate([("F1", "$F_1$"), ("F2", "$F_2$"), ("F3", "$F_3$"), ("M1", "$M_1$"), ("M2", "$M_2$"), ("M3", "$M_3$")]):
        ax.plot(elem_nl["s"], elem_nl[key], color=colors[k % len(colors)], lw=2, label=f"{label} nonlinear")
        ax.plot(elem_lin["s"], elem_lin[key], color=colors[k % len(colors)], ls="--", label=f"{label} linear")
    ax.set_xlabel("span $s$ [m]")
    ax.set_ylabel("internal load [N, N m]")
    ax.set_title("Internal loads in the deformed frame")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 0]
    ax.plot(elem_nl["s"], elem_nl["bimoment"], color=colors[0], lw=2, label=r"$M_\omega$ nonlinear")
    ax.set_xlabel("span $s$ [m]")
    ax.set_ylabel(r"bimoment $M_\omega$ [N m$^2$]")
    ax.set_title("Vlasov bimoment and twist rate")
    ax2 = ax.twinx()
    ax2.plot(nodal_nl["s"], nodal_nl["twistRate"], color=colors[1], lw=1.5, label=r"$\kappa_1$ (twist rate)")
    ax2.set_ylabel(r"$\kappa_1$ [1/m]")
    ax.legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")

    ax = axes[2, 1]
    ax.plot(history["loadFactor"], history["tipU3Nonlinear"], "o-", color=colors[0], lw=2, label="tip $u_3$ nonlinear")
    ax.plot(history["loadFactor"], history["tipU3Linear"], "--", color=colors[0], label="tip $u_3$ linear")
    ax.set_xlabel("load factor")
    ax.set_ylabel("tip flap deflection [m]")
    ax.set_title("Tip response along the load path")
    ax2 = ax.twinx()
    ax2.plot(history["loadFactor"], np.degrees(history["tipTwistNonlinear"]), color=colors[2], lw=1.5, label="tip twist nonlinear")
    ax2.set_ylabel("tip twist [deg]")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("CFRP NACA0009 cantilever, chord 0.1 m, span 1 m, +15° fibers, upward tip force 300 N", y=0.995)
    fig.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path)
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
