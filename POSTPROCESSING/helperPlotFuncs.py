# --- Python 3.9 ---
"""
@File    :   helperPlotFuncs.py
@Time    :   2023/02/03
@Author  :   Galen Ng
@Desc    :   Contains functions for plotting flutter and divergence points. 
             Snippets from Dr. Eirikur Jonsson
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tecplot as tp

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots as nplt
from helperFuncs import get_bendingtwisting, compute_normFactorModeShape

# ==============================================================================
#                         GLOBAL VARIABLES
# ==============================================================================
# Continuous colormap
niceColors = sns.color_palette("cool")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
ccm = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ==============================================================================
#                         FUNCTIONS
# ==============================================================================
def plot_wingPlanform(DVDict: dict, nNodes, cm):
    """
    Use design var dictionary to plot the wing

    Parameters
    ----------
    DVDict : dict
        Contains all design variables
    """
    # Create figure object
    labelpad = 30
    legfs = 15
    nrows = 1
    ncols = 1
    figsize = (9 * ncols, 5 * nrows)
    moldColor = "black"
    alpha = 0.5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, constrained_layout=True, figsize=figsize)

    y = np.linspace(0, DVDict["s"], nNodes)

    # ax = axes[0]
    # ************************************************
    #     Plot planform
    # ************************************************
    # --- Plot outer mold shape ---
    ax.plot(y, -0.5 * np.array(DVDict["c"]), color=moldColor)
    ax.plot(y, 0.5 * np.array(DVDict["c"]), color=moldColor)
    ax.plot([y[-1], y[-1]], [0.5 * DVDict["c"][-1], -0.5 * DVDict["c"][-1]], color=moldColor)

    # --- Plot elastic axis (E.A.) ---
    ab = -np.array(DVDict["ab"])
    ax.plot(y, ab, color=cm[0], ls="--", alpha=alpha)
    ax.annotate(
        f"E.A.",
        xy=(y[nNodes // 2], ab[nNodes // 2]),
        color=cm[0],
        fontsize=legfs,
        xytext=(10, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
        arrowprops=dict(
            facecolor=cm[0],
            edgecolor=None,
            shrink=0.05,
            headwidth=8,
            width=2,
        ),
    )

    # --- Plot static imbalance arm from E.A. ---
    xalpha = -np.array(DVDict["x_ab"]) - np.array(DVDict["ab"])
    ax.plot(y, xalpha, color=cm[1], ls="-.", alpha=alpha)
    ax.annotate(
        "C.G.",
        xy=(y[nNodes // 3], xalpha[nNodes // 3]),
        color=cm[1],
        fontsize=legfs,
        xytext=(10, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
        arrowprops=dict(
            facecolor=cm[1],
            edgecolor=None,
            shrink=0.05,
            headwidth=8,
            width=2,
        ),
    )

    # --- Plot evaluation nodes ---
    ax.scatter(y, np.zeros_like(y), color=moldColor, marker="o", alpha=alpha)

    ax.set_xticks([y[0], y[-1]])
    ax.set_yticks([-0.5 * DVDict["c"][0], 0.0, 0.5 * DVDict["c"][0]])
    ax.set_xlim([y[0], y[-1] * 1.25])
    ax.set_ylim([-0.5 * DVDict["c"][0] * 1.5, 0.5 * DVDict["c"][0] * 1.5])
    ax.set_xlabel("$y$ [m]")
    ax.set_ylabel("$x$ [m]", labelpad=labelpad, rotation=0, ha="right")
    ax.set_aspect("equal")

    ax.set_title("Wing planform")

    # --- Text annotations ---
    geomText = f"$\\alpha_0$\n$\\Lambda$\n$\\theta_f$\n$t/c$\nnNode\nnDOF"
    try:
        toc = DVDict["toc"][0] * 100
    except:
        toc = DVDict["toc"] * 100
    valText = (
        f"{DVDict['α₀']}"
        + "$^{{\\circ}}$\n"
        + f"{DVDict['Λ']*180/np.pi:.1f}"
        + "$^{{\\circ}}$\n"
        + f"{DVDict['theta_f']*180/np.pi:.1f}"
        + "$^{{\\circ}}$\n"
        + f"{toc:0.1f}%\n"
        + f"{nNodes}\n"
        + f"{nNodes*3}"
    )
    ax.annotate(
        geomText,
        xy=(0.85, 0.95),
        xycoords="axes fraction",
        fontsize=legfs,
        va="top",
        bbox=dict(boxstyle="round", ec="white", fc="white", alpha=0.8),
    )
    ax.annotate(
        valText,
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=legfs * 1.034,  # I have no idea why they won't just line up
        va="top",
    )

    plt.tight_layout()

    # for ax in axes.flatten():
    nplt.adjust_spines(ax, outward=True)

    return fig, ax


def plot_deformedShape(fig, ax, DVDict: dict, bending, twisting):
    """
    Creates a 3D surface visualization of the deformed wing with nodes as dots

    Parameters
    ----------
    ax : _type_
        _description_
    DVDict : dict
        _description_
    u : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # ************************************************
    #     Helper functions
    # ************************************************
    def get_LETE(twist: float, semichord: float):
        """
        Returns LE and TE height deltas from midchord

          /|
         / |
        /  | angle
        -------

        """
        le_dz = semichord * np.sin(twist)
        le_dx = semichord - semichord * np.cos(twist)
        te_dz = -le_dz
        te_dx = le_dx
        return le_dx, le_dz, te_dx, te_dz

    # ************************************************
    #     Determine points to plot
    # ************************************************
    semispan = DVDict["s"]
    semichords = np.divide(DVDict["c"], 2)
    nNodes = len(bending)

    # Make dummy Z value first
    X = np.linspace(-1, 1, 2)
    Y = np.linspace(0, semispan, nNodes)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(X**2, Y**2)
    for ii, twist in enumerate(twisting):
        le_dx, le_dz, te_dx, te_dz = get_LETE(twist, semichords[ii])
        Z[ii, :] = bending[ii] * le_dz / semispan + le_dx

    from matplotlib import cm

    # --- Plot the surface in 3D ---
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.coolwarm,
        antialiased=False,
        # vmin=-bendLim,
        # vmax=bendLim,
        alpha=0.8,
        zorder=1,
    )

    # --- Set labels and axes ---
    ax.set_xlabel(r"$x$", labelpad=30)
    ax.set_ylabel(r"$y$", labelpad=20, va="center")
    ax.set_zlabel(r"$w(y)$ [in]", labelpad=20)
    ax.view_init(20, 25)  # [elevtion, azimuthal]
    cbar = fig.colorbar(surf, shrink=0.5, aspect=7)
    cbar.ax.set_ylabel(r"$w(y)$ [in]", rotation=360, va="center")
    cbar.ax.get_yaxis().labelpad = 25
    ax.tick_params(pad=10)

    return fig, ax


def plot_static2d(
    fig,
    axes,
    nodes,
    bending,
    twisting,
    spanLift,
    spanMoment,
    funcs,
    label: str,
    lc,
    fs_lgd: float,
    iic: int,
    solverOptions=None,
    do_annotate=False,
    ls="-",
):
    lpad = 40
    set_my_plot_settings(True)

    if solverOptions is not None:
        nnodewing = solverOptions["nNodes"]
        bending = bending[:nnodewing]
        twisting = twisting[:nnodewing]
        spanLift = spanLift[:nnodewing]
        spanMoment = spanMoment[:nnodewing]
    else:
        nodes
    cl = funcs["cl"]
    lift = funcs["lift"]
    mom = funcs["moment"]
    cmy = funcs["cmy"]

    ax = axes[0, 0]
    ax.plot(nodes, bending, ls=ls, color=lc, label=label)
    ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    ax.set_ylabel("$w$ [m]", rotation="horizontal", ha="left", labelpad=lpad, va="center")

    ax = axes[0, 1]
    ax.plot(nodes, twisting, ls=ls, color=lc, label=label)
    ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    ax.set_ylabel("$\\theta$ [$^{\\circ}$]", rotation="horizontal", ha="left", labelpad=lpad, va="center")

    liftTitle = f"Lift ({lift:0.1e}N, $C_L$={cl:.2f})"
    momTitle = f"Mom. ({mom:0.1e}N-m," + " $C_{My}$=" + f"{cmy:.2f})"

    ax = axes[1, 0]
    xloc = 0.05
    ax.plot(nodes, spanLift, ls=ls, color=lc, label=label)
    ax.set_ylabel("$L$ [N/m]", rotation="horizontal", ha="left", labelpad=lpad, va="center")
    ax.set_xlabel("$y$ [m]")
    if do_annotate:
        ax.annotate(
            liftTitle,
            xy=(xloc, 0.1 * iic),
            ls=ls,
            color=lc,
            xycoords="axes fraction",
            fontsize=fs_lgd,
        )

    ax = axes[1, 1]
    ax.plot(nodes, spanMoment, ls=ls, color=lc, label=label)
    ax.set_ylabel(
        "$M_y$\n[N-m/m]",
        rotation="horizontal",
        ha="left",
        labelpad=lpad,
    )
    ax.set_xlabel("$y$ [m]")
    if do_annotate:
        ax.annotate(
            momTitle,
            xy=(xloc, 0.1 * iic),
            color=lc,
            xycoords="axes fraction",
            fontsize=fs_lgd,
        )

    return fig, axes


def plot_dragbuildup(
    fig,
    axes,
    funcs,
    label: str,
    cm,
    fs_lgd: float,
    iic: int,
    includes=["cdpr", "cdi", "cdw", "cds", "cdj"],
):
    alllabels = ["$C_{D,{pr}}$", "$C_{D,{i}}$", "$C_{D,w}$", "$C_{D,{s}}$", "$C_{D,{j}}$"]
    costData = []
    labels = []
    for ii, cost in enumerate(includes):
        costData.append(funcs[cost])
        labels.append(alllabels[ii])

    # costData = [funcs["cdpr"], funcs["cdi"], funcs["cds"], funcs["cdj"]]

    def absolute_value(val):
        """Callback to return labels"""
        a = np.round(val / 100.0 * np.array(costData).sum(), 0)
        a = f"{a}\n{val:.2f}\%"  # output text
        a = f"{val:.2f}\%"  # output text
        return a

    ax = axes.flatten()[iic]
    # _, _, autotexts = ax.pie(
    #     costData,
    #     labels=labels,
    #     colors=cm,
    #     autopct=absolute_value,
    # )
    # for autotext in autotexts:
    #     autotext.set_color("white")

    # Bar plot with drag components
    ax.bar(labels, costData, color=cm[0])
    ax.set_ylabel("$C_D$", rotation="horizontal", ha="right", va="center")
    # Put percentages on top of bars
    for ii, cost in enumerate(costData):
        ax.text(
            ii,
            cost,
            f"{cost/np.array(costData).sum()*100:.2f}\%",
            ha="center",
            va="bottom",
            fontsize=fs_lgd,
        )

    # Set title with vertical label pad
    ax.set_title(f"{label}", pad=40)

    return fig, axes


def plot_dimdragbuildup(
    fig,
    axes,
    funcs,
    title: str,
    label: str,
    cm,
    fs_lgd: float,
    iic: int,
    includes=["cdpr", "cdi", "cdw", "cds", "cdj"],
):
    alllabels = ["${D_{pr}}$", "$D_{i}$", "$D_w$", "${D_{s}}$", "${D_{j}}$"]
    costData = []
    labels = []
    for ii, cost in enumerate(includes):
        costData.append(funcs[cost])
        labels.append(alllabels[ii])

    xticks = np.arange(len(labels))
    axes.set_xticks(xticks)
    width = 0.4
    # costData = [funcs["cdpr"], funcs["cdi"], funcs["cds"], funcs["cdj"]]

    def absolute_value(val):
        """Callback to return labels"""
        a = np.round(val / 100.0 * np.array(costData).sum(), 0)
        a = f"{a}\n{val:.2f}\%"  # output text
        a = f"{val:.2f}\%"  # output text
        return a

    # ax = axes.flatten()[iic]
    ax = axes
    # _, _, autotexts = ax.pie(
    #     costData,
    #     labels=labels,
    #     colors=cm,
    #     autopct=absolute_value,
    # )
    # for autotext in autotexts:
    #     autotext.set_color("white")

    # Bar plot with drag components
    # ax.bar(labels, costData, color=cm[0])
    offset = -width / 2 + width * iic
    ax.bar(xticks + offset, costData, width, color=cm[iic], label=label)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$D$ [N]", rotation="horizontal", ha="right", va="center")
    # Put percentages on top of bars
    for ii, cost in enumerate(costData):
        ax.text(
            xticks[ii] + offset,
            cost,
            f"{cost/np.array(costData).sum()*100:.2f}\%",
            ha="center",
            va="bottom",
            fontsize=fs_lgd,
            zorder=10,
        )

    # Set title with vertical label pad
    ax.set_title(f"{title}", pad=40)

    return fig, axes


def plot_forced(
    fig,
    axes,
    fExtSweep,
    waveAmpSpectrum,
    deflectionRAO,
    dynLiftRAO,
    dynMomentRAO,
    genXferFcn,
    flowSpeed,
    fs_lgd,
    cm,
    alpha,
    elem=1,
):
    """
    Plot harmonically forced response of the tip of the wing

    Parameters
    ----------
    fExtSweep : _type_
        external forcing frequency sweep
    dynLiftRAO : _type_
        frequency RAO of lift [N]
    dynMomentRAO : _type_
        frequency RAO of moment [N-m] about the midchord
    genRAO : _type_
        General response amplitude operator of deflections wrt vectorized forces
    flowSpeed : _type_
        flow speed [m/s]
    fs_lgd : _type_
        legend font size
    elem : _type_
        element type
    """

    # Get tip DOFs based on element type
    if elem == 0:
        TwistIdx = -2
    else:
        TwistIdx = -5
    if elem == 0:
        OOPIdx = -4
    else:
        OOPIdx = -7

    xLabel = "$f$ [Hz]"

    # # ************************************************
    # #     Plot deflection FRF
    # # ************************************************
    # ax = axes[0, 0]
    # yLabel = r"$M_{\bar{w}\bar{f}_z}(\omega)$"
    # M_wFRF = np.zeros_like(fExtSweep)
    # # for ii, entry in enumerate(genRAO[:, OOPIdx, OOPIdx]):
    # # for ii, entry in enumerate(genRAO[:, 2::9, OOPIdx]):
    # for ii, entry in enumerate(fExtSweep):
    #     # for ii, entry in enumerate(deflectionRAO[:,OOPIdx]):
    #     meanDef = 0.0  # mean deflection over the span

    #     ndef = len(genXferFcn[ii, 2::9, OOPIdx])

    #     bendingXferFcn = genXferFcn[ii, 2::9, 2::9]

    #     for jj in range(ndef):
    #         for kk in range(ndef):
    #             meanDef += np.sqrt(bendingXferFcn[jj, kk][0] ** 2 + bendingXferFcn[jj, kk][1] ** 2) / (ndef**2)

    #     M_wFRF[ii] = meanDef
    #     # M_wtipRAO[ii] = np.sqrt(entry[0] ** 2 + entry[1] ** 2)
    #     # M_wtipRAO[ii] = entry

    # ax.plot(fExtSweep, M_wFRF, color=cm[0], label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    # ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")

    # ax = axes[1, 0]
    # yLabel = r"$M_{\overline{\vartheta} \bar{\tau_y}}(\omega)$"
    # M_thetaFRF = np.zeros_like(fExtSweep)
    # # for ii, entry in enumerate(genRAO[:, TwistIdx, TwistIdx]):
    # # for ii, entry in enumerate(genRAO[:, 4::9, TwistIdx]):
    # # for ii, entry in enumerate(deflectionRAO[:, TwistIdx]):
    # for ii, entry in enumerate(fExtSweep):
    #     meanDef = 0.0  # mean deflection over the span

    #     ndef = len(genXferFcn[ii, 4::9, TwistIdx])

    #     twistXferFcn = genXferFcn[ii, 4::9, 4::9]

    #     for jj in range(ndef):
    #         for kk in range(ndef):
    #             meanDef += np.sqrt(twistXferFcn[jj, kk][0] ** 2 + twistXferFcn[jj, kk][1] ** 2) / (ndef**2)

    #     M_thetaFRF[ii] = meanDef
    #     # M_thetatipRAO[ii] = np.sqrt(entry[0] ** 2 + entry[1] ** 2)
    #     # M_thetatipRAO[ii] = entry

    # ax.plot(fExtSweep, M_thetaFRF, color=cm[0], label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    # ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")

    # ************************************************
    #     Deflection RAO
    # ************************************************
    angleLim = 90
    ax = axes[0, 0]
    ylabel = r"$\frac{|\overline{w}|}{\zeta}$"
    M_wRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(fExtSweep):
        ndef = len(deflectionRAO[ii, 2::9])
        bendingRAO = deflectionRAO[ii, 2::9]
        meanDef = 0.0
        for jj in range(ndef):
            meanDef += np.sqrt(bendingRAO[jj][0] ** 2 + bendingRAO[jj][1] ** 2) / ndef

        M_wRAO[ii] = meanDef

    ax.plot(fExtSweep, M_wRAO, color=cm[0], alpha=alpha, label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    ax.set_ylabel(ylabel, rotation="horizontal", ha="right", va="center")

    ax = axes[1, 0]
    ylabel = r"$ \angle \frac{w_{tip}}{\zeta}$ [$^\circ$]"
    arg_wRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(fExtSweep):
        ndef = len(deflectionRAO[ii, 2::9])
        bendingRAO = deflectionRAO[ii, 2::9]
        # meanDef = 0.0
        # for jj in range(ndef):
        # meanDef += (bendingRAO[jj][0] + 1j * bendingRAO[jj][1]) / ndef
        tipDef = bendingRAO[-1][0] + 1j * bendingRAO[-1][1]
        arg_wRAO[ii] = np.angle(tipDef, deg=True)

    ax.plot(fExtSweep, arg_wRAO, color=cm[0], alpha=alpha, label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    ax.set_ylabel(ylabel, rotation="horizontal", ha="right", va="center")
    ax.set_ylim(-angleLim, angleLim)

    ax = axes[0, 1]
    ylabel = r"$\frac{|\overline{\vartheta}|}{\zeta}$ [$^\circ$/m]"
    M_thetaRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(fExtSweep):
        ndef = len(deflectionRAO[ii, 4::9])
        twistingRAO = deflectionRAO[ii, 4::9]
        meanDef = 0.0
        for jj in range(ndef):
            meanDef += np.sqrt(twistingRAO[jj][0] ** 2 + twistingRAO[jj][1] ** 2) / ndef

        M_thetaRAO[ii] = meanDef

    ax.plot(fExtSweep, np.rad2deg(M_thetaRAO), color=cm[0], alpha=alpha, label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    ax.set_ylabel(ylabel, rotation="horizontal", ha="right", va="center")

    ax = axes[1, 1]
    ylabel = r"$ \angle \frac{\vartheta_{tip}}{\zeta}$ [$^\circ$]"
    arg_thetaRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(fExtSweep):
        ndef = len(deflectionRAO[ii, 4::9])
        twistingRAO = deflectionRAO[ii, 4::9]
        # meanDef = 0.0
        # for jj in range(ndef):
        # meanDef += (bendingRAO[jj][0] + 1j * bendingRAO[jj][1]) / ndef
        tipDef = twistingRAO[-1][0] + 1j * twistingRAO[-1][1]

        arg_thetaRAO[ii] = np.angle(tipDef, deg=True)

    ax.plot(fExtSweep, arg_thetaRAO, color=cm[0], alpha=alpha, label="$U_{\infty}=$%.1f m/s" % (flowSpeed))
    ax.set_ylabel(ylabel, rotation="horizontal", ha="right", va="center")
    ax.set_ylim(-angleLim, angleLim)

    # NOTE: TBH these are not terribly useful unless you're looking at transmitted force into the hull of the boat
    # which you should not, because you want to couple the foil to the ship model!
    # Answer and theory, because the wave loads go to zero at high frequency, this will taper off significantly
    # You actually want it wrt the magnitude of the input force at the tip
    # ************************************************
    #     Plot forces
    # ************************************************
    ax = axes[0, 2]
    yLabel = r"$\frac{|\tilde{L}|}{\zeta}$"  # Lift

    LiftRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(dynLiftRAO):
        LiftRAO[ii] = np.sqrt(entry[0] ** 2 + entry[1] ** 2)
    ax.plot(fExtSweep, LiftRAO, color=cm[0], alpha=alpha)
    ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")
    # ax.set_xlabel(xLabel)

    ax = axes[1, 2]
    yLabel = r"$\angle \frac{\tilde{L}}{\zeta}$ [$^\circ$]"  # Lift
    argLiftRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(dynLiftRAO):
        argLiftRAO[ii] = np.angle(entry[0] + 1j * entry[1], deg=True)
    ax.plot(fExtSweep, argLiftRAO, color=cm[0], alpha=alpha)
    ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")
    ax.set_ylim(-angleLim, angleLim)

    ax = axes[0, 3]
    yLabel = r"$\frac{|\tilde{M}_y|}{\zeta}$"  # Moment
    MomRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(dynMomentRAO):
        MomRAO[ii] = np.sqrt(entry[0] ** 2 + entry[1] ** 2)
    ax.plot(fExtSweep, MomRAO, color=cm[0], alpha=alpha)
    ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")

    ax = axes[1, 3]
    yLabel = r"$\angle \frac{\tilde{M}_y}{\zeta}$ [$^\circ$]"
    argMomRAO = np.zeros_like(fExtSweep)
    for ii, entry in enumerate(dynMomentRAO):
        argMomRAO[ii] = np.angle(entry[0] + 1j * entry[1], deg=True)
    ax.plot(fExtSweep, argMomRAO, color=cm[0], alpha=alpha)
    ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")
    ax.set_ylim(-angleLim, angleLim)

    # # ************************************************
    # #     Wave spectrum plot
    # # ************************************************
    # ax = axes[0, -1]
    # yLabel = r"$\zeta(\omega)$"
    # ax.plot(fExtSweep, waveAmpSpectrum, color=cm[0])
    # ax.set_ylabel(yLabel, rotation="horizontal", ha="right", va="center")
    # # ax.set_xlabel(xLabel)

    for ax in axes[1, :].flatten():
        ax.set_xlabel(xLabel)
    for ax in axes.flatten():
        # ax.legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False)
        nplt.adjust_spines(ax, outward=True)

    return fig, axes


def plot_naturalModeShapes(fig, axes, y, nModes: int, modeShapes: dict, modeFreqs: dict, ls="-", nshift=12):
    """
    Plot the mode shapes for the structural and wet modes in quiescent fluid (U = 0 m/s)

    Parameters
    ----------
    y : array
        spanwise coordinate [m]
    nModes : int
        number of modes to plot
    modeShapes : dict
        stores dry and wet mode shapes
    modeFreqs : dict
        stores dry and wet natural frequencies
    ls : str
        Line styles
    nshift : int
        number of nodes to shift the labels by
    """
    # Check if we want to save

    eta = y / y[-1]  # Normalized spanwise coordinate

    # Font options
    labelpad = 40
    legfs = 20

    # Unpack data
    structBM = modeShapes["structBM"]
    structTM = modeShapes["structTM"]
    wetBM = modeShapes["wetBM"]
    wetTM = modeShapes["wetTM"]
    wetNatFreqs = modeFreqs["wetNatFreqs"]
    structNatFreqs = modeFreqs["structNatFreqs"]

    bendLabel = f"OOP\nbending"
    twistLabel = f"Twist"

    # ---------------------------
    #   Struct modes
    # ---------------------------
    # --- Bending modes ---
    for ii in range(nModes):
        ax = axes[0, 0]

        modeShapeDict = {
            "structBM": structBM[ii, :],
            "structTM": structTM[ii, :],
        }
        maxVal = compute_normFactorModeShape(modeShapeDict)

        structBM[ii, :] /= maxVal

        # ax.plot(eta, structBM[ii, :], label=bendLabel, ls=ls, c=color)
        # ax.plot(eta, structTM[ii, :], label=twistLabel, ls=ls[1], c=color)
        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(
            eta,
            structBM[ii, :],
            label=f"Mode {ii+1} {labelString}",
            ls=ls,
            color=ccm[ii],
        )
        # ax.annotate(
        #     f"Mode {ii+1} {labelString}",
        #     xy=(eta[-1 - nshift], structBM[ii, -1 - nshift]),
        #     color=ccm[ii],
        #     bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
        #     va="top",
        #     xytext=(0, -2),
        #     textcoords="offset points",
        #     size=legfs,
        # )

    ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_title("Dry Modes", pad=labelpad)
    ax.legend(
        fontsize=legfs,
        labelcolor="linecolor",
        loc="center left",
        frameon=False,
        ncol=1,
        bbox_to_anchor=(1, 0.5),
    )

    # --- Twist modes ---
    for ii in range(nModes):
        ax = axes[1, 0]

        modeShapeDict = {
            "structBM": structBM[ii, :],
            "structTM": structTM[ii, :],
        }
        maxVal = compute_normFactorModeShape(modeShapeDict)

        structTM[ii, :] /= maxVal

        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(
            eta,
            structTM[ii, :],
            label=f"Mode {ii+1} {labelString}",
            ls=ls,
            color=ccm[ii],
        )

    ax.set_ylabel(twistLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_xlabel(r"$\bar{y}$ [-]")

    # ---------------------------
    #   Wet modes
    # ---------------------------
    # --- Bend modes ---
    # First normalize the modes by max
    for ii in range(nModes):
        ax = axes[0, 1]

        modeShapeDict = {
            "wetBM": wetBM[ii, :],
            "wetTM": wetTM[ii, :],
        }
        maxVal = compute_normFactorModeShape(modeShapeDict)

        wetBM[ii, :] /= maxVal

        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, wetBM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, color=ccm[ii])
        ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad, va="center")
        # ax.annotate(
        #     f"Mode {ii+1} {labelString}",
        #     xy=(eta[-1 - nshift], wetBM[ii, -1 - nshift]),
        #     color=ccm[ii],
        #     bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
        #     va="top",
        #     xytext=(0, -2),
        #     textcoords="offset points",
        #     size=legfs,
        # )
    ax.set_title("Wet Modes", pad=labelpad)
    ax.legend(
        fontsize=legfs,
        labelcolor="linecolor",
        loc="center left",
        frameon=False,
        ncol=1,
        bbox_to_anchor=(1, 0.5),
    )

    # --- Twist modes ---
    for ii in range(nModes):
        ax = axes[1, 1]

        modeShapeDict = {
            "wetBM": wetBM[ii, :],
            "wetTM": wetTM[ii, :],
        }
        maxVal = compute_normFactorModeShape(modeShapeDict)

        wetTM[ii, :] /= maxVal

        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, wetTM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, color=ccm[ii])

    ax.set_ylabel(twistLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_xlabel("$\\bar{y}$ [-]")

    fig.suptitle("Normalized mode shapes", fontsize=40)

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)
        ax.set_ylim([-1.1, 1.1])
    return fig, axes


def plot_modeShapes(
    comm,
    vRange,
    y,
    chordVec,
    flutterSol: dict,
    fact: float,
    ls="-",
    alpha=1.0,
    units="m/s",
    outputDir=None,
):
    """
    Plot the hydroelastic mode shapes where each processor has its own mode

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to plot on
    axes : matplotlib.axes._subplots.AxesSubplot
        axes to plot on, 2
    y : array_like
        spanwise location
    flutterSol : dict
        see 'postprocess_flutterevals()' for data structure
    ls : str, optional
        line style for plots, by default "-"
    alpha : float, optional
        alpha for plots, by default 1.0
    """

    # Sort keys to get consistent plotting
    sortedModesNumbers = sorted(flutterSol.keys(), key=int)

    labelpad = 40
    legfs = 20
    xLabel = r"$\bar{y}$ [-]"

    vLow = vRange[0]
    vHigh = vRange[1]

    # ************************************************
    #     Loop over modes
    # ************************************************
    for mm, key in enumerate(sortedModesNumbers):
        if comm.rank == mm:
            iic = mm % len(cm)  # color index
            vSweep = flutterSol[key]["U"]
            gSweep = flutterSol[key]["pvals_r"]
            fSweep = flutterSol[key]["pvals_i"]
            if units == "kts":
                vSweep = 1.94384 * np.array(vSweep)  # kts
            elif units == "m/s":
                pass
            else:
                raise ValueError(f"Unsupported units: {units}")
            # ---------------------------
            #   Loop speeds
            # ---------------------------
            for jj, v in enumerate(vSweep):
                # Only plot if in speed range of interest
                if v < vLow or v > vHigh:
                    continue
                else:
                    figsize = (8 * fact, 10 * fact)
                    fig, axes = plt.subplots(
                        nrows=2,
                        ncols=1,
                        sharey="row",
                        constrained_layout=True,
                        figsize=figsize,
                    )

                    # --- Pull out shapes and normalize ---
                    w_r, psi_r = get_bendingtwisting(flutterSol[key]["R_r"][:, jj], nDOF=4)
                    w_i, psi_i = get_bendingtwisting(flutterSol[key]["R_i"][:, jj], nDOF=4)
                    w_mag = np.sqrt(w_r**2 + w_i**2)
                    psi_mag = np.sqrt(psi_r**2 + psi_i**2)

                    modeShapes = {
                        "w": w_mag,
                        "psi": psi_mag,
                    }
                    maxVal = compute_normFactorModeShape(modeShapes)

                    labelString = f"({fSweep[jj]:.2f}" + " Hz)"

                    # --- Bending ---
                    ax = axes[0]
                    mShape = np.hstack([0.0, w_mag / maxVal])  # add zero at root
                    ax.plot(
                        y,
                        mShape,
                        label=f"Mode {mm+1} {labelString}",
                        ls=ls,
                        color=cm[iic],
                    )
                    ax.set_ylabel("OOP\nBending", rotation=0, labelpad=labelpad)
                    ax.legend(labelcolor="linecolor", loc="best", frameon=False)

                    # --- Twisting ---
                    ax = axes[1]
                    mShape = np.hstack([0.0, psi_mag / maxVal])  # add zero at root
                    ax.plot(y, mShape, ls=ls, color=cm[iic])
                    ax.set_ylabel("Twist", rotation=0, labelpad=labelpad)
                    ax.set_xlabel(xLabel)

                    mathStr = "$U_{\infty} = %.1f$" % v
                    axes[0].set_title(f"Hydroelastic mode\n{mathStr} {units}")
                    plt.tight_layout()

                    for ax in axes.flatten():
                        nplt.adjust_spines(ax, outward=True)
                        ax.set_ylim([-1.1, 1.1])
                        ax.set_xlim([0, 1.05])
                    plt.savefig(f"{outputDir}/mode{mm+1}-{jj:04d}.png", dpi=200)
                    plt.close()


def plot_vg_vf_rl(
    fig,
    axes,
    flutterSol: dict,
    cm,
    ls="-",
    alpha=1.0,
    units="m/s",
    marker=None,
    showRLlabels=False,
    annotateModes=False,
    nShift=0,
    instabPts=None,
):
    """
    Plot the V-g, V-f, and R-L diagrams

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to plot on
    axes : matplotlib.axes._subplots.AxesSubplot
        axes to plot on, there must be 2x2 axes
    flutterSol : dict
        see 'postprocess_flutterevals()' for data structure
    ls : str
        line style for plots
    alpha : float
        alpha for plots
    units : str, optional
        units for velocity, by default "m/s"
    marker : str, optional
        marker for line plots to see points, by default None
    showRLlabels : bool, optional
        show R-L speed labels, by default True
    annotateModes : bool, optional
        annotate modes on plots, by default False
    nShift : int, optional
        number of points to shift labels, by default 0
    instabPts : list, optional
        list of points to mark instability
    """

    # Sort keys to get consistent plotting
    sortedModesNumbers = sorted(flutterSol.keys(), key=int)

    flutterColor = "magenta"

    # Hardcoded settings
    xytext = (-5, -2)
    xytextVG = (-5, -3)
    # xytext = (-7, 10)
    labelpad = 40
    legfs = 20

    # gLabel = "$g$ [rad/s]"
    # fLabel = "$f$ [rad/s]"
    # hz2rad = 2 * np.pi
    gLabel = "$g$ [1/s]"
    fLabel = "$f$ [Hz]"
    hz2rad = 1
    # ************************************************
    #     V-g diagram
    # ************************************************
    ax = axes[0, 0]
    xlabel = "$U_{\infty}$ " + f"[{units}]"
    yticks = []
    emptyModes = []
    bifCtr = 0  # Bifurcation counter
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)  # color index

        # --- Units for labeling ---
        vSweep = flutterSol[key]["U"]
        gSweep = flutterSol[key]["pvals_r"] * hz2rad
        fSweep = flutterSol[key]["pvals_i"] * hz2rad  # for checking bifurcation

        if units == "kts":
            vSweep = 1.94384 * np.array(vSweep)  # kts
        elif units == "m/s":
            pass
        else:
            raise ValueError(f"Unsupported units: {units}")

        try:  # Plot only if the data exists
            # breakpoint()
            ax.plot(
                vSweep,
                gSweep,
                ls=ls,
                color=cm[iic],
                label=f"Mode {key}",
                marker=marker,
                alpha=alpha,
            )
            # ax.scatter(vSweep, gSweep, color=(cm[iic]), marker=marker)
            # --- Label mode number on the line ---
            if annotateModes:
                # check if mode is bifurcated and then alternate the label position
                if fSweep[0] < 1e-2:
                    if bifCtr % 2 == 0:
                        va = "top"
                        xytextVG = (-7, 0)
                    else:
                        xytextVG = (-7, 0)
                        va = "bottom"
                    ha = "right"
                    nShift = 3
                    bifCtr += 1
                elif ii + 1 == 2:  # special case don't delete
                    xytext = (7, 7)
                    va = "top"
                    ha = "left"
                elif ii + 1 == 4:  # special case don't delete
                    nShift = 20
                else:
                    ha = "right"
                    va = "top"

                start = np.array([vSweep[0 + nShift], gSweep[0 + nShift]])
                end = np.array([vSweep[-1], gSweep[-1]])

                ax.annotate(
                    f"Mode {ii+1}",
                    xy=(start[0], start[1]),
                    # xy=(end[0], end[1]),
                    ha=ha,
                    va=va,
                    color=cm[iic],
                    fontsize=legfs,
                    xytext=xytextVG,
                    textcoords="offset points",
                    bbox=dict(boxstyle="square,pad=-0.1", ec="white", linewidth=0, fc="white", alpha=0.9),
                )
            yticks.append(gSweep[-1])
        except Exception:
            emptyModes.append(key)
    print(f"Empty modes: {emptyModes}")
    ax.set_ylim(top=10)
    ax.set_ylabel(gLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_title("$V$-$g$", pad=labelpad)
    ax.set_xlabel(xlabel)
    # --- Put flutter boundary on plot ---
    ax.axhline(
        y=0.0,
        # label="Flutter boundary",
        color=flutterColor,
        ls="-",
        alpha=0.5,
        # path_effects=[patheffects.withTickedStroke()], # ugly
    )
    ax.annotate(
        "Hydroelastic instability",
        xy=(0.5, 0.95),
        ha="center",
        xycoords="axes fraction",
        size=legfs,
        color=flutterColor,
        alpha=0.5,
    )
    # ax.set_yticks(yticks + [0])

    # --- Instability points ---
    if instabPts is not None:
        for pt in instabPts:
            iic = (int(pt[2]) - 1) % len(cm)
            if units == "kts":
                critSpeed = 1.94384 * pt[0]
            elif units == "m/s":
                critSpeed = pt[0]
            else:
                print(f"Unsupported units: {units}")
            ax.scatter(critSpeed, pt[1], color=cm[iic], marker="x", s=100, alpha=alpha)

    # ************************************************
    #     V-f diagram
    # ************************************************
    ax = axes[1, 0]
    bifCtr = 0  # Bifurcation counter
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)  # color index

        vSweep = flutterSol[key]["U"]
        fSweep = flutterSol[key]["pvals_i"] * hz2rad
        # --- Units for labeling ---
        if units == "kts":
            vSweep = 1.94384 * np.array(vSweep)  # kts
        elif units == "m/s":
            pass
        else:
            raise ValueError(f"Unsupported units: {units}")

        try:  # Plot only if the data exists
            ax.plot(
                vSweep,
                fSweep,
                ls=ls,
                color=cm[iic],
                label=f"Mode {key}",
                marker=marker,
                alpha=alpha,
            )
            # ax.scatter(vSweep, fSweep, color=(cm[iic]), marker=marker)
            start = np.array([vSweep[0], fSweep[0]])
            end = np.array([vSweep[-1], fSweep[-1]])
            # --- Label mode number on the line ---
            # check if mode is bifurcated and then alternate the label position
            if fSweep[0] < 1e-2:
                if bifCtr % 2 == 0:
                    va = "bottom"
                    xytext = (-5, 0)
                else:
                    xytext = (-5, 0)
                    va = "top"
                ha = "right"
                bifCtr += 1
            # elif ii+1 == 2: # special case
            #     xytext = (-5, -2)
            #     va = "top"
            else:
                ha = "left"
                va = "bottom"

            if annotateModes:
                ax.annotate(
                    f"Mode {ii+1}",
                    xy=(start[0], start[1]),
                    ha=ha,
                    va=va,
                    color=cm[iic],
                    fontsize=legfs,
                    xytext=xytext,
                    textcoords="offset points",
                    bbox=dict(boxstyle="square,pad=-0.1", ec="white", linewidth=0, fc="white", alpha=0.9),
                )
        except Exception:
            continue

    # --- Instability points ---
    # Also plot point at frequency of instability
    if instabPts is not None:
        for pt in instabPts:
            iic = (int(pt[2]) - 1) % len(cm)
            if units == "kts":
                critSpeed = 1.94384 * pt[0]
            elif units == "m/s":
                critSpeed = pt[0]
            else:
                print(f"Unsupported units: {units}")
            ax.scatter(critSpeed, pt[-1], color=cm[iic], marker="x", s=100, alpha=alpha, clip_on=False)

    ax.set_ylabel(fLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_title("$V$-$f$", pad=labelpad)
    # ax.set_xlim(vSweep[0] * 0.99, vSweep[-1] * 1.01)
    ax.set_xlabel(xlabel)

    if not annotateModes:
        ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

    # ************************************************
    #     Root-locus diagram
    # ************************************************
    markerSize = 8
    ax = axes[1, 1]
    bifCtr = 0  # Bifurcation counter
    yticks = []
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)

        vSweep = flutterSol[key]["U"]
        gSweep = flutterSol[key]["pvals_r"] * hz2rad
        fSweep = flutterSol[key]["pvals_i"] * hz2rad
        # --- Units for labeling ---
        if units == "kts":
            vSweep = 1.94384 * np.array(vSweep)  # kts
        elif units == "m/s":
            pass
        else:
            raise ValueError(f"Unsupported units: {units}")

        try:  # Plot only if the data exists
            ax.plot(
                gSweep,
                fSweep,
                ls=ls,
                color=cm[iic],
                label=f"Mode {key}",
                marker=marker,
                alpha=alpha,
            )
            # ax.scatter(gSweep, fSweep, color=(cm[iic]), marker=marker)
            start = np.array([gSweep[0], fSweep[0]])
            end = np.array([gSweep[-1], fSweep[-1]])
            # ax.plot(start[0], start[1], marker="o", markersize=markerSize, color=cm[iic], markeredgecolor="gray")
            # ax.plot(end[0], end[1], marker="^", markersize=markerSize, color=cm[iic], markeredgecolor="gray")

            # --- Label mode number on the line ---
            # check if mode is bifurcated and then alternate the label position
            if fSweep[0] < 1e-2:  # low freq
                if bifCtr % 2 == 0:
                    ha = "right"
                    xytext = (-5, 5)
                else:
                    ha = "left"
                    xytext = (5, 5)
                bifCtr += 1
            else:  # defaults
                ha = "left"
                va = "bottom"
                xytext = (5, 5)

            if annotateModes:
                ax.annotate(
                    f"Mode {ii+1}",
                    # xy=(end[0], end[1]),
                    xy=(start[0], start[1]),
                    ha=ha,
                    color=cm[iic],
                    fontsize=legfs,
                    xytext=(5, 5),
                    textcoords="offset points",
                    # bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                )
            if showRLlabels:  # show R-L speed labels
                ax.annotate(
                    f"{vSweep[0]:.1f}{units}",
                    xy=(start[0], start[1]),
                    color=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, -5),
                    textcoords="offset points",
                    # bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                    va="top",
                )
                ax.annotate(
                    f"{vSweep[-1]:.1f}{units}",
                    xy=(end[0], end[1]),
                    color=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, -5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5, pad=-0.3),
                    va="top",
                )

            # --- Add arror pointing in speed ---
            nmid = int(len(vSweep) // 4)
            ax.annotate(
                "",
                xytext=(np.array([gSweep[-nmid - 1], fSweep[-nmid - 1]])),  # arrow start
                xy=(np.array([gSweep[-nmid], fSweep[-nmid]])),  # arrow end
                arrowprops=dict(
                    # arrowstyle="->",
                    arrowstyle="fancy",
                    # shrinkA=2,
                    color=cm[iic],
                    alpha=0.5,
                    # headwidth=0.3, # does not work
                ),
            )

            yticks.append(fSweep[0])
        except Exception:
            continue

    ax.set_ylabel(fLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_title("Root locus", pad=labelpad)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)
    ax.set_xlabel(gLabel)
    # ax.set_yticks(yticks)

    # --- Put flutter boundary on plot ---
    # ax.set_xlim(right=10)
    ax.axvline(
        x=0.0,
        # label="Flutter boundary",
        color=flutterColor,
        ls="-",
        alpha=0.5,
        # path_effects=[patheffects.withTickedStroke()], # ugly
    )
    ax.annotate(
        "Hydroelastic\ninstability",
        xy=(0.85, 0.5),
        ha="left",
        xycoords="axes fraction",
        size=legfs,
        color=flutterColor,
        alpha=0.5,
    )

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)

    return fig, axes


def plot_dlf(
    fig,
    axes,
    flutterSol: dict,
    cm,
    semichord: float,
    sweepAng: float,
    ls="-",
    alpha=1.0,
    units="m/s",
    nShift=0,
):
    """
    Plot the damping loss factor diagrams.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to plot on
    axes : matplotlib.axes.Axes
        axes to plot on, must be length 2
    flutterSol : dict
        see 'postprocess_flutterevals()' for data structure
    semichord : float
        mean semichord of the wing
    sweepAng : float
        sweep angle of wing in radians
    ls : str, optional
        line style, by default "-"
    alpha : float, optional
        line transparency, by default 1.0
    units : str, optional
        units for velocity, by default "m/s"
    """

    # Sort keys to get consistent plotting
    sortedModesNumbers = sorted(flutterSol.keys(), key=int)

    labelpad = 40
    legfs = 20
    xytext = (-5, 5)
    flutterColor = "magenta"

    yLabel = "$\eta$ [-]"

    # ************************************************
    #     Plot DLF vs. velocity
    # ************************************************
    ax = axes[0]
    xLabel = "$U_{\infty}$ " + f"[{units}]"
    emptyModes = []
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)  # color index

        vSweep = flutterSol[key]["U"]
        # --- Units for labeling ---
        if units == "kts":
            vSweep = 1.94384 * np.array(vSweep)  # kts
        elif units == "m/s":
            pass
        else:
            raise ValueError(f"Unsupported units: {units}")

        dlf = -2 * np.divide(flutterSol[key]["pvals_r"], flutterSol[key]["pvals_i"])

        try:
            ax.plot(vSweep, dlf, ls=ls, color=cm[iic], label=f"Mode {key}", alpha=alpha)
            start = np.array([vSweep[0 + nShift], dlf[0 + nShift]])
            end = np.array([vSweep[-1], dlf[-1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(start[0], start[1]),
                ha="right",
                # xy=(end[0], end[1]),
                color=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
        except Exception:
            emptyModes.append(key)

    print(f"Empty modes: {emptyModes}")

    # --- Put flutter boundary on plot ---
    ax.axhline(
        y=0.0,
        # label="Flutter boundary",
        color=flutterColor,
        ls="--",
        # path_effects=[patheffects.withTickedStroke()], # ugly
    )
    ax.annotate(
        "Hydroelastic instability",
        xy=(0.5, 0.0),
        ha="center",
        xycoords="axes fraction",
        size=legfs,
        color=flutterColor,
        zorder=1,
    )

    ax.set_ylabel(yLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_xlabel(xLabel)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

    print(f"nshift for dlf plot: {nShift}")
    # ************************************************
    #     Plot DLF vs. k
    # ************************************************
    ax = axes[1]
    xLabel = "$k$ [-]"
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)  # color index

        fSweep = flutterSol[key]["pvals_i"]
        vSweep = flutterSol[key]["U"]
        kSweep = np.divide(fSweep * 2 * np.pi * semichord, vSweep * np.cos(sweepAng))
        dlf = -2 * np.divide(flutterSol[key]["pvals_r"], fSweep)

        try:
            ax.loglog(kSweep, dlf, ls=ls, color=cm[iic], label=f"Mode {key}", alpha=alpha)
            start = np.array([kSweep[0], dlf[0]])
            end = np.array([kSweep[-1], dlf[-1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(start[0], start[1]),
                ha="left",
                # xy=(end[0], end[1]),
                color=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
        except Exception:
            continue

    ax.set_ylabel(yLabel, rotation=0, labelpad=labelpad, va="center")
    ax.set_xlabel(xLabel)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

    plt.suptitle("Damping loss factor trends")

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)

    return fig, axes


def pytecplot_plotmesh(args, fname: str):
    if not args.batch:
        tp.session.connect()

    lay = ntp.Layout()


def set_my_plot_settings(is_paper=False):
    fs_lgd = 20
    fs = 25
    plt.style.use(nplt.get_style())  # all settings
    myOptions = {
        "font.size": fs,
        "font.family": "sans-serif",  # set to "serif" to get the same as latex
        "font.sans-serif": ["Helvetica"],  # this does not work on all systems
        "text.usetex": False,
        "text.latex.preamble": [
            r"\usepackage{lmodern}",  # latin modern font
            r"\usepackage{amsmath}",  # for using equation commands
            r"\usepackage{helvet}",  # should make latex serif in helvet now
            r"\usepackage{sansmath}",
            r"\sansmath",  # supposed to force math to be rendered in sans-serif font
        ],
    }
    if is_paper:
        myOptions.update(
            {
                "font.family": "serif",
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{lmodern} \usepackage{amsmath}  \usepackage{helvet}",
            }
        )
    plt.rcParams.update(myOptions)

    # colormap
    niceColors = sns.color_palette("tab10")
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ls = ["-", "--", "-.", ":"]
    markers = ["o", "^", "v", "s"]

    return cm, fs_lgd, fs, ls, markers
