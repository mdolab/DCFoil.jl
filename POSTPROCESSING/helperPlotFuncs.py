# --- Python 3.9 ---
"""
@File    :   divAndFlutterPlots.py
@Time    :   2023/02/03
@Author  :   Galen Ng
@Desc    :   Contains functions for plotting flutter and divergence points. Some ideas taken from Dr. Eirikur Jonsson
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

# import matplotlib
# import matplotlib.patheffects as patheffects
# from matplotlib.offsetbox import AnchoredText

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots

# import nicetecplots as ntp

# --- Enable niceplot colors as a list ---
# plt.style.use(niceplots.get_style())
# niceColors = niceplots.get_colors().values()
niceColors = sns.color_palette("tab10")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Continuous colormap
niceColors = sns.color_palette("cool")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
ccm = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_wing(DVDict: dict):
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

    nNodes = DVDict["nNodes"]
    y = np.linspace(0, DVDict["s"], nNodes)

    # ax = axes[0]
    # ************************************************
    #     Plot planform
    # ************************************************
    # --- Plot outer mold shape ---
    ax.plot(y, -0.5 * np.array(DVDict["c"]), c=moldColor)
    ax.plot(y, 0.5 * np.array(DVDict["c"]), c=moldColor)
    ax.plot([y[-1], y[-1]], [0.5 * DVDict["c"][-1], -0.5 * DVDict["c"][-1]], c=moldColor)

    # --- Plot elastic axis (E.A.) ---
    ab = -np.array(DVDict["ab"])
    ax.plot(y, ab, c=cm[0], ls="--", alpha=alpha)
    ax.annotate(
        f"E.A.",
        xy=(y[nNodes // 2], ab[nNodes // 2]),
        c=cm[0],
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
    xalpha = -np.array(DVDict["x_αb"]) - np.array(DVDict["ab"])
    ax.plot(y, xalpha, c=cm[1], ls="-.", alpha=alpha)
    ax.annotate(
        "C.G.",
        xy=(y[nNodes // 3], xalpha[nNodes // 3]),
        c=cm[1],
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
    ax.scatter(y, np.zeros_like(y), c=moldColor, marker="o", alpha=alpha)

    ax.set_xticks([y[0], y[-1]])
    ax.set_yticks([-0.5 * DVDict["c"][0], 0.0, 0.5 * DVDict["c"][0]])
    ax.set_xlim([y[0], y[-1] * 1.25])
    ax.set_ylim([-0.5 * DVDict["c"][0] * 1.5, 0.5 * DVDict["c"][0] * 1.5])
    ax.set_xlabel("$y$ [m]")
    ax.set_ylabel("$x$ [m]", labelpad=labelpad, rotation=0)
    ax.set_aspect("equal")

    ax.set_title("Wing planform")

    # --- Text annotations ---
    geomText = f"$\\alpha_0$\n$\\Lambda$\n$\\theta_f$\n$t/c$\nnNode\nnDOF"
    valText = (
        f"{DVDict['α₀']}"
        + "$^{{\\circ}}$\n"
        + f"{DVDict['Λ']*180/np.pi:.1f}"
        + "$^{{\\circ}}$\n"
        + f"{DVDict['θ']*180/np.pi:.1f}"
        + "$^{{\\circ}}$\n"
        + f"{DVDict['toc']*100:0.1f}%\n"
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
    niceplots.adjust_spines(ax, outward=True)

    return fig, ax


def plot_forced(fExtSweep, dynTipBending, dynTipTwisting, dynLift, dynMoment, fname=None):
    """
    Plot forced response of the tip of the wing

    Parameters
    ----------
    fExtSweep : _type_
        external forcing frequency sweep
    dynTipBending : _type_
        frequency response of tip bending [m]
    dynTipTwisting : _type_
        frequency response of tip twisting [deg]
    dynLift : _type_
        frequency response of lift [N]
    dynMoment : _type_
        frequency response of moment [N-m] about the midchord
    """

    # Create figure object
    labelpad = 30
    legfs = 15
    nrows = 2
    ncols = 2
    figsize = (6 * ncols, 6 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, constrained_layout=True, figsize=figsize)
    xLabel = "$f_{ext}$ [Hz]"
    # ************************************************
    #     Plot tip deflections
    # ************************************************
    ax = axes[0, 0]
    yLabel = r"$\frac{w}{w(f_{ext}=0)}$"
    nondim = dynTipBending[0]  # nondimensionalize by the static value
    ax.plot(fExtSweep, dynTipBending / nondim, c=cm[0])
    ax.set_ylabel(yLabel, labelpad=labelpad, rotation=0)
    ax.set_xlabel(xLabel)

    ax = axes[0, 1]
    yLabel = r"$\frac{\psi}{\psi(f_{ext}=0)}$"
    nondim = dynTipTwisting[0]  # nondimensionalize by the static value
    ax.plot(fExtSweep, dynTipTwisting / nondim, c=cm[0])
    ax.set_ylabel(yLabel, labelpad=labelpad, rotation=0)
    ax.set_xlabel(xLabel)

    # ************************************************
    #     Plot forces
    # ************************************************
    ax = axes[1, 0]
    yLabel = r"$\frac{L}{L(f_{ext}=0)}$"  # Lift
    nondim = dynLift[0]  # nondimensionalize by the static value
    ax.plot(fExtSweep, dynLift / nondim, c=cm[0])
    ax.set_ylabel(yLabel, labelpad=labelpad, rotation=0)
    ax.set_xlabel(xLabel)

    ax = axes[1, 1]
    yLabel = r"$\frac{M_y}{M_y(f_{ext}=0)}$"  # Moment
    nondim = dynMoment[0]  # nondimensionalize by the static value
    ax.plot(fExtSweep, dynMoment / nondim, c=cm[0])
    ax.set_ylabel(yLabel, labelpad=labelpad, rotation=0)
    ax.set_xlabel(xLabel)

    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)

    fig.suptitle("Frequency response spectra")

    return fig, axes


def plot_mode_shapes(fig, axes, y, nModes: int, modeShapes: dict, modeFreqs: dict, ls="-", nshift=12):
    """
        Plot the mode shapes for the structural and wet modes

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

        # Normalize by maximum value, if negative then flip sign
        maxValList = [np.max(abs(structBM[ii, :])), np.max(abs(structTM[ii, :]))]
        maxValListNoabs = [np.max((structBM[ii, :])), np.max((structTM[ii, :]))]
        argmax = np.argmax(maxValList)
        maxVal = maxValList[argmax]
        if maxVal != maxValListNoabs[argmax]:
            maxVal *= -1

        # maxVal = np.max(abs(structBM[ii, :]))
        # maxValNoabs = np.max((structBM[ii, :]))
        # if maxVal != maxValNoabs:
        #     maxVal *= -1.0

        structBM[ii, :] /= maxVal

        # ax.plot(eta, structBM[ii, :], label=bendLabel, ls=ls, c=color)
        # ax.plot(eta, structTM[ii, :], label=twistLabel, ls=ls[1], c=color)
        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, structBM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, c=ccm[ii])
        # ax.annotate(
        #     f"Mode {ii+1} {labelString}",
        #     xy=(eta[-1 - nshift], structBM[ii, -1 - nshift]),
        #     c=ccm[ii],
        #     bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
        #     va="top",
        #     xytext=(0, -2),
        #     textcoords="offset points",
        #     size=legfs,
        # )

    ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad)
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

        # Normalize by maximum value, if negative then flip sign
        maxValList = [np.max(abs(structBM[ii, :])), np.max(abs(structTM[ii, :]))]
        maxValListNoabs = [np.max((structBM[ii, :])), np.max((structTM[ii, :]))]
        argmax = np.argmax(maxValList)
        maxVal = maxValList[argmax]
        if maxVal != maxValListNoabs[argmax]:
            maxVal *= -1

        # maxVal = np.max(abs(structTM[ii, :]))
        # maxValNoabs = np.max((structTM[ii, :]))
        # if maxVal != maxValNoabs:
        #     maxVal *= -1.0

        structTM[ii, :] /= maxVal

        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, structTM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, c=ccm[ii])

    ax.set_ylabel(twistLabel, rotation=0, labelpad=labelpad)
    ax.set_xlabel(r"$\widebar{y}$ [-]")

    # ---------------------------
    #   Wet modes
    # ---------------------------
    # --- Bend modes ---
    # First normalize the modes by max
    for ii in range(nModes):
        ax = axes[0, 1]

        # Normalize by maximum value, if negative then flip sign
        maxVal = np.max(abs(wetBM[ii, :]))
        maxValNoabs = np.max((wetBM[ii, :]))
        if maxVal != maxValNoabs:
            maxVal *= -1.0

        wetBM[ii, :] /= maxVal

        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, wetBM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, c=ccm[ii])
        ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad)
        # ax.annotate(
        #     f"Mode {ii+1} {labelString}",
        #     xy=(eta[-1 - nshift], wetBM[ii, -1 - nshift]),
        #     c=ccm[ii],
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

        maxVal = np.max(abs(wetTM[ii, :]))
        maxValNoabs = np.max((wetTM[ii, :]))
        if maxVal != maxValNoabs:
            maxVal *= -1.0

        wetTM[ii, :] /= maxVal

        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, wetTM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls, c=ccm[ii])

    ax.set_ylabel(twistLabel, rotation=0, labelpad=labelpad)
    ax.set_xlabel("$\\widebar{y}$ [-]")

    fig.suptitle("Normalized mode shapes", fontsize=40)

    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
        ax.set_ylim([-1.1, 1.1])
    return fig, axes


def plot_vg_vf_rl(
    fig,
    axes,
    flutterSol: dict,
    ls="-",
    alpha=1.0,
    units="m/s",
    marker=None,
    showRLlabels=False,
    annotateModes=False,
    nShift=0,
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
    """

    # Sort keys to get consistent plotting
    sortedModesNumbers = sorted(flutterSol.keys(), key=int)

    flutterColor = "magenta"

    # Hardcoded settings
    xytext = (-5, 6)
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
    for ii, key in enumerate(sortedModesNumbers):
        iic = ii % len(cm)  # color index

        # --- Units for labeling ---
        vSweep = flutterSol[key]["U"]
        gSweep = flutterSol[key]["pvals_r"] * hz2rad
        if units == "kts":
            vSweep = 1.94384 * np.array(vSweep)  # kts
        elif units == "m/s":
            pass
        else:
            raise ValueError(f"Unsupported units: {units}")

        try:  # Plot only if the data exists
            ax.plot(
                vSweep,
                gSweep,
                ls=ls,
                c=cm[iic],
                label=f"Mode {key}",
                marker=marker,
                alpha=alpha,
            )
            # ax.scatter(vSweep, gSweep, color=(cm[iic]), marker=marker)
            start = np.array([vSweep[0 + nShift], gSweep[0 + nShift]])
            end = np.array([vSweep[-1], gSweep[-1]])
            # --- Label mode number on the line ---
            if annotateModes:
                ax.annotate(
                    f"Mode {ii+1}",
                    xy=(start[0], start[1]),
                    ha="left",
                    # xy=(end[0], end[1]),
                    c=cm[iic],
                    fontsize=legfs,
                    xytext=xytext,
                    textcoords="offset points",
                    # bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                )
            yticks.append(gSweep[end])
        except Exception:
            emptyModes.append(key)
    print(f"Empty modes: {emptyModes}")
    ax.set_ylim(top=10)
    ax.set_ylabel(gLabel, rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$g$", pad=labelpad)
    ax.set_xlabel(xlabel)
    # --- Put flutter boundary on plot ---
    ax.axhline(
        y=0.0,
        # label="Flutter boundary",
        c=flutterColor,
        ls="--",
        # path_effects=[patheffects.withTickedStroke()], # ugly
    )
    ax.annotate(
        "Hydroelastic instability", xy=(0.5, 0.9), ha="center", xycoords="axes fraction", size=legfs, color=flutterColor
    )
    # ax.set_yticks(yticks + [0])
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
            ax.plot(vSweep, fSweep, ls=ls, c=cm[iic], label=f"Mode {key}", marker=marker, alpha=alpha)
            # ax.scatter(vSweep, fSweep, c=(cm[iic]), marker=marker)
            start = np.array([vSweep[0], fSweep[0]])
            end = np.array([vSweep[-1], fSweep[-1]])
            # --- Label mode number on the line ---
            # check if mode is bifurcated and then alternate the label position
            if fSweep[0] < 1e-2:
                if bifCtr % 2 == 0:
                    va = "bottom"
                else:
                    xytext = (-5, -3)
                    va = "top"
                ha = "right"
                bifCtr += 1
            else:
                ha = "left"
                va = "bottom"

            if annotateModes:
                ax.annotate(
                    f"Mode {ii+1}",
                    xy=(start[0], start[1]),
                    ha=ha,
                    va=va,
                    c=cm[iic],
                    fontsize=legfs,
                    xytext=xytext,
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                )
        except Exception:
            continue

    ax.set_ylabel(fLabel, rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$f$", pad=labelpad)
    # ax.set_xlim(vSweep[0] * 0.99, vSweep[-1] * 1.01)
    ax.set_xlabel(xlabel)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

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
            ax.plot(gSweep, fSweep, ls=ls, c=cm[iic], label=f"Mode {key}", marker=marker, alpha=alpha)
            # ax.scatter(gSweep, fSweep, c=(cm[iic]), marker=marker)
            start = np.array([gSweep[0], fSweep[0]])
            end = np.array([gSweep[-1], fSweep[-1]])
            # ax.plot(start[0], start[1], marker="o", markersize=markerSize, c=cm[iic], markeredgecolor="gray")
            # ax.plot(end[0], end[1], marker="^", markersize=markerSize, c=cm[iic], markeredgecolor="gray")

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
                    c=cm[iic],
                    fontsize=legfs,
                    xytext=(5, 5),
                    textcoords="offset points",
                    # bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                )
            if showRLlabels:  # show R-L speed labels
                ax.annotate(
                    f"{vSweep[0]:.1f}{units}",
                    xy=(start[0], start[1]),
                    c=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, -5),
                    textcoords="offset points",
                    # bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                    va="top",
                )
                ax.annotate(
                    f"{vSweep[-1]:.1f}{units}",
                    xy=(end[0], end[1]),
                    c=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, -5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", ec="white", linewidth=0, fc="white", alpha=0.5),
                    va="top",
                )

            # --- Add arror pointing in speed ---
            nmid = int(len(vSweep) // 4)
            ax.annotate(
                "",
                xytext=(np.array([gSweep[-nmid - 1], fSweep[-nmid - 1]])),  # arrow start
                xy=(np.array([gSweep[-nmid], fSweep[-nmid]])),  # arrow end
                arrowprops=dict(arrowstyle="-|>", shrinkA=2, color=cm[iic], alpha=0.5),
            )

            yticks.append(fSweep[0])
        except Exception:
            continue

    ax.set_ylabel(fLabel, rotation=0, labelpad=labelpad)
    ax.set_title("Root locus", pad=labelpad)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)
    ax.set_xlabel(gLabel)
    # ax.set_yticks(yticks)

    # --- Put flutter boundary on plot ---
    # ax.set_xlim(right=10)
    ax.axvline(
        x=0.0,
        # label="Flutter boundary",
        c=flutterColor,
        ls="--",
        # path_effects=[patheffects.withTickedStroke()], # ugly
    )
    ax.annotate(
        "Hydroelastic\ninstability", xy=(0.85, 0.5), ha="left", xycoords="axes fraction", size=legfs, color=flutterColor
    )

    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)

    return fig, axes


def plot_dlf(fig, axes, flutterSol: dict, semichord: float, sweepAng: float, ls="-", alpha=1.0, units="m/s", nShift=0):
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
            ax.plot(vSweep, dlf, ls=ls, c=cm[iic], label=f"Mode {key}", alpha=alpha)
            start = np.array([vSweep[0 + nShift], dlf[0 + nShift]])
            end = np.array([vSweep[-1], dlf[-1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(start[0], start[1]),
                ha="right",
                # xy=(end[0], end[1]),
                c=cm[iic],
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
        c=flutterColor,
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

    ax.set_ylabel(yLabel, rotation=0, labelpad=labelpad)
    ax.set_xlabel(xLabel)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

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
            ax.loglog(kSweep, dlf, ls=ls, c=cm[iic], label=f"Mode {key}", alpha=alpha)
            start = np.array([kSweep[0], dlf[0]])
            end = np.array([kSweep[-1], dlf[-1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(start[0], start[1]),
                ha="left",
                # xy=(end[0], end[1]),
                c=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
        except Exception:
            continue

    ax.set_ylabel(yLabel, rotation=0, labelpad=labelpad)
    ax.set_xlabel(xLabel)
    # ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False)

    plt.suptitle("Damping loss factor trends")

    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)

    return fig, axes


def pytecplot_plotmesh(args, fname: str):
    if not args.batch:
        tp.session.connect()

    lay = ntp.Layout()
