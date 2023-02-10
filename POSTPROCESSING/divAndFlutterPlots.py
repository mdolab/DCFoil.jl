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
import os
import sys
import copy
import json

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import matplotlib.patheffects as patheffects
# from matplotlib.offsetbox import AnchoredText

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots

# --- Enable niceplot colors as a list ---
# plt.style.use(niceplots.get_style())
# niceColors = niceplots.get_colors().values()
niceColors = sns.color_palette("tab10")
# niceColors = matplotlib.cm.get_cmap("tab20").colors
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Continuous colormap
niceColors = sns.color_palette("cool")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
ccm = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_mode_shapes(y, nModes: int, modeShapes: dict, modeFreqs: dict, ls: list):
    """
        Plot the mode shapes for the structural and wet modes
        # TODO: make 2x2 plot

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
    ls : list
        Line styles
    fname : str, optional
        Filename to save, by default None
    """
    # Check if we want to save

    eta = y / y[-1]  # Normalized spanwise coordinate

    # Create figure object
    labelpad = 40
    legfs = 15
    nrows = 2
    ncols = 2
    figsize = (9 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, constrained_layout=True, figsize=figsize)

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
        # maxValList = [np.max(abs(structBM[ii, :])), np.max(abs(structTM[ii, :]))]
        # maxValListNoabs = [np.max((structBM[ii, :])), np.max((structTM[ii, :]))]
        # argmax = np.argmax(maxValList)
        # maxVal = maxValList[argmax]
        # if maxVal != maxValListNoabs[argmax]:
        #     maxVal *= -1
        maxVal = np.max(abs(structBM[ii, :]))
        maxValNoabs = np.max((structBM[ii, :]))
        if maxVal != maxValNoabs:
            maxVal *= -1.0

        structBM[ii, :] /= maxVal

        # ax.plot(eta, structBM[ii, :], label=bendLabel, ls=ls[0], c=color)
        # ax.plot(eta, structTM[ii, :], label=twistLabel, ls=ls[1], c=color)
        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, structBM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls[0], c=ccm[ii])
    ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad)
    ax.set_title("Dry Modes")
    ax.legend(fontsize=legfs, labelcolor="linecolor", loc="center left", frameon=False, ncol=1, bbox_to_anchor=(1, 0.5))

    # --- Twist modes ---
    for ii in range(nModes):
        ax = axes[1, 0]

        maxVal = np.max(abs(structTM[ii, :]))
        maxValNoabs = np.max((structTM[ii, :]))
        if maxVal != maxValNoabs:
            maxVal *= -1.0

        structTM[ii, :] /= maxVal

        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, structTM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls[0], c=ccm[ii])

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
        ax.plot(eta, wetBM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls[0], c=ccm[ii])
        ax.set_ylabel(bendLabel, rotation=0, labelpad=labelpad)
    ax.set_title("Wet Modes")
    ax.legend(fontsize=legfs, labelcolor="linecolor", loc="center left", frameon=False, ncol=1, bbox_to_anchor=(1, 0.5))

    # --- Twist modes ---
    for ii in range(nModes):
        ax = axes[1, 1]

        maxVal = np.max(abs(wetTM[ii, :]))
        maxValNoabs = np.max((wetTM[ii, :]))
        if maxVal != maxValNoabs:
            maxVal *= -1.0

        wetTM[ii, :] /= maxVal

        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.plot(eta, wetTM[ii, :], label=f"Mode {ii+1} {labelString}", ls=ls[0], c=ccm[ii])

    ax.set_ylabel(twistLabel, rotation=0, labelpad=labelpad)
    ax.set_xlabel("$\\widebar{y}$ [-]")

    fig.suptitle("Normalized mode shapes", fontsize=40)

    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
        ax.set_ylim([-1, 1])
    return fig, axes


def plot_vg_vf_rl(
    vSweep, fSweep, gSweep, nModes: int, ls: list, units="m/s", marker=None, showRLlabels=False, modeList=None
):
    """
    Plot the V-g, V-f, and R-L diagrams

    Parameters
    ----------
    vSweep : 1d array
        velocity sweept [m/s]
    fSweep : 2d array
        flutter frequency sweep [Hz]
    gSweep : 2d array
        damping sweep [1/s]
    nModes : int
        number of modes to plot
    units : str, optional
        units for velocity, by default "m/s"
    marker : str, optional
        marker for line plots to see points, by default None
    showRLlabels : bool, optional
        show R-L speed labels, by default True
    modeList : list, optional
        list of modes to plot, by default None; 0-indexed
    """

    # Create figure object
    labelpad = 40
    legfs = 15
    fact = 1  # scale size
    figsize = (18 * fact, 13 * fact)
    xytext = (-5, 5)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", constrained_layout=True, figsize=figsize)
    flutterColor = "magenta"

    # --- Units for labeling ---
    if units == "kts":
        vSweep = 1.94384 * np.array(vSweep)  # kts
    elif units == "m/s":
        pass
    else:
        raise ValueError(f"Unsupported units: {units}")
    xlabel = "$U_{\infty}$ " + f"[{units}]"

    if modeList is None:
        modeList = np.arange(nModes)
    # ************************************************
    #     V-g diagram
    # ************************************************
    ax = axes[0, 0]
    for ii in modeList:
        iic = ii % len(cm)  # color index
        if all(gSweep[ii, :] == 0):
            # Don't plot if all zeros
            print(f"Mode {ii+1} is all zeros")
            pass
        else:
            ax.plot(vSweep, gSweep[ii, :], ls=ls[0], c=cm[iic], label=f"Mode {ii+1}", marker=marker)
            start = np.array([vSweep[0], gSweep[ii, 0]])
            end = np.array([vSweep[-1], gSweep[ii, -1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                # xy=(start[0], start[1]),
                # ha="right",
                xy=(end[0], end[1]),
                c=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
    ax.set_ylim(top=10)
    ax.set_ylabel("$g$ [1/s]", rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$g$")
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
    X = [vSweep[0], vSweep[-1] * 20]
    Y = [0.0, 0.0]
    ax.fill_between(X, Y, 10, color=flutterColor, alpha=0.2)
    ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False, ncol=nModes // 4)

    # ************************************************
    #     V-f diagram
    # ************************************************
    ax = axes[1, 0]
    for ii in modeList:
        iic = ii % len(cm)  # color index
        if all(gSweep[ii, :] == 0):
            # Don't plot if all zeros
            pass
        else:
            ax.plot(vSweep, fSweep[ii, :], ls=ls[0], c=cm[iic], label=f"Mode {ii+1}", marker=marker)
            start = np.array([vSweep[0], fSweep[ii, 0]])
            end = np.array([vSweep[-1], fSweep[ii, -1]])
            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(start[0], start[1]),
                ha="right",
                c=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
            # if ii == 3:
            #     breakpoint()
    ax.set_ylabel("$f$ [Hz]", rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$f$")
    ax.set_xlim(vSweep[0] * 0.99, vSweep[-1] * 1.01)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False, ncol=nModes // 4)

    # ************************************************
    #     Root-locus diagram
    # ************************************************
    markerSize = 8
    ax = axes[1, 1]
    for ii in modeList:
        iic = ii % len(cm)
        if all(gSweep[ii, :] == 0):
            # Don't plot if all zeros
            pass
        else:
            ax.plot(gSweep[ii, :], fSweep[ii, :], ls=ls[0], c=cm[iic], label=f"Mode {ii+1}", marker=marker)
            start = np.array([gSweep[ii, 0], fSweep[ii, 0]])
            end = np.array([gSweep[ii, -1], fSweep[ii, -1]])
            ax.plot(start[0], start[1], marker="o", markersize=markerSize, c=cm[iic], markeredgecolor="gray")
            ax.plot(end[0], end[1], marker="o", markersize=markerSize, c=cm[iic], markeredgecolor="gray")

            # Label mode number on the line
            ax.annotate(
                f"Mode {ii+1}",
                xy=(end[0], end[1]),
                ha="right",
                c=cm[iic],
                fontsize=legfs,
                xytext=xytext,
                textcoords="offset points",
            )
            if showRLlabels:  # show R-L speed labels
                ax.annotate(
                    f"{vSweep[0]:.1f}{units}",
                    xy=(start[0], start[1]),
                    c=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )
                ax.annotate(
                    f"{vSweep[-1]:.1f}{units}",
                    xy=(end[0], end[1]),
                    c=cm[iic],
                    fontsize=legfs * 0.8,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    ax.set_ylabel("$f$ [Hz]", rotation=0, labelpad=labelpad)
    ax.set_title("Root locus")
    ax.legend(fontsize=legfs * 0.5, labelcolor="linecolor", loc="best", frameon=False, ncol=nModes // 4)
    ax.set_xlabel("$g$ [1/s]")
    # --- Put flutter boundary on plot ---
    ax.set_xlim(right=10)
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
