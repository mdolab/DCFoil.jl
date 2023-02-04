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
from matplotlib.pyplot import cm


# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots


def plotModeShapes(y: np.ndarray, nModes: int, modeShapes: dict, modeFreqs: dict, ls: list, cm: list, fname=None):
    """
        Plot the mode shapes for the structural and wet modes


    Parameters
    ----------
    y : np.ndarray
        spanwise coordinate [m]
    nModes : int
        number of modes to plot
    modeShapes : dict
        stores dry and wet mode shapes
    modeFreqs : dict
        stores dry and wet natural frequencies
    ls : list
        Line styles
    cm : list
        Colors
    fname : str, optional
        Filename to save, by default None
    """
    # Check if we want to save
    dosave = not not fname

    eta = y / y[-1]  # Normalized spanwise coordinate

    # Create figure object
    labelpad = 40
    legfs = 15
    nrows = nModes
    ncols = 2
    figsize = (5 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, constrained_layout=True, figsize=figsize)

    # Unpack data
    structBM = modeShapes["structBM"]
    structTM = modeShapes["structTM"]
    wetBM = modeShapes["wetBM"]
    wetTM = modeShapes["wetTM"]
    wetNatFreqs = modeFreqs["wetNatFreqs"]
    structNatFreqs = modeFreqs["structNatFreqs"]

    # --- Structural modes ---
    # First normalize the modes by max
    for ii in range(nModes):
        ax = axes[ii, 0]

        # Normalize by maximum value, if negative then flip sign
        maxValList = [np.max(abs(structBM[ii, :])), np.max(abs(structTM[ii, :]))]
        maxValListNoabs = [np.max((structBM[ii, :])), np.max((structTM[ii, :]))]
        argmax = np.argmax(maxValList)
        maxVal = maxValList[argmax]
        if maxVal != maxValListNoabs[argmax]:
            maxVal *= -1

        structBM[ii, :] /= maxVal
        structTM[ii, :] /= maxVal

        ax.plot(eta, structBM[ii, :], label=f"OOP bending", ls=ls[0])
        ax.plot(eta, structTM[ii, :], label=f"Twisting", ls=ls[1])
        labelString = f"({structNatFreqs[ii]:.2f}" + " Hz)"
        ax.set_ylabel(f"Mode {ii+1}\n{labelString}", rotation=0, labelpad=labelpad)
        if ii == 0:
            ax.set_title("Dry Modes")
            ax.legend(fontsize=legfs, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

    ax.set_xlabel("$\\bar{y}$ [-]")

    # --- Wet modes ---
    # First normalize the modes by max
    for ii in range(nModes):
        ax = axes[ii, 1]

        # Normalize by maximum value, if negative then flip sign
        maxValList = [np.max(abs(wetBM[ii, :])), np.max(abs(wetTM[ii, :]))]
        maxValListNoabs = [np.max((wetBM[ii, :])), np.max((wetTM[ii, :]))]
        argmax = np.argmax(maxValList)
        maxVal = maxValList[argmax]
        if maxVal != maxValListNoabs[argmax]:
            maxVal *= -1

        wetBM[ii, :] /= maxVal
        wetTM[ii, :] /= maxVal

        ax.plot(eta, wetBM[ii, :], label=f"OOP bending", ls=ls[0])
        ax.plot(eta, wetTM[ii, :], label=f"Twisting", ls=ls[1])
        labelString = f"({wetNatFreqs[ii]:.2f}" + " Hz)"
        ax.set_ylabel(f"Mode {ii+1}\n{labelString}", rotation=0, labelpad=labelpad)
        if ii == 0:
            ax.set_title("Wet Modes")
            ax.legend(fontsize=legfs, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

    ax.set_xlabel("$\\bar{y}$ [-]")

    fig.suptitle("Mode Shapes", fontsize=40)

    plt.show(block=(not dosave))
    for ii in range(2):
        for ax in axes[:, ii]:
            niceplots.adjust_spines(ax, outward=True)
            ax.set_ylim([-1, 1])
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()

    return fig, axes


def plotVgVf(vSweep, fSweep, gSweep, ls: list, fname=None):
    """
    Plot the V-g and V-f diagrams

    Parameters
    ----------
    vSweep : 1d array
        velocity sweept [m/s]
    fSweep : 2d array
        flutter frequency sweep [Hz]
    gSweep : 2d array
        damping sweep [1/s]
    fname : str, optional
        filename, by default None
    """
    dosave = not not fname

    # Create figure object
    labelpad = 40
    legfs = 15
    fig, axes = plt.subplots(nrows=2, sharex=True, constrained_layout=True, figsize=(8, 10))

    nModes = fSweep.shape[0]  # number of modes
    xlabel = "$V$ [m/s]"
    colors = cm.rainbow(np.linspace(0, 1, nModes))

    # --- V-g ---
    ax = axes[0]
    for ii in range(nModes):
        ax.plot(vSweep, gSweep[ii, :], ls=ls[0], c=colors[ii], label=f"Mode {ii+1}")
    ax.set_ylim(top=10)
    ax.set_ylabel("$g$ [1/s]", rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$g$")
    ax.legend(fontsize=legfs, labelcolor="linecolor", loc="best", frameon=False, ncol=nModes // 4)
    ax.set_xlabel(xlabel)

    # --- V-f ---
    ax = axes[1]
    for ii in range(nModes):
        ax.plot(vSweep, fSweep[ii, :], ls=ls[0], c=colors[ii], label=f"Mode {ii+1}")
    # ax.set_xlim(left=vSweep[0]*0.75)
    ax.set_ylabel("$f$ [Hz]", rotation=0, labelpad=labelpad)
    ax.set_title("$V$-$f$")
    ax.legend(fontsize=legfs, labelcolor="linecolor", loc="best", frameon=False, ncol=nModes // 4)
    ax.set_xlabel(xlabel)

    # plt.tight_layout()
    plt.show(block=(not dosave))
    for ax in axes:
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()

    return fig, axes


def plotRL():
    """
    Plot the root-locus plot
    """
