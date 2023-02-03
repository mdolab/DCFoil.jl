import argparse
import pickle
import numpy as np
import os
from pprint import pprint as pp
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

import niceplots

parser = argparse.ArgumentParser()
parser.add_argument("histFile",  nargs="+", type=str, default='hist.pkl')
parser.add_argument("--basefn",  type=str, default="", help="Filename prepended to plot type")
parser.add_argument("--alpha", nargs="+", type=float, default=None)
parser.add_argument("--lineStyle", action='store_true', default=False)
parser.add_argument("--lineWidth", nargs="+", type=float, default=None)
parser.add_argument("--boundaryFile", type=str, default='')
parser.add_argument("--crossingFile", type=str, default='')
parser.add_argument("--single", action='store_true', default=False)
parser.add_argument("--legend", type=str, default=None, choices=["normal", "normal-best", "draggable", "outside", "autoplace"])
parser.add_argument("--draggableLegends", action='store_true', default=False)
parser.add_argument("--yminmax", nargs=2, type=float, default=[])
parser.add_argument("--ylimFile", type=str, default="", help="Filename containing the ylim for various figures.\n\
Example: \n\
0 -4.0 6\n\
1 -2.0 2\n\
4 -43.0 22\n")
parser.add_argument("--xlimFile", type=str, default="", help="Filename containing the xlim for various figures.")


# Figure options
parser.add_argument("--figOutDir", type=str, default="./figures")
parser.add_argument("--showFig", action='store_true', default=False)

args = parser.parse_args()


# ------------------------ PLOTTING INPUT ------------------------

hideXAxis = False


# Load the stylesheet
plt.style.use(['latex','presentation'])

# Set the default colormap to use
colors = matplotlib.cm.get_cmap("tab20").colors
mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

# Set markercolor and sizes
markerColor = "black"
markerSize = 5


# ------------------------ HELPER FUNCTIONS ------------------------
# Load the data we need
def load_python_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_tecplot_file(filename):
    return np.loadtxt(filename, skiprows=3, ndmin=2)


# ------------------------ PARSE ARGUMENTS ------------------------
# Create the output folder
if not os.path.exists(args.figOutDir):
    os.makedirs(args.figOutDir)


# Load the exported data
dlmSol = []
for histFile in args.histFile:
    dlmSol.append(load_python_obj(histFile))

# Load boundary file
swd = None
if args.boundaryFile:
    if os.path.splitext(args.boundaryFile)[1] == ".dat":
        swd = load_tecplot_file(args.boundaryFile)

# Load crossing file
if args.crossingFile:
    if os.path.splitext(args.crossingFile)[1] == ".dat":
        dfp = load_tecplot_file(args.crossingFile)

# Determine alpha for plot
if args.alpha and ( len(args.alpha) == len(args.histFile) ):
#if args.alpha:
    # Read in what user supplied
    alpha = args.alpha

    # Create a scaled alpha
    #alpha = []
    #for i in range(len(args.histFile)):
    #    alpha.append(1.0/(i+1))
else:
    alpha = len(args.histFile)*[1.0]

# Line style
#if args.lineStyle and ( len(args.lineStyle) == len(args.histFile) ):
if args.lineStyle:
    # Load the user input
    #lineStyle = args.lineStyle
    lineStyle = ["-", "--", "."]
else:
    lineStyle = []
    lineStyle = len(args.histFile)*["-"]

# Line thickness
if args.lineWidth and ( len(args.lineWidth) == len(args.histFile) ):
    # Load the user input
    lineWidth = args.lineWidth
else:
    # Get default valus which come from the current set value (loaded above)
    lw = mpl.rcParams["lines.linewidth"]
    lineWidth = []
    lineWidth = len(args.histFile)*[lw]

#print alpha
#print lineStyle
#print lineWidth

# Y limits
if args.ylimFile:
    # Load the file
    tmp = np.loadtxt(args.ylimFile)
    ylimRanges = {}
    for i in range(tmp.shape[0]):
        ylimRanges[int(tmp[i,0])] = tmp[i,1:]


# X limits
if args.xlimFile:
    # Load the file
    tmp = np.loadtxt(args.xlimFile)
    xlimRanges = {}
    for i in range(tmp.shape[0]):
        xlimRanges[int(tmp[i,0])] = tmp[i,1:]



# Plotting
yminmax = args.yminmax


# --------------------- PLOT FUNCTIONS ----------------------------
# def annotate(axes,boxes,labels,data,**kwargs):
#     #slide should be relevant edge of bbox - e.g. (0,0) for left, (0,1) for bottom ...
#     try: slide = kwargs.pop("slide")
#     except KeyError: slide = None
#     try:
#         xytexts = kwargs.pop("xytexts")
#         xytext  = xytexts
#     except KeyError:
#         xytext = (0,2)
#         xytexts = None
#     try: boxes = kwargs.pop("boxes")
#     except KeyError: boxes = list()
#     pixel_diff = 1
#     newlabs = []
#     for i in range(len(labels)):
#         try:
#             len(xytexts[i])
#             xytext = xytexts[i]
#         except TypeError: pass

#         a = axes.annotate(labels[i],xy=data[i],textcoords='offset pixels',
#                                     xytext=xytext,**kwargs)
#         newlabs.append(a)
#     plt.draw()
#     for i in range(len(labels)):
#         cbox = a.get_window_extent()
#         if slide is not None:
#             direct  = int((slide[0] - 0.5)*2)
#             current = -direct*float("inf")
#             arrow = False
#             while True:
#                 overlaps = False
#                 count = 0
#                 for box in boxes:
#                     if cbox.overlaps(box):
#                         if direct*box.get_points()[slide] > direct*current:
#                             overlaps = True
#                             current =  box.get_points()[slide]
#                             shift   = direct*(current - cbox.get_points()[1-slide[0],slide[1]])
#                 if not overlaps: break
#                 arrow = True
#                 position = array(a.get_position())
#                 position[slide[1]] += shift * direct * pixel_diff
#                 a.set_position(position)
#                 plt.draw()
#                 cbox = a.get_window_extent()
#                 x,y =  axes.transData.inverted().transform(cbox)[0]
#             if arrow:
#                 axes.arrow(x,y,data[i][0]-x,data[i][1]-y,head_length=0,head_width=0)
#         boxes.append(cbox)
#     plt.draw()
#     return boxes

# def autoPlaceLegend(ax, xLocRelLoc=None, color_on=True, **kwargs):
#     """
#     This function tries to automatically place legend text
#     close to the respective line and tries to minimize the
#     possible overlap with other lines and text. This function uses
#     a the adjustText module.

#     Inputs:
#         ax : single or array of matplotlib axes
#         xLocRelLoc :  list or scalar
#             Relative location (between 0 and 1) on the x axis
#             where to *try* to place the legend text. List has to be the
#             same length as the number of lines in plot.
#             If not specified random location will be chosen.
#         color_on : bool
#             Use the color on the line to color the legend text.
#         kwargs : optional keyword arguments
#             This is intended to tweak the adjustText tool (passed onwards)
#     """
#     nLines = len(ax.lines)
#     texts = []


#     # If using a scalar generate a full list needed for placing legends
#     if type(xLocRelLoc) is float:
#         xLocRelLoc = nLines * [xLocRelLoc]


#     # Loop over the lines in the axes to get legend text and color
#     for i, line in enumerate(ax.lines):

#         # Set the starting coordinates of the label
#         # First extract the xy data (numpy Nx2 array) for a given line
#         coords = line.get_xydata()
#         if xLocRelLoc is not None:
#             idx = int(xLocRelLoc[i] * coords[:,0].shape[0])
#         else:
#             # Select randomly one point from the dataset to place the text
#             idx = np.random.randint(0,coords.shape[0])

#         label = line.get_label()
#         # Get the color of each line to set the label color as the same
#         if color_on:
#             color = line.get_color()
#         else:
#             color = 'k'

#         # Create the text object and place it at this random location on line
#         t = ax.text(coords[idx,0], coords[idx,1], label, ha="center", va="center", color=color)
#         texts.append(t)

#     # Now we have all the text and now we want to place them


#     # Use the line data to generate some points that we can use to repel the actual text from the line itself
#     #from scipy import interpolate
#     #f = interpolate.interp1d(coords[:,0], coords[:,1])
#     #x = np.linspace(min(coords[:,0]), max(coords[:,0]), 500)
#     #y = f(x)
#     #adjust_text(texts, x=x, y=y, ax=ax)

#     # Try to repel the text from other lines, does not work since it uses a bounding box for objects!
#     #adjust_text(texts, ax=ax, add_objects=ax.lines)

#     # Now use all the coordinates from the lines as repelling points to avoid crossing
#     #coordsAllNP = np.array(coordsAll)
#     #x = coordsAllNP[:,:,0].flatten()
#     #y = coordsAllNP[:,:,1].flatten()


#     #adjust_text(texts, x=x, y=y, ax=ax, **kwargs)

#     adjust_text(texts, ax=ax)

#     return texts



def plotDampingAndFreq(ax,d,x,y,xlabel=None,ylabel=None,alpha=None,lineStyle=None,lineLabel=True,lineWidth=None,figID=None):


    # Fallback values, although all values should be set
    if alpha is None:
        alpha = 1.0

    if lineStyle is None:
        lineStyle = "-"

    if lineWidth is None:
        lineWidth = 1.0


    # Sort the keys such that we get consistent plotting coloring scheme
    sortedModesNumbers = sorted(d.keys(), key=int)

    # Plot damping and freq vals
    dataProps = {}
    for key in sortedModesNumbers:

        xx = d[key][x]
        if x == "dynp":
            # Plot the kPa instead of Pa
            xx = xx/1000.0

        label=" "
        if lineLabel:
            # Set the line label to the mode number
            label="Mode {0}".format(key)

        # 20200142 HACK: to have matching modes and colors
        # if int(key) == 5:
        #     color = "C{0:d}".format(int(key)+1-1)
        # elif int(key) == 6:
        #     color = "C{0:d}".format(int(key)-1-1)
        # else:
        #     color = "C{0:d}".format(int(key)-1)
        # print "Mode", key, color
        # ax.plot(xx, d[key][y], linestyle=lineStyle, alpha=alpha, linewidth=lineWidth, label=label, color=color)

        ax.plot(xx, d[key][y], linestyle=lineStyle, alpha=alpha, linewidth=lineWidth, label=label)

        # Set xy labels
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Label start and finish of the root locus
        if figID == "eval-root-loci" or figID == "eval-root-loci-nondim":
            start = [d[key][x][0], d[key][y][0]]
            end = [d[key][x][-1], d[key][y][-1]]
            ax.plot(start[0], start[1], color=markerColor, marker="o", markersize=markerSize, fillstyle="none")
            ax.plot(end[0], end[1], color=markerColor, marker="o", markersize=markerSize)


# Define the figure properties and initialize list
# fig = {"f":"handle",
#         "axarr":"handle",
#         "fSaveName":"fname+ext"}
figs = []

# Set the number of plots to
if args.single:

    xvar = ["dynp", "dynp", "U", "U",
            "dynp", "dynp", "U", "U",
            "dynp", "U",
            "pvals_r", "p_r"]
    #xvar = ["U", "U", "U", "U", "U", "pvals_r", "p_r"]
    yvar = ["pvals_r", "pvals_i", "pvals_r", "pvals_i",
            "p_r", "p_i", "p_r", "p_i",
            "pmG", "pmG",
            "pvals_i", "p_i"]
    xlab = [r"$q$ [kPa]", r"$q$ [kPa]", r"$U$ [m/s]", r"$U$ [m/s]",
            r"$q$ [kPa]", r"$q$ [kPa]", r"$U$ [m/s]", r"$U$ [m/s]",
            r"$q$ [kPa]", r"$U$ [m/s]",
            r"$\sigma$ [rad/s]", r"$g$"]
    #xlab = [r"$U$ [m/s]", r"$U$ [m/s]", r"$U$ [m/s]", r"$U$ [m/s]", r"$U$ [m/s]", r"$\sigma$ [rad/s]", r"$g$"]
    ylab = [r"$\sigma$ [rad/s]", r"$\omega$ [rad/s]", r"$\sigma$ [rad/s]", r"$\omega$ [rad/s]",
            r"$g$", r"$k$", r"$g$", r"$k$",
            r"$\sigma - G$ [rad/s]", r"$\sigma - G$ [rad/s]",
            r"$\omega$ [rad/s]", r"$k$"]
    fname = ["dampingVsq", "frequencyVsq", "dampingVsU", "frequencyVsU",
             "dampingVsq-nondim", "frequencyVsq-nondim", "dampingVsU-nondim", "frequencyVsU-nondim",
             "dampingVsq-pmG", "dampingVsU-pmG",
             "eval-root-loci", "eval-root-loci-nondim"]

    f = []
    ax = []
    # Plot modes
    for n in range(len(xvar)):
        fn, axn = plt.subplots(1, 1)
        fn.set_size_inches(10,8, forward=True)


        lineLable = True
        for k,d in enumerate(dlmSol):
            plotDampingAndFreq(axn,d,xvar[n],yvar[n],xlabel=xlab[n],ylabel=ylab[n],alpha=alpha[k],lineStyle=lineStyle[k],lineLabel=lineLable,lineWidth=lineWidth[k],figID=fname[n])
            # reset color
            axn.set_prop_cycle(None)
            # Only the first set should have labels
            lineLable = False


        # Store for later
        #fig = {"f":fn, "axarr":axn, "fName":[args.basefn + fname[n]+".png",args.basefn + fname[n]+".pdf"]}
        fig = {"f":fn, "axarr":axn, "fName":[args.basefn + fname[n]+".png"], "description":fname[n]}
        figs.append(fig)
        #f.append(fn)
        #ax.append(axn)

    # Plot boundary
    if args.boundaryFile:
        # Always scale to kPa
        xx = swd[:,0] / 1000.0
        if swd.shape[0] > 2:
            # Only plot the line if we have the actual safety window otherwise its just a line
            figs[0]["axarr"].plot(xx, swd[:,1], color="black", label=r"$G(q)$", alpha=0.5)
        #ax[0].fill_between(s[:,0], s[:,1], 31, color="red", alpha=0.15)
        figs[0]["axarr"].fill_between(xx, swd[:,1], 31, color="red", alpha=0.15)

    # Plot crossing if any
    if args.crossingFile:
        for item in dfp:
            print item
            #ax[0].plot(item[0], item[1], "o", color="black")
            #figs[0]["axarr"].plot(item[0], item[1], "o", color="black", label=" ")
            xx = item[0]/1000.0 # For kPa
            figs[0]["axarr"].plot(xx, item[1], "o", color="black")


    # Do some cosmetic updates
    for fig in figs:
        # Adjust the spines Dumont style
        niceplots.adjust_spines(fig["axarr"], spines=["left", "bottom"])

        # Add draggable legends
        if args.legend == "normal":
            fig["axarr"].legend(loc="lower left")
        elif args.legend == "normal-best":
            fig["axarr"].legend(loc="best")
        elif args.legend == "draggable":
            niceplots.draggable_legend(fig["axarr"])
        elif args.legend == "outside":
            fig["axarr"].legend(bbox_to_anchor=(1.04,1), loc="upper left")
        elif args.legend == "autoplace":
            niceplots.auto_place_legend(fig["axarr"], rel_xloc=0.89)

        # Add the start and end points to legend if we have root loci figs

        if args.legend is not None and "eval-root-loci" in fig["description"]:
            handles, labels = fig["axarr"].get_legend_handles_labels()
            handles.append(mlines.Line2D([], [], linestyle="", color=markerColor, marker="o", markersize=markerSize, fillstyle="none", label="Start"))
            handles.append(mlines.Line2D([], [], linestyle="", color=markerColor, marker="o", markersize=markerSize, label="Stop"))
            fig["axarr"].legend(handles=handles)

        fig["f"].tight_layout()


    # Set limits
    if args.ylimFile:
        for key,val in ylimRanges.iteritems():
            figs[key]["axarr"].set_ylim([val[0],val[1]])
    # Set limits
    if args.xlimFile:
        for key,val in xlimRanges.iteritems():
            figs[key]["axarr"].set_xlim([val[0],val[1]])

    # Set limits
    if args.yminmax:
        figs[0]["axarr"].set_ylim([yminmax[0],yminmax[1]])
        figs[2]["axarr"].set_ylim([yminmax[0],yminmax[1]])




else:
    exportFileName = "rect-damping-frequncy"

    # Define the axes and figure
    f1, axarr1 = plt.subplots(2, 1)
    f1.set_size_inches(10,10, forward=True)

    # ------------------------ Figure 1 ------------------------

    # Plot damping and freq vals
    plotDampingAndFreq(axarr1[0],d,"U","pvals_r",xlabel=None,ylabel=r"$\sigma$ (rad/s)")
    plotDampingAndFreq(axarr1[1],d,"U","pvals_i",xlabel=r"$U$ [m/s]",ylabel=r"$\omega$ (rad/s)")

    # for k,v in d.iteritems():
    #     axarr1[0].plot(d[k]["U"], d[k]["pvals_r"], label=k)
    #     axarr1[0].set_ylabel(r"$\gamma$ (rad/s)")

    #     axarr1[1].plot(d[k]["U"], d[k]["pvals_i"], label=k)
    #     axarr1[1].set_xlabel(r"$V$ (m/s)")
    #     axarr1[1].set_ylabel(r"$\omega$ (rad/s)")


    if args.legend == "draggable":
        # Add draggable ledgends
        niceplots.draggable_legend(axarr1[0])
        niceplots.draggable_legend(axarr1[1])


    # Plot the safety window if relevant
    if args.boundaryFile:
        #axarr1[0].plot(s[:,0], s[:,1], color="red", alpha=0.2)
        axarr1[0].fill_between(s[:,0], s[:,1], 31, color="red", alpha=0.15)

    # Plot crossing if any
    if args.crossingFile:
        for item in dfp:
            axarr1[0].plot(item[0], item[1], "o", color="black")

    # Set limits
    #axarr1[0].set_xlim([0,20])
    axarr1[0].set_ylim([-30,ymax])
    #axarr1[1].set_xlim([0,20])

    # Adjust the spines Dumont style
    niceplots.adjust_spines(axarr1[0])
    niceplots.adjust_spines(axarr1[1])

    if hideXAxis:
        # Hide the x axis for the damping plot such that they share one axis
        axarr1[0].get_xaxis().set_ticks([])
        niceplots.adjust_spines(axarr1[0], off_spines=["top","right","bottom"])


    # Save figure
    f1.savefig(exportFileName + ".png")
    f1.savefig(exportFileName + ".pdf")



    # ------------------------ Figure 2 ------------------------
    # Define the axes and figure
    f2, axarr2 = plt.subplots(2, 1)
    f2.set_size_inches(10,10, forward=True)


    # Plot damping and freq vals
    plotDampingAndFreq(axarr2[0],d,"U","p_r",xlabel=None,ylabel=r"$g")
    plotDampingAndFreq(axarr2[1],d,"U","p_i",xlabel=r"$U$ [m/s]",ylabel=r"$k$")
    # for k,v in d.iteritems():
    #     axarr2[0].plot(d[k]["U"], d[k]["p_r"], label=k)
    #     axarr2[0].set_ylabel(r"$g$")

    #     axarr2[1].plot(d[k]["U"], d[k]["p_i"], label=k)
    #     axarr2[1].set_xlabel(r"$V$ (m/s)")
    #     axarr2[1].set_ylabel(r"$k$")

    if args.legend == "draggable":
        # Add draggable ledgends
        niceplots.draggable_legend(axarr2[0])
        niceplots.draggable_legend(axarr2[1])

    # Set limits
    #axarr2[0].set_xlim([0,20])
    #axarr2[0].set_ylim([-20,20])
    #axarr2[1].set_xlim([0,20])

    # Adjust the spines Dumont style
    niceplots.adjust_spines(axarr2[0])
    niceplots.adjust_spines(axarr2[1])

    if hideXAxis:
        # Hide the x axis for the damping plot such that they share one axis
        axarr2[0].get_xaxis().set_ticks([])
        niceplots.adjust_spines(axarr2[0], off_spines=["top","right","bottom"])

    # Save figure
    f2.savefig(exportFileName + "-nondim.png")
    f2.savefig(exportFileName + "-nondim.pdf")




# Save the plots. Loop over all figures
for fig in figs:
    # Save as many times as we have extensions
    for fName in fig["fName"]:
        fName = os.path.join(args.figOutDir,fName)
        print "Saving ... {0}".format(fName)
        fig["f"].savefig(fName)


# Draw the plots
if args.showFig:
    plt.show()