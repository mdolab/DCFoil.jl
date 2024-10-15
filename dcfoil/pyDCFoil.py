# --- Python 3.10 ---
"""
@File    :   pyDCFoil.py
@Time    :   2024/01/22
@Author  :   Galen Ng, Prof. Sicheng He
@Desc    :   Python interface to DCFoil containing two classes
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import time
from collections import OrderedDict
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from julia import Main, Pkg


class DCFOILWarning(object):
    """
    Format a warning message
    """

    def __init__(self, message):
        msg = "\n+" + "-" * 78 + "+" + "\n" + "| pyDCFOIL Warning: "
        i = 19
        for word in message.split():
            if len(word) + i + 1 > 78:  # Finish line and start new one
                msg += " " * (78 - i) + "|\n| " + word + " "
                i = 1 + len(word) + 1
            else:
                msg += word + " "
                i += len(word) + 1
        msg += " " * (78 - i) + "|\n" + "+" + "-" * 78 + "+" + "\n"
        print(msg)


class DCFOIL:
    def __init__(self, DVDictList: list, evalFuncs, options=None, debug=False):
        """
        Create the flutter solver class

        Parameters
        ----------
        DVDict : Dict
            design variable dictionary
        evalFuncs : list of strings
            Contains the names of the functions to be evaluated
        options : Dict, optional
            Solver options, by default None
        debug : bool, optional
            Developer mode in the python layer not to be confused with the debug option, by default False

        Raises
        ------
        DCFOILWarning
            Prettier warning message so you know it's from pyDCFOIL
        """
        startInitTime = time.time()

        # ************************************************
        #     Load DCFOIL module from Julia
        # ************************************************
        # TODO LINK: https://julialang.github.io/PackageCompiler.jl/stable/libs.html#libraries
        try:
            if debug:
                # THIS PART RUNS KINDA SLOWLY THE VERY FIRST TIME.
                # It will be faster when the package is in the registry (else statement)
                # Pull from local directory
                repoDir = Path(__file__).parent.parent
                Pkg.activate(f"{repoDir}")
                Main.include(f"{repoDir}/src/DCFoil.jl")
                Main.using(".DCFoil")
                DCFoil = Main.DCFoil
            else:
                # Pull from Julia package registry (online)
                Pkg.add("DCFoil")
                from julia import DCFoil
            self.DCFoil = DCFoil

        except Exception:
            raise DCFOILWarning("Could not load DCFOIL module from Julia")

        importTime = time.time()

        # Coordinates of the stick mesh
        self.LEcoords = None
        self.TEcoords = None
        self.LEconn = None
        self.TEconn = None
        # ************************************************
        #     Set options, DVs info
        # ************************************************
        defaultOptions = self._getDefaultOptions()

        # self.possibleDVs, self.dcfoilCostFunctions = self._getObjectivesAndDVs()

        setupTime = time.time()

        # if type(DVDictList) == list:
        self.DVDictList = DVDictList
        # else:
        #     DCFOILWarning("DVs must be a list of dictionaries")

        self.evalFuncs = evalFuncs

        self.solverOptions = {}
        # --- Set all solver options ---
        for key, val in defaultOptions.items():
            if key not in options:  # Use default
                self.solverOptions[key] = val
            else:
                self.solverOptions[key] = options[key]
        # self.solverOptions = options

        # --- Make output directory ---
        Path(self.solverOptions["outputDir"]).mkdir(parents=True, exist_ok=True)

        initTime = time.time()

        if self.solverOptions["printTiming"]:
            print("+--------------------------------------------------+")
            print("|")
            print("| Initialization Times:")
            print("|")
            print("| %-30s: %10.3f sec" % ("Import Time", importTime - startInitTime))
            # print("| %-30s: %10.3f sec" % ("Setup Time", setupTime - importTime))
            # print("| %-30s: %10.3f sec" % ("Initialize Time", initTime - setupTime))
            print("|")
            print("| %-30s: %10.3f sec" % ("Total Init Time", initTime - startInitTime))
            print("+--------------------------------------------------+")

    # ==============================================================================
    #                         Internal functions
    # ==============================================================================
    def setOption(self, name, value):
        """
        Set `solverOptions` Value
        """
        self.solverOptions[name] = value

    @staticmethod
    def _getDefaultOptions():
        defaultOptions = {
            # ---------------------------
            #   I/O
            # ---------------------------
            "name": "default",
            "debug": False,
            "printTiming": True,
            "outputDir": "./OUTPUT/",
            "writeTecplotSolution": True,
            # ---------------------------
            #   General appendage options
            # ---------------------------
            "gravityVector": [0.0, 0.0, -9.81],
            # ---------------------------
            #   Flow
            # ---------------------------
            "Uinf": 5.0,  # free stream velocity [m/s]
            "rhof": 1000.0,  # fluid density [kg/m³]
            "use_cavitation": False,
            "use_freeSurface": False,
            "use_ventilation": False,
            "use_dwCorrection": True,
            # ---------------------------
            #   Solver modes
            # ---------------------------
            # --- Static solve ---
            "run_static": False,
            # --- Forced solve ---
            "run_forced": False,
            # "fSweep": np.linspace(0.0, 1.0, 10),
            # --- p-k (Eigen) solve ---
            "run_modal": False,
            "run_flutter": False,
            "nModes": 3,  # Number of struct modes to solve for (starting)
            # "uRange": [1.0, 5.0],  # Range of velocities to sweep
            "maxQIter": 100,  # max dyn pressure iters
            "rhoKS": 80.0,
        }
        return defaultOptions

    def _getObjectivesAndDVs(self):
        """
        All possible objectives and design variables
        """
        iDV = OrderedDict()
        # iDV["theta"] = self.DCFoil.

        dcfoilCostFunctions = {
            "ksflutter": self.dcfoil,
            "wtip": self.dcfoil,
            "psitip": self.dcfoil,
            "cl": self.dcfoil,
            "cmy": self.dcfoil,
            "lift": self.dcfoil,
            "moment": self.dcfoil,
        }

        return iDV, dcfoilCostFunctions

    def _setAeroProblemData(self, aeroProblem):
        """
        Set the aeroproblem data
        """
        AP = aeroProblem

        if AP.chordRef is None:
            raise DCFOILWarning("'chordRef' must be set in the AeroProblem object")

    # ==============================================================================
    #                         Main functions
    # ==============================================================================
    def addMesh(self, gridFile:str):
        """
        Add component to mesh

        shape: (n_nodes, 3)
        """

        Grid = self.DCFoil.MeshIO.add_mesh(gridFile)
        LE_X, LE_conn, TE_X, TE_conn = Grid.LEMesh.T, Grid.LEConn.T, Grid.TEMesh.T, Grid.TEConn.T


        if self.LEcoords is None:
            # Set mesh and connectivity
            self.LEcoords = LE_X
            self.LEconn = LE_conn
            self.TEcoords = TE_X
            self.TEconn = TE_conn
        else:
            # Append to existing mesh
            self.LEcoords = np.vstack((self.LEcoords, LE_X))
            self.conn = np.vstack((self.LEconn, LE_conn + self.nnodes))
            self.TEcoords = np.vstack((self.TEcoords, TE_X))
            self.conn = np.vstack((self.TEconn, TE_conn + self.nnodes))

        # set new number of collocation nodes
        self.nnodes = self.LEcoords.shape[0]

    def solve(self):
        """
        Solve foil problem
        """

        DVDictList = self.DVDictList
        evalFuncs = self.evalFuncs

        solverOptions = self.solverOptions

        breakpoint()
        # TODO: WHY IS THERE A BUG HERE
        self.DCFoil.init_model(DVDictList, evalFuncs, solverOptions=solverOptions)
        SOLDICT = self.DCFoil.run_model(DVDictList, evalFuncs, solverOptions=solverOptions)

        self.SOLDICT = SOLDICT

        # if self.getOption("printTiming"):
        #     print("+-------------------------------------------------+")
        #     print("|")
        #     print("| Solution Timings:")
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Set AeroProblem Time", t1 - startCallTime))
        #     print("| %-30s: %10.3f sec" % ("Solution Time", solTime))
        #     print("| %-30s: %10.3f sec" % ("Write Solution Time", writeSolutionTime - t2))
        #     print(
        #         "| %-30s: %10.3f sec"
        #         % (
        #             "Stability Parameter Time",
        #             stabilityParameterTime - writeSolutionTime,
        #         )
        #     )
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Total Call Time", stabilityParameterTime - startCallTime))
        #     print("+--------------------------------------------------+")

    def evalFunctions(self, aeroProblem, funcs, evalFuncs=None):
        """
        Evaluate desired functions in 'evalFuncs' and add
        them to the dictionary 'funcs'.

        Parameters
        ----------
        aeroProblem : :class:`~baseclasses:baseclasses.problems.pyAero_problem.AeroProblem`
            The aerodynamic problem to to get the solution for

        funcs : dict
            Dictionary into which functions are saved

        evalFuncs : iterable object containing strings, optional
            If not None, use these functions to evaluate.

        """
        startEvalTime = time.time()

        # Set the AP
        self.setAeroProblem(aeroProblem)

        aeroProblemTime = time.time()

        if evalFuncs is None:
            evalFuncs = sorted(self.curAP.evalFuncs)

        costFuncs = self.DCFoil.evalFuncs(self.SOLDICT, self.DVDictList, evalFuncs, self.solverOptions)
        # Convert costFuncs to a dictionary to fill 'funcs'

        for key, val in costFuncs.items():
            if key in evalFuncs:
                funcs[key] = val

        self.costFuncs = costFuncs

        getSolutionTime = time.time()

        if self.solverOptions["printTiming"]:
            print("+---------------------------------------------------+")
            print("|")
            print("| Function Timings:")
            print("|")
            print("| %-30s: %10.3f sec" % ("Function AeroProblem Time", aeroProblemTime - startEvalTime))
            print("| %-30s: %10.3f sec" % ("Function Evaluation Time", getSolutionTime - aeroProblemTime))
            print("|")
            print("| %-30s: %10.3f sec" % ("Total Function Evaluation Time", getSolutionTime - startEvalTime))
            print("+--------------------------------------------------+")

        return costFuncs

    def evalFunctionsSens(self, aeroProblem, funcsSens, evalFuncs=None):
        """
        Evaluate sensitivity information of desired functions
        and add them to dictionary 'funcsSens'.

        Parameters
        ----------
        funcsSens : dict
            Dictionary into which the function derivatives are saved
        evalFuncs : iterable object containing strings, optional
            The additional functions the user wants returned that are
            not already defined in the aeroProblem, by default None
        """

        startEvalSensTime = time.time()

        self.setAeroProblem(aeroProblem)

        if evalFuncs is None:
            evalFuncs = sorted(self.evalFuncs)

        # Determine all the design variable sizes
        DVGeo = self.DVGeo
        nGeoDV, nStructDV = self._getDVSizes(DVGeo)

        # ************************************************
        #     DCFoil sensitivity
        # ************************************************
        for obj in self.curAP.evalFuncs:

            self.Xb = self.DCFoil.evalFuncsSens(
                self.SOLDICT, self.DVDictList, evalFuncs, self.solverOptions, mode="ADJOINT"
            )

            # FFD sensitivities
            self.evalFFDSens()

            # Now add derivatives to funcsSens
            key = self.curAP.name + "_%s" % obj
            funcsSens[key] = {}

            # ============================
            #   Geometric Variables
            # ============================
            funcsSens[key].update(self.dIdx_geo_total)

        # if self.getOption("printTiming") and self.comm.rank == 0:
        #     print("+--------------------------------------------------+")
        #     print("|")
        #     print("| Adjoint Times:")
        #     print("|")
        #     for f in evalFuncs:
        #         print(
        #             "| %-30s: %10.3f sec"
        #             % (
        #                 "Adjoint Solve Time - %s" % (f),
        #                 adjointEndTime[f] - adjointStartTime[f],
        #             )
        #         )
        #         print(
        #             "| %-30s: %10.3f sec"
        #             % (
        #                 "Total Sensitivity Time - %s" % (f),
        #                 totalSensEndTime[f] - adjointEndTime[f],
        #             )
        #         )
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Complete Sensitivity Time", finalEvalSensTime - startEvalSensTime))
        #     print("+--------------------------------------------------+")

        return costFuncsSens

    def evalFFDSens(self):
        """
        Here we evaluate what has been accumulated in xptsens
        """

        # Get the aero mesh sensitivities
        dIdx_aero = self.DVGeo.totalSensitivity(self.Xb, self.curAP.ptSetName)

        # dIdx_struct = self.DVGeo.totalSensitivity(structXptSens, self.structSolver.curSP.ptSetName)

        # print dIdx_aero
        # print dIdx_struct
        # print "Total chord:", dIdx_struct["chord"][0,0] + dIdx_aero["chord"][0,0]
        # print "Total span:", dIdx_struct["span"][0,0] + dIdx_aero["span"][0,0]

        self.dIdx_geo_total = {}

        for key in dIdx_aero:
            self.dIdx_geo_total[key] = dIdx_aero[key]

    def writeSolution(self, number, baseName):
        """TODO"""

        # self.DCFoil.write
        pass

    def setDesignVars(self, DVs):
        """
        Set the design variables

        Parameters
        ----------
        DVs : dict
            Design variables
        """

        self.DVDict = DVs

    def setDVGeo(self, geo):
        """
        Adds DVGeo object to the solver
        """
        self.DVGeo = geo

    def setAeroProblem(self, aeroProblem):
        """
        Sets the aeroProblem object
        Based on Eirikur's DLM4PY code
        """

        # Add the point-set name of the aero mesh coords embedded in the ffd
        ptSetName = f"dcfoil_{aeroProblem.name}_coords"

        # Now check if we have an DVGeo object to deal with:
        if self.DVGeo is not None:

            # DVGeo appeared and we have not embedded points!
            if not ptSetName in self.DVGeo.points:
                self.DVGeo.addPointSet(self.LEcoords, ptSetName)
                aeroProblem.ptSetName = ptSetName

            # Check if our point-set is up to date:
            if not self.DVGeo.pointSetUpToDate(ptSetName):
                self.LEcoords = self.DVGeo.update(ptSetName, config=aeroProblem.name)

        # update the aero mesh that the transfer object has
        # self.transfer.setAeroSurfaceNodes(np.ravel(self.X))

        # Set the aeroproblem data
        self._setAeroProblemData(aeroProblem)

        # Finally update the aeroproblem
        self.curAP = aeroProblem

    def __call__(self, aeroProblem):

        self.setAeroProblem(aeroProblem)
        self.curAP.callCounter += 1

        self.solve()

        self.writeSolution()
