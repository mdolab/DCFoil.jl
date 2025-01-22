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
    def __init__(self, appendageParamsList: list, evalFuncs, options=None, debug=False):
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
        self.nodeConn = None

        self.debugDVGeo = False

        self.callCounter = 0

        self.dtype = float
        # ************************************************
        #     Set options, DVs info
        # ************************************************
        defaultOptions = self._getDefaultOptions()

        # self.possibleDVs, self.dcfoilCostFunctions = self._getObjectivesAndDVs()

        setupTime = time.time()

        # if type(DVDictList) == list:
        self.appendageParamsList = appendageParamsList
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
        print(f"Solver options:")
        for key, val in self.solverOptions.items():
            print(f"{key}: {val}")

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
        """
        All keys need to be here
        """
        defaultOptions = {
            # ---------------------------
            #   I/O
            # ---------------------------
            "name": "default",
            "debug": False,
            "printTiming": True,
            "gridFile": None,
            "outputDir": "./OUTPUT/",
            "writeTecplotSolution": True,
            # ---------------------------
            #   General appendage options
            # ---------------------------
            "gravityVector": [0.0, 0.0, -9.81],
            "appendageList": [],
            # ---------------------------
            #   Flow
            # ---------------------------
            "Uinf": 5.0,  # free stream velocity [m/s]
            "rhof": 1000.0,  # fluid density [kg/m³]
            "nu": 1.1892e-06,  # fluid kinematic viscosity [m²/s]
            "use_nlll": True,
            "use_cavitation": False,
            "use_freeSurface": False,
            "use_ventilation": False,
            "use_dwCorrection": True,
            # ---------------------------
            #   Solver modes
            # ---------------------------
            # --- Static solve ---
            "run_static": False,
            "res_jacobian": "analytic",
            # --- Forced solve ---
            "run_forced": False,
            "fRange": [0.1, 1000.0],
            "df": 0.1,
            # --- p-k (Eigen) solve ---
            "run_modal": False,
            "run_flutter": False,
            "nModes": 3,  # Number of struct modes to solve for (starting)
            "uRange": [1.0, 5.0],  # Range of velocities to sweep
            "maxQIter": 100,  # max dyn pressure iters
            "rhoKS": 500.0,
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
    def addMesh(self, gridFiles):
        """
        Add component to mesh

        shape: (n_nodes, 3)
        """

        # # ************************************************
        # #     Add meshes one by one
        # # ************************************************
        # Grid = self.DCFoil.MeshIO.add_mesh(gridFile)

        # ************************************************
        #     Or do all at once
        # ************************************************
        meshOptions = {"junction-first": True}
        if len(gridFiles) > 1:
            Grid = self.DCFoil.MeshIO.add_meshfiles(gridFiles, meshOptions)

        LE_X, node_conn, TE_X = Grid.LEMesh.T, Grid.nodeConn.T, Grid.TEMesh.T
        # Check shape
        assert LE_X.shape[1] == 3

        if self.LEcoords is None:
            # Set mesh and connectivity
            self.LEcoords = LE_X
            self.nodeConn = node_conn
            self.TEcoords = TE_X
        else:
            # Append to existing mesh
            self.LEcoords = np.vstack((self.LEcoords, LE_X))
            self.conn = np.vstack((self.nodeConn, node_conn + self.nnodes))
            self.TEcoords = np.vstack((self.TEcoords, TE_X))

        # set new number of collocation nodes
        self.nnodes = self.LEcoords.shape[0]

        self.X = np.vstack((self.LEcoords, self.TEcoords))
        # pyGeo should preserve this order

        if self.debugDVGeo:
            print(20 * "-")
            print(f"Read mesh from {gridFiles}")
            print(20 * "-")
            print(f"LE coords:\n{self.LEcoords}")
            print(f"TE coords:\n{self.TEcoords}")
            print(f"Node conn:\n{self.nodeConn}")
            print(f"all coords:\n{self.X}")

    def set_structDamping(self):
        """
        When running optimization, the alpha and beta parameters in the proportional damping model have to be held constant for a fair optimization
        """

        alphaConst, betaConst = self.DCFoil.FEMMethods.compute_proportional_damping(
            Ks, Ms, appendageParams["zeta"], solverOptions["nModes"]
        )

    def solve(self):
        """
        Solve foil problem
        """

        appendageParamsList = self.appendageParamsList
        evalFuncs = self.evalFuncs

        # --- Julia is transposed! ---
        LECoords = self.LEcoords.T
        nodeConn = self.nodeConn.T
        TECoords = self.TEcoords.T

        solverOptions = self.solverOptions

        self.DCFoil.init_model(
            LECoords, nodeConn, TECoords, solverOptions=solverOptions, appendageParamsList=appendageParamsList
        )
        SOLDICT = self.DCFoil.run_model(
            LECoords,
            nodeConn,
            TECoords,
            evalFuncs,
            solverOptions=solverOptions,
            appendageParamsList=appendageParamsList,
        )

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

        # Make it a list if it's not
        if type(evalFuncs) is not list:
            evalFuncs = [evalFuncs]

        costFuncs = self.DCFoil.evalFuncs(
            self.SOLDICT,
            self.LEcoords.T,
            self.nodeConn.T,
            self.TEcoords.T,
            self.appendageParamsList,
            evalFuncs,
            self.solverOptions,
        )
        # Convert costFuncs to a dictionary to fill 'funcs'

        for key, val in costFuncs.items():
            # if key in evalFuncs:  #
            funcKey = key.split("-")[0]
            if funcKey in evalFuncs:
                funcs[f"{aeroProblem.name}_{funcKey}"] = val
                if "cd" in funcKey:
                    print("Hard coded to take first value of CD")
                    funcs[f"{aeroProblem.name}_{funcKey}"] = val[0]

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
        # nGeoDV, nStructDV = self._getDVSizes(DVGeo)

        # ************************************************
        #     DCFoil sensitivity
        # ************************************************
        costFuncsSensDict = self.DCFoil.evalFuncsSens(
            self.SOLDICT,
            self.appendageParamsList,
            self.LEcoords.T,
            self.nodeConn.T,
            self.TEcoords.T,
            evalFuncs,
            self.solverOptions,
            mode="ADJOINT",
        )

        for obj in evalFuncs:

            # Get the sensitivity of the cost function wrt all coordinates
            # this is 'dIdpt' of size(Npt, 3)
            try:
                self.Xb = costFuncsSensDict[f"{obj}"]["mesh"].T
                self.paramSens = costFuncsSensDict[f"{obj}"]["params"]
            except KeyError:
                print(f"Could not find {obj} in costFuncsSensDict")
                continue

            # --- check shape ---
            assert self.Xb.shape == (self.nnodes * 2, 3)

            # FFD sensitivities
            self.evalFFDSens()

            # Now add derivatives to funcsSens
            key = f"{self.curAP.name}_{obj}"
            funcsSens[key] = {}

            # ============================
            #   Geometric Variables
            # ============================
            funcsSens[key].update(self.dIdx_geo_total)

            # ============================
            #   Structural Variables
            # ============================
            for paramKey, value in self.paramSens.items():
                funcsSens[key][paramKey] = value

            finalEvalSensTime = time.time()

        # if self.getOption("printTiming") and self.comm.rank == 0:
        #     print("+--------------------------------------------------+")
        #     # print("|")
        #     # print("| Adjoint Times:")
        #     # print("|")
        #     # for f in evalFuncs:
        #     #     print(
        #     #         "| %-30s: %10.3f sec"
        #     #         % (
        #     #             "Adjoint Solve Time - %s" % (f),
        #     #             adjointEndTime[f] - adjointStartTime[f],
        #     #         )
        #     #     )
        #     #     print(
        #     #         "| %-30s: %10.3f sec"
        #     #         % (
        #     #             "Total Sensitivity Time - %s" % (f),
        #     #             totalSensEndTime[f] - adjointEndTime[f],
        #     #         )
        #     #     )
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Complete Sensitivity Time", finalEvalSensTime - startEvalSensTime))
        #     print("+--------------------------------------------------+")

        return funcsSens

    def evalFFDSens(self):
        """
        Here we evaluate what has been accumulated in xptsens
        """

        # Get the aero mesh sensitivities
        dIdx = self.DVGeo.totalSensitivity(self.Xb, self.curAP.ptSetName)

        self.dIdx_geo_total = {}

        for key in dIdx:
            self.dIdx_geo_total[key] = dIdx[key]

    def writeSolution(self, number=None, baseName=None):
        """
        Very specific
        """

        # self.DCFoil.write

        if number is None:
            number = self.callCounter
        if baseName is None:
            baseName = self.curAP.name

        # if self.debugDVGeo:
        print("Writing curves to tecplot...")
        from pyspline.utils import writeTecplot1D, openTecplot, closeTecplot

        outDir = self.solverOptions["outputDir"]
        f = openTecplot(f"{outDir}/{baseName}_mesh_{number:03d}.dat", 3)
        writeTecplot1D(f, "LE", self.LEcoords)
        writeTecplot1D(f, "TE", self.TEcoords)
        closeTecplot(f)

    def setDesignVars(self, DVs):
        """
        Set the design variables (appendageParams dict in dcfoil)
        Call this before solving

        Parameters
        ----------
        DVs : dict
            Design variables
        """

        if self.appendageParamsList is not None:
            for dvKey, value in DVs.items():
                if dvKey in self.appendageParamsList[0].keys():
                    print(f"Setting {dvKey} to {value}")
                    if len(value) == 1:  # scalar case
                        self.appendageParamsList[0][dvKey] = value[0]
                    else:  # vector case
                        self.appendageParamsList[0][dvKey] = value

    def addVariablesPyOpt(self, optProb, dvName, valDict, lowerDict, upperDict, scaleDict):
        """
        Add current set of variables to the optProb object.
        This is based on pytacs example.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition
        """

        ndv = self.getNumDesignVars(dvName)
        print(f"Adding {ndv} {dvName} design variables to the optimization problem")
        optProb.addVarGroup(
            dvName,
            ndv,
            "c",
            value=valDict[dvName],
            lower=lowerDict[dvName],
            upper=upperDict[dvName],
            scale=scaleDict[dvName],
        )

    def getValues(self):

        x = {}

        for key, value in self.appendageParamsList[0].items():
            if key in self.DCFoil.DesignVariables.allDesignVariables:
                x[key] = value

        x.update(self.DVGeo.getValues())

        return x

    def getNumDesignVars(self, dvName):
        """
        Get number of design variables
        """
        try:
            return len(self.appendageParamsList[0][dvName])
        except TypeError:
            return 1

    def setDVGeo(self, geo, debug=False):
        """
        Adds DVGeo object to the solver
        """
        self.DVGeo = geo
        self.debugDVGeo = debug

    def setAeroProblem(self, aeroProblem):
        """
        Sets the aeroProblem object
        """

        # Add the point-set name of the mesh coords embedded in the ffd
        ptSetName = f"dcfoil_{aeroProblem.name}_coords"

        # Now check if we have an DVGeo object to deal with:
        if self.DVGeo is not None:

            # DVGeo appeared and we have not embedded points!
            if not ptSetName in self.DVGeo.points:
                self.DVGeo.addPointSet(self.X, ptSetName)
                aeroProblem.ptSetName = ptSetName

            # Check if our point-set is up to date:
            if not self.DVGeo.pointSetUpToDate(ptSetName):
                self.X = self.DVGeo.update(ptSetName, config=aeroProblem.name)

                # Update the coordinates in the LEcoords
                self.LEcoords = self.X[: self.nnodes]
                self.TEcoords = self.X[self.nnodes :]

                if self.debugDVGeo:
                    print(f"Updated point set {ptSetName}")
                    print(f"LE coords:\n{self.LEcoords}")

            # print("Setting aero problem data")
            # breakpoint()
        # Set the aeroproblem data
        self._setAeroProblemData(aeroProblem)

        # Finally update the aeroproblem
        self.curAP = aeroProblem

    def __call__(self, aeroProblem):

        self.setAeroProblem(aeroProblem)
        self.callCounter += 1

        self.solve()

        self.writeSolution()
