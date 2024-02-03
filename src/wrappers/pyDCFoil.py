# --- Python 3.10 ---
"""
@File    :   mach2dcfoil.py
@Time    :   2024/01/22
@Author  :   Galen Ng
@Desc    :   Python interface to DCFoil
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


# ==============================================================================
#                         Wrapper class
# ==============================================================================
class pyDCFOIL:
    def __init__(self, DVDict: dict, evalFuncs, options=None, debug=False):
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
        try:
            if debug:
                # Pull from local directory
                Pkg.activate("../../")
                Main.include("../DCFoil.jl")
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

        # ************************************************
        #     Set options, DVs info
        # ************************************************
        defaultOptions = self._getDefaultOptions()

        # self.possibleDVs, self.dcfoilCostFunctions = self._getObjectivesAndDVs()

        setupTime = time.time()

        self.DVDict = DVDict
        self.evalFuncs = evalFuncs

        self.solverOptions = {}
        # --- Set all solver options ---
        for key, val in defaultOptions.items():
            if key not in options:  # Use default
                self.solverOptions[key] = val
            else:
                self.solverOptions[key] = options[key]
        self.solverOptions = options
        
        # --- Make output directory ---
        Path(self.solverOptions["outputDir"]).mkdir(parents=True, exist_ok=True)

        initTime = time.time()

        # if self.getOption("printTiming"):
        #     print("+--------------------------------------------------+")
        #     print("|")
        #     print("| Initialization Times:")
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Import Time", importTime - startInitTime))
        #     print("| %-30s: %10.3f sec" % ("Setup Time", setupTime - importTime))
        #     print("| %-30s: %10.3f sec" % ("Initialize Time", initTime - setupTime))
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Total Init Time", initTime - startInitTime))
        #     print("+--------------------------------------------------+")

    @staticmethod
    def _getDefaultOptions():
        defaultOptions = {
            # ---------------------------
            #   I/O
            # ---------------------------
            "name": "default",
            "debug": False,
            "outputDir": "./OUTPUT/",
            "writeTecplotSolution": True,
            # ---------------------------
            #   General appendage options
            # ---------------------------
            "config": "wing",
            "nNodes": 10,  # number of nodes on foil half wing
            "nNodeStrut": 10,  # nodes on strut
            "rotation": 0.0,  # Rotation of the wing about the x-axis [deg]
            "gravityVector": [0.0, 0.0, -9.81],
            "use_tipMass": False,
            # ---------------------------
            #   Flow
            # ---------------------------
            "U∞": 5.0,  # free stream velocity [m/s]
            "ρ_f": 1000.0,  # fluid density [kg/m³]
            "use_cavitation": False,
            "use_freeSurface": False,
            "use_ventilation": False,
            # ---------------------------
            #   Structure
            # ---------------------------
            "material": "cfrp",  # preselect from material library
            "strut_material": "cfrp",
            # ---------------------------
            #   Solver modes
            # ---------------------------
            # --- Static solve ---
            "run_static": False,
            # --- Forced solve ---
            "run_forced": False,
            "fSweep": np.linspace(0.0, 1.0, 10),
            "tipForceMag": 0.0,
            # --- p-k (Eigen) solve ---
            "run_modal": False,
            "run_flutter": False,
            "nModes": 3,  # Number of struct modes to solve for (starting)
            "uRange": [1.0, 5.0],  # Range of velocities to sweep
            "maxQIter": 100,  # max dyn pressure iters
            "rhoKS":  80.0,
        }
        return defaultOptions

    def solve(self):
        """
        Solve foil problem
        """

        DVDict = self.DVDict
        evalFuncs = self.evalFuncs

        solverOptions = self.solverOptions

        self.DCFoil.init_model(DVDict, evalFuncs, solverOptions)
        FLUTTERSOL = self.DCFoil.run_model(
            DVDict, evalFuncs, solverOptions=solverOptions
        )

        self.FLUTTERSOL = FLUTTERSOL

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

    def evalFunctions(self, funcs, evalFuncs=None):
        """
        Evaluate desired functions in 'evalFuncs' and add
        them to the dictionary 'funcs'.

        Parameters
        ----------
        funcs : dict
            Dictionary into which functions are saved
        evalFuncs : iterable object containing strings, optional
            If not None, use these functions to evaluate.
        """

        if evalFuncs is None:
            evalFuncs = sorted(self.evalFuncs)

        solverOptions = self.solverOptions

        FLUTTERSOL = self.FLUTTERSOL

        costFuncs = self.DCFoil.evalFuncs(FLUTTERSOL, evalFuncs, solverOptions)
        # funcs = DCFoil.evalFuncs(FLUTTERSOL, evalFuncs, solverOptions)
        # Convert costFuncs to a dictionary to fill 'funcs'

        for key, val in costFuncs.items():
            funcs[key] = val
        self.costFuncs = costFuncs

        # if self.getOption("printTiming"):
        #     print("+---------------------------------------------------+")
        #     print("|")
        #     print("| Function Timings:")
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Function AeroProblem Time", aeroProblemTime - startEvalTime))
        #     print("| %-30s: %10.3f sec" % ("Function Evaluation Time", getSolutionTime - aeroProblemTime))
        #     print("| %-30s: %10.3f sec" % ("User Function Evaluation Time", userFuncTime - getSolutionTime))
        #     print("|")
        #     print("| %-30s: %10.3f sec" % ("Total Function Evaluation Time", userFuncTime - startEvalTime))
        #     print("+--------------------------------------------------+")

        return costFuncs

    def evalFunctionsSens(self, funcsSens, evalFuncs=None):
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

        if evalFuncs is None:
            evalFuncs = sorted(self.evalFuncs)

        solverOptions = self.solverOptions

        costFuncsSens = self.DCFoil.evalFuncsSens(
            DVDict, evalFuncs, solverOptions, mode="RAD"
        )

        self.costFuncsSens = costFuncsSens

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

    def writeSolution():
        """TODO"""
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


# ==============================================================================
#                         MAIN DRIVER (test optimization problem here)
# ==============================================================================
if __name__ == "__main__":

    # ==============================================================================
    #                         Extension modules
    # ==============================================================================
    from pprint import pprint as pp
    from pyoptsparse import Optimization, OPT, History
    import pyDCFoil

    # ************************************************
    #     Create dynamic beam solver
    # ************************************************
    nNodes = 3
    nNodesStrut = 3
    solverOptions = {
        # ---------------------------
        #   I/O
        # ---------------------------
        "name": "test",
        "debug": True,
        "writeTecplotSolution": True,
        "outputDir": "./OUTPUT/",
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "config": "wing",
        # "config": "t-foil",
        "nNodes": nNodes,  # number of nodes on foil half wing
        "nNodeStrut": nNodesStrut,  # nodes on strut
        "rotation": 0.0,  # deg
        "gravityVector": [0.0, 0.0, -9.81],
        "use_tipMass": False,
        # ---------------------------
        #   Flow
        # ---------------------------
        "U∞": 5.0,  # free stream velocity [m/s]
        "ρ_f": 1000.0,  # fluid density [kg/m³]
        "use_freeSurface": False,
        "use_cavitation": False,
        "use_ventilation": False,
        # ---------------------------
        #   Structure
        # ---------------------------
        "material": "cfrp",  # preselect from material library
        "strut_material": "cfrp",
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static": True,
        # --- Forced solve ---
        "run_forced": True,
        "fSweep": np.linspace(0.1, 1000.0, 1000),
        "tipForceMag": 1.0,
        # --- p-k (Eigen) solve ---
        "run_modal": True,
        "run_flutter": True,
        "nModes": 4,
        "uRange": [170.0, 190.0],
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 80.0,
    }
    DVDict = {
        "α₀": 6.0,  # initial angle of attack [deg]
        "Λ": np.deg2rad(-15.0),  # sweep angle [rad]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        "c": 0.1 * np.ones(nNodes),  # chord length [m]
        "s": 0.3,  # semispan [m]
        "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
        "toc": 0.12 * np.ones(nNodes),  # thickness-to-chord ratio
        "x_αb": 0 * np.ones(nNodes),  # static imbalance [m]
        "θ": np.deg2rad(15),  # fiber angle global [rad]
        # --- Strut vars ---
        "beta": 0.0,  # yaw angle wrt flow [deg]
        "s_strut": 0.4,  # from Yingqian
        "c_strut": 0.1 * np.ones(nNodesStrut),  # chord length [m]
        "toc_strut": 0.12 * np.ones(nNodesStrut),  # thickness-to-chord ratio
        "ab_strut": 0 * np.ones(nNodesStrut),  # dist from midchord to EA [m]
        "x_αb_strut": 0 * np.ones(nNodesStrut),  # static imbalance [m]
        "θ_strut": np.deg2rad(15),  # fiber angle global [rad]
    }
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

    DynamicSolver = pyDCFOIL(DVDict, evalFuncs, debug=True, options=solverOptions)

    # ************************************************
    #     Functions
    # ************************************************
    def cruiseFuncs(x):
        funcs = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["θ"] = x["fiberangle"].item()
        DynamicSolver.setDesignVars(DVDict)

        # --- Solve ---
        DynamicSolver.solve()

        # --- Grab cost funcs ---
        funcs = DynamicSolver.evalFunctions(funcs)

        # --- Set objective ---
        funcs["obj"] = funcs["ksflutter"]

        pp(funcs)

        return funcs

    def cruiseFuncsSens(x, funcs):
        funcsSens = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["θ"] = x["fiberangle"].item()
        DynamicSolver.setDesignVars(DVDict)

        # --- Solve sensitivity ---
        funcsSens = DynamicSolver.evalFunctionsSens(funcsSens)

        # --- Set objective ---
        funcsSens["obj"] = funcsSens["ksflutter"]

        pp(funcsSens)

        return funcsSens

    # def objCon(funcs): # this part is only for multipoint

    #     funcs["obj"] = funcs["ksflutter"]

    #     if printOK:
    #         print("funcs in obj: ", funcs)

    #     return funcs

    # ************************************************
    #     Setup optimizer
    # ************************************************
    optProb = Optimization("opt", cruiseFuncs)

    optProb.addObj("obj")

    # ---------------------------
    #   DVs
    # ---------------------------
    optProb.addVarGroup(
        name="chord",
        nVars=nNodes,
        varType="c",
        value=DVDict["c"],
        lower=0.01,
        upper=0.2,
    )
    # optProb.addVarGroup(name="ab", nVars=nNodes,varType="c", value=DVDict["ab"], lower=-0.1, upper=0.1)
    optProb.addVarGroup(
        name="fiberangle",
        nVars=1,
        varType="c",
        value=DVDict["θ"],
        lower=-np.deg2rad(-45),
        upper=np.deg2rad(45),
        scale=1.0,
    )

    optProb.printSparsity()

    optOptions = {
        "IFILE": "SLSQP.out",
    }

    opt = OPT("SLSQP", options=optOptions)

    sol = opt(optProb, sens=cruiseFuncsSens, storeHistory="opt.hst")

    # print(sol)
