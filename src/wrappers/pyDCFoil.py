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
    def __init__(self, options=None, debug=False):
        """
        Create DCFoil object
        """
        startInitTime = time.time()

        # --- Load module from Julia ---
        try:
            if debug:
                Pkd.dev(".")
                # Pkg.add("DCFoil")
            else:
                Pkg.add("DCFoil")
            self.dcfoil = Main.using("DCFoil")
        except Exception:
            raise DCFOILWarning("Could not load DCFOIL module from Julia")

        importTime = time.time()

        # --- Set options, DVs info ---
        defaultOptions = self._getDefaultOptions()

        self.possibleDVs, self.dcfoilCostFunctions = self._getObjectivesAndDVs()

        setupTime = time.time()

        # --- Initialization ---
        # TODO: dcfoil should breakup the run_model() routine into initialize and run_model
        self.dcfoil.init_model
        initTime = time.time()

        if self.getOption("printTiming"):
            print("+--------------------------------------------------+")
            print("|")
            print("| Initialization Times:")
            print("|")
            print("| %-30s: %10.3f sec" % ("Import Time", importTime - startInitTime))
            print("| %-30s: %10.3f sec" % ("Setup Time", setupTime - importTime))
            print("| %-30s: %10.3f sec" % ("Initialize Time", initTime - setupTime))
            print("|")
            print("| %-30s: %10.3f sec" % ("Total Init Time", initTime - startInitTime))
            print("+--------------------------------------------------+")

    @staticmethod
    def _getDefaultOptions():
        defaultOptions = {
            # ---------------------------
            #   I/O
            # ---------------------------
            "name": [str, None],
            "debug": [bool, False],
            "outputDir": [str, "./OUTPUT/"],
            "writeTecplotSolution": [bool, True],
            # ---------------------------
            #   General appendage options
            # ---------------------------
            "config": [str, "wing"],
            "nNodes": [int, 10],  # number of nodes on foil half wing
            "nNodeStrut": [int, 10],  # nodes on strut
            "rotation": [float, 0.0],  # Rotation of the wing about the x-axis [deg]
            "gravityVector": [list, [0.0, 0.0, -9.81]],
            "use_tipMass": [bool, False],
            # ---------------------------
            #   Flow
            # ---------------------------
            "Uinf": [float, 5.0],  # free stream velocity [m/s]
            "rhof": [float, 1000.0],  # fluid density [kg/m³]
            "use_cavitation": [bool, False],
            "use_freeSurface": [bool, False],
            "use_ventilation": [bool, False],
            # ---------------------------
            #   Structure
            # ---------------------------
            "material": [str, "cfrp"],  # preselect from material library
            # ---------------------------
            #   Solver modes
            # ---------------------------
            # --- General solver options ---
            "config": [str, "wing"],
            # --- Static solve ---
            "run_static": [bool, False],
            # --- Forced solve ---
            "run_forced": [bool, False],
            "fSweep": [list, [0.0]],
            "tipForceMag": [float, 0.0],
            # --- p-k (Eigen) solve ---
            "run_modal": [bool, False],
            "run_flutter": [bool, False],
            "nModes": [int, 3],  # Number of struct modes to solve for (starting)
            "uRange": [list, [1.0, 5.0]],  # Range of velocities to sweep
            "maxQIter": [int, 200],  # max dyn pressure iters
            "rhoKS": [float, 80.0],
        }
        return defaultOptions

    def __call__(self):
        """
        Solve the problem
        """

        if self.getOption("printTiming"):
            print("+-------------------------------------------------+")
            print("|")
            print("| Solution Timings:")
            print("|")
            print("| %-30s: %10.3f sec" % ("Set AeroProblem Time", t1 - startCallTime))
            print("| %-30s: %10.3f sec" % ("Solution Time", solTime))
            print("| %-30s: %10.3f sec" % ("Write Solution Time", writeSolutionTime - t2))
            print("| %-30s: %10.3f sec" % ("Stability Parameter Time", stabilityParameterTime - writeSolutionTime))
            print("|")
            print("| %-30s: %10.3f sec" % ("Total Call Time", stabilityParameterTime - startCallTime))
            print("+--------------------------------------------------+")

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

        if self.getOption("printTiming"):
            print("+---------------------------------------------------+")
            print("|")
            print("| Function Timings:")
            print("|")
            print("| %-30s: %10.3f sec" % ("Function AeroProblem Time", aeroProblemTime - startEvalTime))
            print("| %-30s: %10.3f sec" % ("Function Evaluation Time", getSolutionTime - aeroProblemTime))
            print("| %-30s: %10.3f sec" % ("User Function Evaluation Time", userFuncTime - getSolutionTime))
            print("|")
            print("| %-30s: %10.3f sec" % ("Total Function Evaluation Time", userFuncTime - startEvalTime))
            print("+--------------------------------------------------+")

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

        if self.getOption("printTiming") and self.comm.rank == 0:
            print("+--------------------------------------------------+")
            print("|")
            print("| Adjoint Times:")
            print("|")
            for f in evalFuncs:
                print(
                    "| %-30s: %10.3f sec" % ("Adjoint Solve Time - %s" % (f), adjointEndTime[f] - adjointStartTime[f])
                )
                print(
                    "| %-30s: %10.3f sec"
                    % ("Total Sensitivity Time - %s" % (f), totalSensEndTime[f] - adjointEndTime[f])
                )
            print("|")
            print("| %-30s: %10.3f sec" % ("Complete Sensitivity Time", finalEvalSensTime - startEvalSensTime))
            print("+--------------------------------------------------+")

    def _getObjectivesAndDVs(self):
        """
        All possible objectives and design variables
        """
        iDV = OrderedDict()
        iDV["theta"] = self.dcfoil
        # TODO: complete this

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
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    DynamicSolver = pyDCFOIL(debug=True)
