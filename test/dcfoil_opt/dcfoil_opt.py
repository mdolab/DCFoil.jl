# --- Python 3.10 ---
"""
@File    :   dcfoil_opt.py
@Time    :   2024/02/20
@Author  :   Galen Ng
@Desc    :   Test script for pyDCFoil optimization
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import date

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
# import niceplots
from pprint import pprint as pp
from pyoptsparse import Optimization, History
from SETUP import setup_dcfoil, setup_dvgeo, setup_opt, setup_utils
from mpi4py import MPI
from baseclasses import AeroProblem
from baseclasses.utils import readJSON, redirectIO, writeJSON
from multipoint import multiPointSparse

from INPUT.flow_constants import TEMP, MU, RHO_F, P_ATM, GRAV, P_VAP
from SPECS.point_specs import boatSpds, clstars, opdepths, alfa0


# ==============================================================================
#                         MODELING FUNCTIONS
# ==============================================================================
def compute_clvent(Fnh: float, Ufs):
    """
    Compute the critical c_l for ventilation

    Parameters
    ----------
    Fnh : float
        Depth-based Froude number

    Returns
    -------
    clvent : float

    TODO: eventually, this should be in the julia layer to get the fnh(XLoc) effect
    """
    PC = P_VAP  # cavity pressure
    sigmav = (P_ATM - PC) / (0.5 * RHO_F * Ufs**2)
    clvent = (1 - np.exp(-sigmav * Fnh)) / np.sqrt(Fnh)

    return clvent


def multipoint_load_balance(nCruise: int, nManeuver: int, task):
    """
    Breaks up the processors into groups for each design point
    """
    MP = multiPointSparse(MPI.COMM_WORLD)  # create multipoint sparse object
    if MPI.COMM_WORLD.size >= (nCruise + nManeuver):  # If there are enough procs, divide them up
        if args.task in ["opt", "run", "make_slice"]:
            nProcPerGroup = MPI.COMM_WORLD.size // (nCruise + nManeuver)  # procs per group (maneuver procs 3:1 ratio)
            nRemainder = MPI.COMM_WORLD.size % (nCruise + nManeuver)  # what is leftover?

            cruiseMemberSizes = []
            for ii in range(nCruise):
                cruiseMemberSizes.append(nProcPerGroup)

            maneuverMemberSizes = nProcPerGroup + nRemainder  # add the remainder to the last group b/c I said so

            # --- Add processor sets ---
            MP.addProcessorSet("cruise", nMembers=(nCruise), memberSizes=cruiseMemberSizes)
            MP.addProcessorSet("maneuver", nMembers=nManeuver, memberSizes=maneuverMemberSizes)

        else:
            # Don't worry about load balance for pure polar
            nRemainder = 0
            cruiseMemberSizes = MPI.COMM_WORLD.size
            MP.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)

        # --- Print out proc division for debugging ---
        if MP.gcomm.rank == 0:
            print(f"Remaining processors: {nRemainder}", flush=True)
            print(
                f"{nCruise} cruise processor sets have this division of procs: {cruiseMemberSizes}",
                flush=True,
            )
            if nManeuver > 0:
                print(
                    f"{nManeuver} maneuver processor sets have this division of procs: {maneuverMemberSizes}",
                    flush=True,
                )

    else:  # If there aren't enough, we should just run everything on one processor TODO: make it work
        warnings.warn("Not enough procs to divide up...")

    # --- Create the communicator ---
    comm, _, _, _, pt_id = MP.createCommunicators()

    return MP, comm, pt_id


def fd_variable(funcsSens, dvDict, dvName, funcs, evalFuncs):
    DH = 1e-5
    funcsFD = {}
    dvDict[dvName] += DH
    DVGeo.setDesignVars(dvDict)
    ap.setDesignVars(dvDict)

    STICKSolver(ap)
    STICKSolver.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)

    dvDict[dvName] -= DH
    DVGeo.setDesignVars(dvDict)
    ap.setDesignVars(dvDict)

    for evalFunc in evalFuncs:
        funcsSens[dvName][evalFunc] = np.divide(funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH)


def setup_fileIO(args):
    date_str = date.today().strftime("%Y-%m-%d")
    caseName = f"{date_str}_{args.task}_{args.foil}"

    if args.restart:
        caseName = f"{date_str}_{args.task}_restart_{args.restart}"

    caseoutputDir = f"{args.output}/{caseName}/"

    return caseName, caseoutputDir


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="INPUT")
    parser.add_argument("--output", type=str, default="OUTPUT")
    parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
    parser.add_argument(
        "--geovar",
        type=str,
        default="w",
        help="Geo vars: twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    parser.add_argument(
        "--optimizer", "-o", help="type of optimizer", choices=["SLSQP", "SNOPT"], type=str, default="SNOPT"
    )
    parser.add_argument("--opt_pts", type=str, default="1")
    parser.add_argument("--restart", type=str, default=None, help="restart name")
    parser.add_argument(
        "--task",
        help="Check end of script for task type",
        type=str,
        default="run",
    )
    parser.add_argument("--is_dynamic", action="store_true", default=False)
    args = parser.parse_args()

    # --- Echo the args ---
    if MPI.COMM_WORLD.rank == 0:
        print(30 * "-")
        print("Arguments are", flush=True)
        for arg in vars(args):
            print(f"{arg:<20}:{getattr(args, arg)}", flush=True)
        print(30 * "-", flush=True)

    # ************************************************
    #     Input Files/ IO
    # ************************************************
    caseName, outputDir = setup_fileIO(args)
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    nCruise = len(args.opt_pts)
    MP, comm, ptID = multipoint_load_balance(nCruise, 0, args.task)

    files = {}
    files["gridFile"] = [
        f"./{args.input}/{args.foil}_foil_stbd_mesh.dcf",
        f"./{args.input}/{args.foil}_foil_port_mesh.dcf",
    ]
    files["FFDFile"] = f"{args.input}/{args.foil}_ffd.xyz"
    # files["FFDFile-port"] = f"{args.input}/{args.foil}_port_ffd.xyz"

    filesSETUP = {
        "adflow": "SETUP/setup_adflow.py",
        "constants": "SETUP/setup_constants.py",
        "dvgeo": "SETUP/setup_dvgeo.py",
    }

    # ************************************************
    #     Geometric design variables
    # ************************************************
    DVGeo = setup_dvgeo.setup(args, comm, files)

    # ==============================================================================
    #                         DCFoil setup
    # ==============================================================================

    evalFuncs = [
        "wtip",
        "psitip",
        "cl",
        "cmy",
        "lift",
        "moment",
        "cd",
        # "ksflutter",
        "kscl",
    ]
    evalFuncsAdj = [
        "wtip",
        # "psitip",
        "cl",
        # "lift",
        "cd",
        # "drag",
        # "ksflutter",
        # "kscl",
    ]

    all_aps = []
    all_clvents = {}
    for prob in args.opt_pts:
        for apname, val in boatSpds.items():
            if apname == "p" + prob:
                print(30 * "=")
                print(f"Setting up AeroProblem for {apname}")

                ap = AeroProblem(
                    name=apname,
                    alpha=alfa0,
                    V=boatSpds[apname],
                    rho=RHO_F,
                    T=TEMP,
                    areaRef=1.0,
                    chordRef=1.0,
                    xRef=0.25,
                    evalFuncs=evalFuncs,
                    muSuthDim=MU,
                    TSuthDim=TEMP,
                )
                Fnh = boatSpds[apname] / np.sqrt(GRAV * opdepths[apname])
                clvent = compute_clvent(Fnh, boatSpds[apname])

                all_aps.append(ap)
                all_clvents[apname] = clvent

                print(f"alfa0\tclvent\tFnh")
                print(f"{ap.alpha:.2f}\t{clvent:.2f}\t{Fnh:.2f}")

    # ************************************************
    #     Create dynamic beam solver
    # ************************************************
    STICKSolver, solverOptions, valDict, lowerDict, upperDict, scaleDict = setup_dcfoil.setup(
        args, comm, files, evalFuncs, outputDir, all_aps[0]
    )
    STICKSolver.setDVGeo(DVGeo)
    STICKSolver.addMesh(solverOptions["gridFile"])

    is_dynamic = solverOptions["run_flutter"] and "ksflutter" in evalFuncsAdj

    # ************************************************
    #     Functions
    # ************************************************
    def cruiseFuncs(x, return_all=False):
        # FLUTTER FUNCS AT SPEED ENVELOPE BOUND ('flutterFuncs(x)' instead?)
        funcs = {}
        tmp = {}

        for ap in all_aps:
            # Set design variables
            ap.setDesignVars(x)
            DVGeo.setDesignVars(x)
            STICKSolver.setDesignVars(x)

            # --- Solve ---
            STICKSolver(ap)

            # --- Grab cost funcs ---
            STICKSolver.evalFunctions(ap, tmp, evalFuncs=evalFuncs)

            for func in evalFuncsAdj:
                funcs[f"{ap.name}_{func}"] = tmp[f"{ap.name}_{func}"]

            out_ext = f"{STICKSolver.callCounter:03d}"
            if MP.gcomm.rank == 0:
                # NOTE: write stuff to the json files because python will know it's a dictionary
                writeJSON(os.path.join(outputDir, f"dvs_{out_ext}.json"), x)
                writeJSON(os.path.join(outputDir, f"funcs_{ap.name}_{out_ext}.json"), tmp)

        if MP.gcomm.rank == 0:
            print("current DVs:")
            pp(x)
            print(f"These are the funcs at DCFoil call {STICKSolver.callCounter:3d}: ")
            pp(tmp)
            print("Funcs to objCon")
            pp(funcs)

        if return_all:
            return tmp
        else:
            return funcs

    def cruiseFuncsSens(x, funcs):
        funcsSens = {}

        for ap in all_aps:
            # --- Solve sensitivity ---
            STICKSolver.evalFunctionsSens(ap, funcsSens, evalFuncs=evalFuncsAdj)

            # The span derivative is the only broken derivative in DCFoil. We FD it here.
            dh = 1e-5
            if "span" in x.keys():
                funcs_i = {}
                funcs_f = {}

                STICKSolver.evalFunctions(ap, funcs_i, evalFuncs=evalFuncsAdj)
                x["span"] += dh
                DVGeo.setDesignVars(x)
                ap.setDesignVars(x)
                STICKSolver(ap)
                STICKSolver.evalFunctions(ap, funcs_f, evalFuncs=evalFuncsAdj)

                x["span"] -= dh

                STICKSolver.callCounter -= 1

                DVGeo.setDesignVars(x)
                ap.setDesignVars(x)
                for key, value in funcs_i.items():
                    funcsSens["span"] = (funcs_f[key] - funcs_i[key]) / dh

        if MP.gcomm.rank == 0:
            print("These are the funcsSens: ")
            pp(funcsSens)

        return funcsSens

    def objCon(funcs, printOK):  # this part is only needed for multipoint
        # Assemble the objective and any additional constraints:

        funcs["obj"] = 0.0

        for ap in all_aps:
            funcs["obj"] += funcs[f"{ap.name}_cd"]

            # ---------------------------
            #   Constraints
            # ---------------------------
            # --- Lift ---
            mycl = clstars[ap.name]
            funcs["cl_con_" + ap.name] = funcs[f"{ap.name}_cl"] - mycl

            # # --- Ventilation ---
            # myventcl = all_clvents[ap.name]
            # funcs[f"ventilation_con_{ap.name}"] = funcs[f"{ap.name}_kscl"] - myventcl

            # --- Tip bending ---
            mywtip = 0.05 * 0.333  # 5% of the initial semispan
            funcs[f"wtip_con_{ap.name}"] = funcs[f"{ap.name}_wtip"] - mywtip

            # --- Dynamics ---
            if is_dynamic:
                # --- Flutter ---
                funcs[f"ksflutter_con_{ap.name}"] = funcs[f"{ap.name}_ksflutter"]

        if printOK:
            print("funcs in obj: ", funcs)

        return funcs

    # ==============================================================================
    #                         Setup optimizer
    # ==============================================================================
    optProb = Optimization(f"{caseName}", MP.obj, comm=MPI.COMM_WORLD)
    optProb.addObj("obj", scale=1e4)

    # ************************************************
    #     Design varaibles
    # ************************************************
    # --- Geometric ---
    DVGeo.addVariablesPyOpt(optProb)
    # --- DCFoil variables ---
    STICKSolver.addVariablesPyOpt(optProb, "alfa0", valDict, lowerDict, upperDict, scaleDict)
    STICKSolver.addVariablesPyOpt(optProb, "theta_f", valDict, lowerDict, upperDict, scaleDict)
    STICKSolver.addVariablesPyOpt(optProb, "toc", valDict, lowerDict, upperDict, scaleDict)

    # ************************************************
    #     Constraints
    # ************************************************
    for ap in all_aps:
        optProb.addCon(f"cl_con_{ap.name}", lower=0.0, upper=0.0, scale=1.0)
        # optProb.addCon(f"ventilation_con_{ap.name}", upper=0.0, scale=1.0)
        optProb.addCon(f"wtip_con_{ap.name}", upper=0.0, scale=1.0)
        if is_dynamic:
            optProb.addCon(f"ksflutter_con_{ap.name}", lower=0.0, upper=0.0, scale=1.0)

    # ************************************************
    #     Finalize optimizer
    # ************************************************
    # The MP object needs the 'obj' and 'sens' function for each proc set,
    # the optimization problem and what the objcon function is:
    MP.setProcSetObjFunc("cruise", cruiseFuncs)
    MP.setProcSetSensFunc("cruise", cruiseFuncsSens)
    MP.setObjCon(objCon)
    MP.setOptProb(optProb)
    optProb.printSparsity()

    opt, optOptions = setup_opt.setup(args, outputDir)

    # ==============================================================================
    #                         RESTART
    # ==============================================================================
    x_init = STICKSolver.getValues()
    if args.restart:
        from restart_dvs import dv_dict

        x_init = dv_dict[args.restart]

    json.dump(x_init, open(f"{outputDir}/init_dvs.json", "w"), cls=setup_utils.NumpyEncoder)
    # ==============================================================================
    #                         TASKS
    # ==============================================================================
    # ---------------------------
    #   OPTIMIZATION
    # ---------------------------
    if args.task == "opt":
        sol = opt(optProb, MP.sens, storeHistory=f"{outputDir}/opt.hst")

        print(sol)
        print(f"final DVs:\n")
        pp(sol.xStar)

        # --- Append DV to restart file ---
        restart_dvs_file = open("./restart_dvs.py", "a")  # append mode
        restart_dvs_file.write(f'\ndv_dict["{caseName}"] = ')
        json.dump(sol.xStar, restart_dvs_file, cls=setup_utils.NumpyEncoder)
        restart_dvs_file.write("\n")
        restart_dvs_file.close()

    # ---------------------------
    #   ANALYSIS
    # ---------------------------
    if args.task in ["run", "opt"]:
        if args.task == "run":
            x_init = DVGeo.getValues()
            run_dvs = x_init
        if args.task == "opt":
            run_dvs = sol.xStar

        funcs = cruiseFuncs(run_dvs, return_all=True)

        # --- Pretty print output ---
        CLtxt = funcs[f"{ap.name}_cl"]
        CDtxt = funcs[f"{ap.name}_cd"]
        if solverOptions["run_flutter"]:
            KSFluttertxt = funcs[f"{ap.name}_ksflutter"]
        else:
            KSFluttertxt = "N/A"
        table = [["CL", CLtxt], ["CD", CDtxt], ["KSFlutter", KSFluttertxt]]
        try:
            from tabulate import tabulate

            print(tabulate(table))
        except Exception:
            import warnings

            warnings.warn("tabulate not installed")
            print(table)
