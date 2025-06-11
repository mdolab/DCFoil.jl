# from pyoptsparse import OPT
import os


def setup(args, outputDir: str):

    if args.optimizer == "SLSQP":
        optOptions = {
            "IFILE": "SLSQP.out",
        }
    elif args.optimizer == "SNOPT":
        optOptions = {
            "Major feasibility tolerance": 1e-4,
            "Major optimality tolerance": 1e-4,
            "Difference interval": 1e-4,
            "Hessian full memory": None,
            "Hessian frequency": 50, # reset approximate Hessian more often (10-30) for nonlinear problems like hydrostructural allegedly
            "Function precision": 1e-8,
            # "Verify level": 3,  # NOTE: verify level 0 is pretty useless; just use level 1--3 when testing a new feature
            "Verify level": -1,  # NOTE: verify level 0 is pretty useless; just use level 1--3 when testing a new feature
            "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
            "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (if derivs are expensive, use this)
            "Major Step Limit": 1e-2, # good for trim problems
            "Major iterations limit": 100,
            # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
            "Print file": os.path.join(outputDir, "SNOPT_print.out"),
            "Summary file": os.path.join(outputDir, "SNOPT_summary.out"),
        }

    if args.task == "opt":
        optOptions["Major Step Limit"] = 1e-2
        # optOptions["Major Step Limit"] = 5e-3 # maybe this was too small
        # optOptions["Backoff factor"]
        optOptions["Penalty parameter"] = 1e-1 # initial penalty parameter for SNOPT; higher means it favors going feasible first

    # opt = OPT(args.optimizer, options=optOptions)

    # return opt, optOptions
    return optOptions
