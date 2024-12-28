from pyoptsparse import OPT
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
            "Function precision": 1e-8,
            "Print file": os.path.join(outputDir, "SNOPT_print.out"),
            "Summary file": os.path.join(outputDir, "SNOPT_summary.out"),
            "Verify level": -1,  # NOTE: verify level 0 is pretty useless; just use level 1--3 when testing a new feature
            # "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
            # "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (nonlinear problem)
            # "Major Step Limit": 5e-3,
            # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
        }

    opt = OPT(args.optimizer, options=optOptions)

    return opt, optOptions
