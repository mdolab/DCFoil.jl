# --- Python 3.10 ---
"""
@File          :   setup_dvgeo.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   setup geometric design variables for dcfoil
"""

from pygeo import DVGeometry
import numpy as np


def setup(args, comm, files: dict):
    # ======================================================================
    #         File setup
    # ======================================================================
    DVGeo = DVGeometry(files["FFDFile"], kmax=4)

    nRefAxPts = DVGeo.addRefAxis("global", xFraction=0.5, alignIndex="j")

    # ==============================================================================
    #                         DVS
    # ==============================================================================
    # ---------------------------
    #   TWIST
    # ---------------------------
    if "t" in args.geovar:
        nTwist = nRefAxPts // 2
        print(f"{nTwist} foil twist vars", flush=True)

        twistAxis = "global"

        def twist_roty_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            """
            nSkip = 0
            for ii in range(nTwist):
                # geo.rot_y[twistAxis].coef[ii] = val[ii]

                geo.rot_y[twistAxis].coef[nSkip - ii + nTwist - 1] = val[ii]
                geo.rot_y[twistAxis].coef[nTwist + ii + nSkip + 1] = val[ii]

        DVGeo.addGlobalDV(
            "twist",
            value=np.zeros(nTwist),
            func=twist_roty_func,
            # func=twist_rottheta_func,
            lower=-15.0,
            upper=15.0,
            scale=1.0,
        )

    # ---------------------------
    #   SWEEP
    # ---------------------------
    if "w" in args.geovar:
        # Determine the number of sections that have sweep control
        nSweep = nRefAxPts // 2

        print(f"{nSweep} foil sweep vars", flush=True)
        sweepAxis = "global"

        def sweep_rot_func(inval, geo):
            # REVERSE OF RH RULE FOR DCFOIL
            val = -inval

            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(sweepAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            # sweep_ref_pt = C_orig[0, :]
            sweep_ref_pt = C_orig[nSweep, :]

            theta = -val[0] * np.pi / 180
            cc = np.cos(theta)
            ss = np.sin(theta)
            rot_mtx = np.array(  # rotation matrix about the z-axis
                [
                    [cc, -ss, 0.0],
                    [ss, cc, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            thetanew = val[0] * np.pi / 180
            ccnew = np.cos(thetanew)
            ssnew = np.sin(thetanew)
            rot_mtxnew = np.array(  # rotation matrix about the z-axis
                [
                    [ccnew, -ssnew, 0.0],
                    [ssnew, ccnew, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nSweep):
                # # get the vector from each ref axis point to the wing root
                # vec = C[ii + nSweep + 1, :] - sweep_ref_pt
                # # need to now rotate this by the sweep angle and add back the wing root loc
                # C[ii + nSweep + 1, :] = sweep_ref_pt + rot_mtx @ vec

                vec = C[-ii + nSweep - 1, :] - sweep_ref_pt
                C[-ii + nSweep - 1, :] = sweep_ref_pt + rot_mtx @ vec

                vec = C[nSweep + ii + 1, :] - sweep_ref_pt
                C[nSweep + ii + 1, :] = sweep_ref_pt + rot_mtxnew @ vec

            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, sweepAxis)

        DVGeo.addGlobalDV(
            dvName="sweep",
            value=0.0,
            func=sweep_rot_func,
            lower=0.0,
            upper=30.0,
            scale=1,
        )

    # ---------------------------
    #   DIHEDRAL
    # ---------------------------
    if "d" in args.geovar:
        nDihedral = nRefAxPts
        dihedralAxis = "global"

        def dihedral_rot_func(val, geo):
            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(dihedralAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            sweep_ref_pt = C_orig[0, :]

            theta = -val[0] * np.pi / 180
            cc = np.cos(theta)
            ss = np.sin(theta)
            rot_mtx = np.array(  # rotation matrix about the z-axis
                [
                    [1, 0.0, 0.0],
                    [0.0, cc, -ss],
                    [0.0, ss, cc],
                ]
            )

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nDihedral):
                # get the vector from each ref axis point to the wing root
                vec = C[ii, :] - sweep_ref_pt
                # need to now rotate this by the sweep angle and add back the wing root loc
                C[ii, :] = sweep_ref_pt + rot_mtx @ vec
            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, dihedralAxis)

            # if args.config in ["full", "fs"]:
            #     C2 = geo.extractCoef("c4_v1")
            #     C_orig = C2.copy()
            #     # we will sweep the wing about the first point in the ref axis
            #     sweep_ref_pt = C_orig[0, :]

            #     theta = val[0] * np.pi / 180
            #     cc = np.cos(theta)
            #     ss = np.sin(theta)
            #     rot_mtx = np.array(  # rotation matrix about the z-axis
            #         [
            #             [1, 0.0, 0.0],
            #             [0.0, cc, -ss],
            #             [0.0, ss, cc],
            #         ]
            #     )

            #     # modify the control points of the ref axis
            #     # by applying a rotation about the first point in the x-z plane
            #     for ii in range(nRefAxPts2):
            #         # get the vector from each ref axis point to the wing root
            #         vec = C2[ii, :] - sweep_ref_pt
            #         # need to now rotate this by the sweep angle and add back the wing root loc
            #         C2[ii, :] = sweep_ref_pt + rot_mtx @ vec
            #     # use the restoreCoef method to put the control points back in the right place
            #     geo.restoreCoef(C2, "c4_v1")

        DVGeo.addGlobalDV(
            dvName="dihedral",
            value=0.0,
            func=dihedral_rot_func,
            lower=-5.0,
            upper=5.0,
            scale=1,
        )

    # ---------------------------
    #   CHORD
    # ---------------------------
    if "r" in args.geovar:

        nSkip = 4

        nTaper = nRefAxPts // 2 + 1
        # def chords(val, geo):
        #     # Set all the global chord values
        #     for ii in range(nSkip, nRefAxPts):
        #         geo.scale["global"].coef[ii] = val[ii - nSkip]

        # DVGeo.addGlobalDV(
        #     "chord",
        #     value=np.ones(nRefAxPts - nSkip) * 0.1,
        #     func=chords,
        #     lower=0.01,
        #     upper=0.2,
        #     scale=1.0,
        # )

        def taper(val, geo):
            s = geo.extractS("global")
            # slope = (val[1] - val[0]) / (s[-1] - s[0])
            slope = (val[1] - val[0]) / (s[-1] - s[nTaper - 1])
            for ii in range(nTaper):
                # geo.scale_x["global"].coef[ii] = slope * (s[ii] - s[0]) + val[0]

                geo.scale_x["global"].coef[ii + nTaper - 1] = slope * (s[ii + nTaper - 1] - s[nTaper - 1]) + val[0]
                geo.scale_x["global"].coef[nTaper - ii - 1] = -slope * (s[nTaper - ii - 1] - s[nTaper - 1]) + val[0]

        DVGeo.addGlobalDV(
            "taper",
            value=np.ones(2) * 1.0,
            func=taper,
            lower=[-0.1, -0.1],
            upper=[0.1, 0.1],
            scale=1.0,
        )

    # ---------------------------
    #   Span
    # ---------------------------
    if "p" in args.geovar:
        # nSpan = nRefAxPts
        nSpan = nRefAxPts // 2 + 1

        def span(val, geo):
            C = geo.extractCoef("global")
            s = geo.extractS("global")
            for ii in range(1, nSpan):
                # C[ii, 1] += val * s[ii]

                C[ii + nSpan - 1, 1] += val.item() * s[ii + nSpan - 1]
                C[nSpan - ii - 1, 1] += -val.item() * s[ii + nSpan - 1]
            geo.restoreCoef(C, "global")

        DVGeo.addGlobalDV(dvName="span", value=0.0, func=span, lower=-0.1, upper=0.1, scale=1 / 0.2)

    # def airfoilThickness(val, geo):
    #     # Set airfoil thickness values
    #     for i in range(nSpanwise):
    #         geo.scale_z["wing"].coef[i] = val[0]

    return DVGeo
