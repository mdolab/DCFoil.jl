# --- Python 3.10 ---
"""
@File          :   setup_dvgeo.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   setup geometric design variables for dcfoil
"""

import numpy as np


def setup(args, model, comm, files: dict):
    # ======================================================================
    #         OM setup
    # ======================================================================
    DVGeoInfo = {
        "defaultDVGeo": {
            "file": files["FFDFile"],
            "type": "info",
            "options": None,
        }
    }

    nRefAxPts = model.geometry.nom_addRefAxis("global", xFraction=0.5, alignIndex="j")

    # ==============================================================================
    #                         DVS
    # ==============================================================================
    # ---------------------------
    #   TWIST
    # ---------------------------
    if "t" in args.geovar:
        # nSkip = 2
        # nTwist = nRefAxPts // 2 - nSkip
        nSkip = 1
        # nSkip = 0
        nTwist = nRefAxPts // 2 - nSkip
        print(f"{nTwist} foil twist vars", flush=True)

        twistAxis = "global"

        def twist_roty_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            """
            nSkip = 2
            nSkip = 1
            # nSkip = 0
            for ii in range(nTwist):
                # geo.rot_y[twistAxis].coef[ii] = val[ii]
                geo.rot_y[twistAxis].coef[-ii + nTwist - 1] = val[ii]
                geo.rot_y[twistAxis].coef[nRefAxPts // 2 + nSkip + ii + 1] = val[ii]

        model.geometry.nom_addGlobalDV(
            "twist",
            value=np.zeros(nTwist),
            func=twist_roty_func,
        )

    # ---------------------------
    #   SWEEP
    # ---------------------------
    if "w" in args.geovar:
        # Determine the number of sections that have sweep control
        nSkip = 2
        nSweep = nRefAxPts // 2 - nSkip

        print(f"{nSweep} foil sweep vars", flush=True)
        sweepAxis = "global"

        def sweep_rot_func(inval, geo):
            nSkip = 2
            # REVERSE OF RH RULE FOR DCFOIL
            val = -inval

            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(sweepAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            # sweep_ref_pt = C_orig[0, :]
            sweep_ref_pt = C_orig[nSweep + nSkip, :]

            sweep_ref_pt_port = C_orig[nSweep + nSkip - nSkip, :]
            sweep_ref_pt_stbd = C_orig[nSweep + nSkip + nSkip, :]

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

                # # --- Rotate about wing root ---
                # vec = C[-ii + nSweep + nSkip - (nSkip + 1), :] - sweep_ref_pt
                # C[-ii + nSweep + nSkip - (nSkip + 1), :] = sweep_ref_pt + rot_mtx @ vec

                # vec = C[nSweep + nSkip + ii + (nSkip + 1), :] - sweep_ref_pt
                # C[nSweep + nSkip + ii + (nSkip + 1), :] = sweep_ref_pt + rot_mtxnew @ vec

                # --- Alternative sweep about offset root (much better) ---
                vec = C[-ii + nSweep + nSkip - (nSkip + 1), :] - sweep_ref_pt_port
                C[-ii + nSweep + nSkip - (nSkip + 1), :] = sweep_ref_pt_port + rot_mtx @ vec

                vec = C[nSweep + nSkip + ii + (nSkip + 1), :] - sweep_ref_pt_stbd
                C[nSweep + nSkip + ii + (nSkip + 1), :] = sweep_ref_pt_stbd + rot_mtxnew @ vec

            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, sweepAxis)

        model.geometry.nom_addGlobalDV(
            dvName="sweep",
            value=0.0,
            func=sweep_rot_func,
        )

    # ---------------------------
    #   DIHEDRAL
    # ---------------------------
    if "d" in args.geovar:  # not tested
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

        model.geometry.nom_addGlobalDV(
            dvName="dihedral",
            value=0.0,
            func=dihedral_rot_func,
        )

    # ---------------------------
    #   CHORD
    # ---------------------------
    if "r" in args.geovar:
        nSkip = 4 + 2

        nTaper = nRefAxPts // 2 + 1

        def taper(val, geo):
            s = geo.extractS("global")
            # slope = (val[1] - val[0]) / (s[-1] - s[0])
            slope = (val[1] - val[0]) / (s[-1] - s[nTaper - 1])
            for ii in range(nTaper):
                # geo.scale_x["global"].coef[ii] = slope * (s[ii] - s[0]) + val[0]

                geo.scale_x["global"].coef[ii + nTaper - 1] = slope * (s[ii + nTaper - 1] - s[nTaper - 1]) + val[0]
                geo.scale_x["global"].coef[nTaper - ii - 1] = -slope * (s[nTaper - ii - 1] - s[nTaper - 1]) + val[0]

        model.geometry.nom_addGlobalDV(
            "taper",
            value=np.ones(2) * 1.0,
            func=taper,
        )

    # ---------------------------
    #   Span
    # ---------------------------
    if "p" in args.geovar:
        # nSpan = nRefAxPts
        nSkip = 2
        nSpan = nRefAxPts // 2 - nSkip

        def span(val, geo):
            nSkip = 2
            C = geo.extractCoef("global")
            s = geo.extractS("global")

            for ii in range(nSpan):
                # stbd wing
                C[nRefAxPts // 2 + ii + nSkip + 1, 1] += val.item() * s[ii + nSpan - 1]
                # port
                C[-ii + nSpan - 1, 1] += -val.item() * s[ii + nSpan - 1]
            geo.restoreCoef(C, "global")

        model.geometry.nom_addGlobalDV(dvName="span", value=0.0, func=span)

    # # def airfoilThickness(val, geo):
    # #     # Set airfoil thickness values
    # #     for i in range(nSpanwise):
    # #         geo.scale_z["wing"].coef[i] = val[0]

    return model
