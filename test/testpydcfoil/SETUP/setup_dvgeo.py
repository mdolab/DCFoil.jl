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
    # rst Create DVGeometry objects
    FFDFile_global = files["ffdFile0"]
    DVGeo = DVGeometry(FFDFile_global)
    DVGeo_foil = DVGeometry(files["ffdFile1"], child=True)  # Child DVGeometry

    # rst Create reference axis
    nRefAxPts_foil = DVGeo_foil.addRefAxis("foil", xFraction=0.25, alignIndex="k")

    DVGeo.addChild(DVGeo_foil)
    nRefAxPts_global = DVGeo.addRefAxis("global", xFraction=1.0, alignIndex="k")

    # --- Design variables for DVgeometry ---
    # ==============================================================================
    #                         DVS
    # ==============================================================================
    # ---------------------------
    #   TWIST
    # ---------------------------
    if "t" in args.geovar_foil:
        if comm.rank == 0:
            print(f"{nTwist} foil twist vars", flush=True)

        def twist_rottheta_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            rottheta method is but the curve
            """

            for ii in range(nTwist):
                geo.rot_theta["c4_v0"].coef[ii + n_skip] = val[ii]

                if args.config in ["full", "fs"]:
                    geo.rot_theta["c4_v1"].coef[ii + n_skip] = val[ii]

        def twist_roty_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            """
            for ii in range(nTwist):
                geo.rot_y["c4_v0"].coef[ii + n_skip] = val[ii]

                if args.config in ["full", "fs"]:
                    geo.rot_y["c4_v1"].coef[ii + n_skip] = val[ii]

        DVGeoFoil.addGlobalDV(
            "foil_twist",
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
    if "w" in args.geovar_foil:
        # Determine the number of sections that have sweep control
        nSweep = nRefAxPts1

        def sweep_rot_func(val, geo):
            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef("c4_v0")
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            sweep_ref_pt = C_orig[0, :]

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

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nRefAxPts1):
                # get the vector from each ref axis point to the wing root
                vec = C[ii, :] - sweep_ref_pt
                # need to now rotate this by the sweep angle and add back the wing root loc
                C[ii, :] = sweep_ref_pt + rot_mtx @ vec

            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, "c4_v0")

            if args.config in ["full", "fs"]:
                C2 = geo.extractCoef("c4_v1")
                C_orig = C2.copy()
                # we will sweep the wing about the first point in the ref axis
                sweep_ref_pt = C_orig[0, :]

                theta = val[0] * np.pi / 180
                cc = np.cos(theta)
                ss = np.sin(theta)

                for ii in range(nRefAxPts2):
                    # get the vector from each ref axis point to the wing root
                    vec = C2[ii, :] - sweep_ref_pt
                    # need to now rotate this by the sweep angle and add back the wing root loc
                    C2[ii, :] = sweep_ref_pt + rot_mtx @ vec

                # use the restoreCoef method to put the control points back in the right place
                geo.restoreCoef(C2, "c4_v1")

        DVGeoFoil.addGlobalDV(
            dvName="sweep",
            value=0.0,
            func=sweep_rot_func,
            lower=-5.0,
            upper=5.0,
            scale=1,
        )

    # ---------------------------
    #   DIHEDRAL
    # ---------------------------
    if "d" in args.geovar_foil:
        nDihedral = nRefAxPts1

        def dihedral_rot_func(val, geo):
            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef("c4_v0")
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
            for ii in range(nRefAxPts1):
                # get the vector from each ref axis point to the wing root
                vec = C[ii, :] - sweep_ref_pt
                # need to now rotate this by the sweep angle and add back the wing root loc
                C[ii, :] = sweep_ref_pt + rot_mtx @ vec
            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, "c4_v0")

            if args.config in ["full", "fs"]:
                C2 = geo.extractCoef("c4_v1")
                C_orig = C2.copy()
                # we will sweep the wing about the first point in the ref axis
                sweep_ref_pt = C_orig[0, :]

                theta = val[0] * np.pi / 180
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
                for ii in range(nRefAxPts2):
                    # get the vector from each ref axis point to the wing root
                    vec = C2[ii, :] - sweep_ref_pt
                    # need to now rotate this by the sweep angle and add back the wing root loc
                    C2[ii, :] = sweep_ref_pt + rot_mtx @ vec
                # use the restoreCoef method to put the control points back in the right place
                geo.restoreCoef(C2, "c4_v1")

        DVGeoFoil.addGlobalDV(
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
    if "r" in args.geovar_foil:

        def chords(val, geo):
            # Set all the global chord values
            for i in range(4, nRefAxPts_global):
                geo.scale["global"].coef[i] = val[i - 4]

        DVGeo.addGlobalDV(
            "chord",
            value=np.ones(nRefAxPts_global - 4) * 0.1,
            func=chords,
            lower=0.01,
            upper=0.2,
            scale=1.0,
        )

        def taper(val, geo):
            s = geo.extractS("wing")
            slope = (val[1] - val[0]) / (s[-1] - s[0])
            for i in range(nRefAxPts):
                geo.scale_x["wing"].coef[i] = slope * (s[i] - s[0]) + val[0]

        DVGeo.addGlobalDV(
            "taper",
            value=[0.1, 0.1],
            func=taper,
            lower=[0.01, 0.01],
            upper=[0.2, 0.2],
            scale=1.0,
        )

    # ---------------------------
    #   Span
    # ---------------------------
    if "b" in args.geovar_foil:
        def span(val, geo):
            C = geo.extractCoef("wing")
            s = geo.extractS("wing")
            for i in range(1, nRefAxPts):
                C[i, 2] += val * s[i]
            geo.restoreCoef(C, "wing")

        DVGeo.addGlobalDV(dvName="span", value=0.0, func=span, lower=-10.0, upper=20.0, scale=0.1)
    
    # def airfoilThickness(val, geo):
    #     # Set airfoil thickness values
    #     for i in range(nSpanwise):
    #         geo.scale_z["wing"].coef[i] = val[0]

    return DVGeo
