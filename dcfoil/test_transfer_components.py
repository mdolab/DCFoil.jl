"""
Test load and displacement transfer components
"""

import unittest
import numpy as np
import openmdao.api as om

from transfer import DisplacementTransfer, LoadTransfer


class TestTransfer(unittest.TestCase):
    def setUp(self):
        # --- set test geometry ---
        # flow points
        collocation_pts = np.array(
            [
                [
                    -0.06920145,
                    -0.06920271,
                    -0.06920509,
                    -0.06920838,
                    -0.06921202,
                    -0.06921536,
                    -0.06921746,
                    -0.06921884,
                    -0.06921917,
                    -0.06921905,
                    -0.06921905,
                    -0.06921917,
                    -0.06921884,
                    -0.06921746,
                    -0.06921536,
                    -0.06921202,
                    -0.06920838,
                    -0.06920509,
                    -0.06920271,
                    -0.06920145,
                ],
                [
                    -0.89446168,
                    -0.85095485,
                    -0.76819995,
                    -0.65429758,
                    -0.52039714,
                    -0.37960578,
                    -0.24570513,
                    -0.13180241,
                    -0.04904724,
                    -0.00554027,
                    0.00554027,
                    0.04904724,
                    0.13180241,
                    0.24570513,
                    0.37960578,
                    0.52039714,
                    0.65429758,
                    0.76819995,
                    0.85095485,
                    0.89446168,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        # structure nodes
        nodes_y = [0, 0.225, 0.45, 0.675, 0.9, -0.225, -0.45, -0.675, -0.9]
        nodes = np.zeros((9, 3))
        nodes[:, 1] = nodes_y

        n_node = nodes.shape[0]
        n_strips = collocation_pts.shape[1]
        xMount = 0.0

        self.prob = om.Problem(reports=False)
        self.prob.model.add_subsystem(
            "displacement_transfer",
            DisplacementTransfer(n_node=n_node, n_strips=n_strips, xMount=xMount, hack_rot_X=False),
            promotes=["*"],
        )
        self.prob.model.add_subsystem(
            "load_transfer", LoadTransfer(n_node=n_node, n_strips=n_strips, xMount=xMount), promotes=["*"]
        )
        self.prob.setup()

        # set inputs
        self.prob.set_val("nodes", nodes)
        self.prob.set_val("collocationPts", collocation_pts)

        deflections = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.1927569045321779e-06,
            1.0984047738126991e-12,
            0.011557862285685425,
            -0.11157307814321313,
            0.08531751535644547,
            -1.1359607239494064e-05,
            -0.5769544869163892,
            0.5525977647070861,
            -6.193224592384721e-05,
            5.013615363168067e-06,
            4.073963110515076e-13,
            0.04875374382245673,
            -0.2112197315707266,
            0.20218185511254783,
            -2.1882255801146763e-05,
            -0.3266707921256885,
            0.44282486697427437,
            -3.6170800638173246e-05,
            1.064317734906179e-05,
            -1.3796186693974942e-12,
            0.1031627233422552,
            -0.26579123466573196,
            0.27921523402103304,
            -2.725912410792279e-05,
            -0.15484004118422837,
            0.24774980117872228,
            -1.0526303545319133e-05,
            1.6897951756440035e-05,
            -3.6201233970901165e-13,
            0.16612301787117145,
            -0.29146525401122303,
            0.322353812822749,
            -2.7848955196643185e-05,
            -0.08897662665456478,
            0.16410272607924087,
            4.13950525328182e-06,
            1.1927541964871855e-06,
            -2.74015939374175e-12,
            0.0115578497012163,
            0.11157295625736594,
            0.08531742177479182,
            1.1359582988094832e-05,
            0.576953847663251,
            0.5525971534793244,
            6.193212217497655e-05,
            5.013604763573339e-06,
            -3.273529875790199e-12,
            0.048753690351619444,
            0.21121949906142845,
            0.20218163182172283,
            2.1882212554359755e-05,
            0.3266704340144425,
            0.4428243783062397,
            3.617075158164511e-05,
            1.064315637193031e-05,
            -2.545869378174126e-12,
            0.1031626100937538,
            0.2657909434846472,
            0.27921492719315066,
            2.7259077871011354e-05,
            0.15483988381598224,
            0.2477495411440651,
            1.0526324502136073e-05,
            1.6897921103672452e-05,
            -4.3413439634257375e-12,
            0.16612283599360067,
            0.2914649376432247,
            0.32235346218749206,
            2.7848916581305804e-05,
            0.0889765401283878,
            0.1641025628962493,
            -4.139456411195874e-06,
        ]
        forces_hydro = np.array(
            [
                [
                    3.54002804e01,
                    1.48402972e02,
                    1.56125150e02,
                    8.20824126e01,
                    -3.16630575e01,
                    -1.27831350e02,
                    -1.80926493e02,
                    -1.63087661e02,
                    -1.15283580e02,
                    -4.31288575e01,
                    -4.31287109e01,
                    -1.15283524e02,
                    -1.63087782e02,
                    -1.80926784e02,
                    -1.27831593e02,
                    -3.16629889e01,
                    8.20828204e01,
                    1.56125626e02,
                    1.48403332e02,
                    3.54003601e01,
                ],
                [
                    -1.24568895e-02,
                    5.64761455e-03,
                    6.26256515e-03,
                    3.30804289e-03,
                    -1.19536196e-03,
                    -3.97557408e-03,
                    -3.92296305e-03,
                    -2.44013961e-03,
                    -4.16497401e-04,
                    3.63390561e-04,
                    -2.22712568e-04,
                    5.76558118e-05,
                    1.60978333e-03,
                    3.42202105e-03,
                    3.40427548e-03,
                    1.16079886e-03,
                    -3.41652575e-03,
                    -6.94292425e-03,
                    -6.90279174e-03,
                    1.14692967e-02,
                ],
                [
                    1.32991778e02,
                    9.66227715e02,
                    2.01421776e03,
                    2.79662515e03,
                    3.03706980e03,
                    2.74495317e03,
                    2.13150139e03,
                    1.49531672e03,
                    8.93635394e02,
                    3.02250905e02,
                    3.02249872e02,
                    8.93634739e02,
                    1.49531699e03,
                    2.13150306e03,
                    2.74495670e03,
                    3.03707378e03,
                    2.79662922e03,
                    2.01421957e03,
                    9.66228363e02,
                    1.32991839e02,
                ],
            ]
        )
        self.prob.set_val("deflections", deflections)
        self.prob.set_val("forces_hydro", forces_hydro)

        # run transfer
        self.prob.run_model()

    def test_consistency(self):
        # Test consistency
        forces_hydro = self.prob.get_val("forces_hydro")
        loads_str = self.prob.get_val("loads_str")

        # sum of hydro forces
        force_x = np.sum(forces_hydro[0, :])
        force_y = np.sum(forces_hydro[1, :])
        force_z = np.sum(forces_hydro[2, :])

        # structural loads
        str_loads_x = loads_str[0::9]
        str_loads_y = loads_str[1::9]
        str_loads_z = loads_str[2::9]

        # check force consistency
        self.assertAlmostEqual(force_x, np.sum(str_loads_x), places=5)
        self.assertAlmostEqual(force_y, np.sum(str_loads_y), places=5)
        self.assertAlmostEqual(force_z, np.sum(str_loads_z), places=5)

    def test_conservation(self):
        # Test virtual work conservation

        # virtual work on structure
        loads_str = self.prob.get_val("loads_str")
        disp_str = self.prob.get_val("deflections")
        work_str = np.sum(loads_str * disp_str)

        # virtual work on hydro forces
        forces_hydro = self.prob.get_val("forces_hydro")
        disp_hydro = self.prob.get_val("disp_colloc")[:3, :]  # just use translational part
        work_hydro = np.sum(forces_hydro * disp_hydro)

        # check work conservation
        self.assertAlmostEqual(work_str, work_hydro, places=5)


if __name__ == "__main__":
    unittest.main()
