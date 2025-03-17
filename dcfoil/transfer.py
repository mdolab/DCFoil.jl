import numpy as np
import openmdao.api as om

import jax.numpy as jnp

import matplotlib.pyplot as plt

from openaerostruct.transfer.displacement_transfer_group import DisplacementTransferGroup


class DisplacementTransfer(om.Group):
    """
    Displacement transfer

    TODO: modify to use y-distance-inverse weight factors for the load transfer to be conservative (only minor difference, though)

    Parameters
    ----------
    ptVec : ndarray, (2 * n_secs * 3,)
        Geometry (LE and TE coordinates of geometry of each section, flattened
        Does not need to be sorted
    nodes: ndarray, (n_node, 3)
        Coordinates of FEM nodes
        Does not need to be sorted
    deflections : ndarray, (9 * n_node)
        Displacement at each FEM nodes (x, y, z, rx, ry, rz, rx_rate, ry_rate, rz_rate)
    

    Returns
    -------
    ptVec_def : ndarray, (2 * n_secs * 3,)
        deformed geometry
    """

    def initialize(self):
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')
        self.options.declare('n_secs', types=int, desc='Number of spanwise sections for geometry definition')

    def setup(self):
        # reshape PtVecs (flattened) to 3D array with shape = (2, n_secs, 3)
        self.add_subsystem('reshape_ptVec', ReshapePtVec(n_secs=self.options['n_secs']), promotes_inputs=['ptVec'])

        # reshape deflections (flattened) to 2D array with shape = (n_node, 9)
        self.add_subsystem('reshape_disp', ReshapeDeflections(n_node=self.options['n_node']), promotes_inputs=['deflections'])

        # passthru component for node for convenience
        self.add_subsystem(
            'passthru',
            om.ExecComp('nodes_out = nodes', has_diag_partials=True, shape=(self.options['n_node'], 3)),
            promotes_inputs=['nodes'],
        )

        # interpolate nodal displacements to hydro mesh vertices in spanwise direction
        self.add_subsystem('interp_disp', InterpolateDisplacement(n_node=self.options['n_node'], n_secs=self.options['n_secs']))
        self.connect('reshape_disp.deflections2D', 'interp_disp.disp', src_indices=om.slicer[:, :6])  # drop last 3 deformations (rot rates)
        self.connect('reshape_ptVec.ptVec3D', 'interp_disp.ptVec_y', src_indices=om.slicer[0, :, 1])  # assume LE and TE has the same spanwise coordinate
        self.connect('passthru.nodes_out', 'interp_disp.nodes_y', src_indices=om.slicer[:, 1])

        # morph hydro mesh
        surface = {'mesh': np.zeros((2, self.options['n_secs'], 3))}   # just need to pass this to tell OAS about the mesh shape
        self.add_subsystem('disp_transfer', DisplacementTransferGroup(surface=surface))
        self.connect('reshape_ptVec.ptVec3D', 'disp_transfer.mesh')
        self.connect('interp_disp.disp_ptVec', 'disp_transfer.disp')
        self.connect('reshape_ptVec.ptVec3D', 'disp_transfer.nodes', src_indices=om.slicer[0, :, :])  # assume LE and TE has the same spanwise coordinate

        # flatten deformed geometry
        self.add_subsystem('flatten_ptVec', FlattenPtVec3D(n_secs=self.options['n_secs']), promotes_outputs=[('ptVec', 'ptVec_def')])
        self.connect('disp_transfer.def_mesh', 'flatten_ptVec.ptVec3D')


class ReshapePtVec(om.ExplicitComponent):
    """
    Reshape flattened PtVec to 3D array of shape (2, n_secs, 3)
    """
    def initialize(self):
        self.options.declare('n_secs', types=int, desc='Number of spanwise sections for geometry definition')

    def setup(self):
        n_secs = self.options['n_secs']
        self.add_input('ptVec', shape=(3 * 2 * n_secs,))   # xyz coordinates for LE and TE for each section
        self.add_output('ptVec3D', shape=(2, n_secs, 3))

        rows = np.arange(2 * n_secs * 3)
        cols = np.arange(2 * n_secs * 3).reshape(2, n_secs, 3).flatten()
        self.declare_partials('*', '*', method='fd')   # rows=rows, cols=cols)  # TODO: CHECK

    def compute(self, inputs, outputs):
        n_secs = self.options['n_secs']
        outputs['ptVec3D'] = inputs['ptVec'].reshape(2, n_secs, 3)


class FlattenPtVec3D(om.ExplicitComponent):
    """
    Flatten 3D array of shape (2, n_secs, 3) to 1D array of shape (2 * n_secs * 3)
    """
    def initialize(self):
        self.options.declare('n_secs', types=int, desc='Number of spanwise sections for geometry definition')

    def setup(self):
        n_secs = self.options['n_secs']
        self.add_input('ptVec3D', shape=(2, n_secs, 3))
        self.add_output('ptVec', shape=(3 * 2 * n_secs,))   # xyz coordinates for LE and TE for each section
        self.declare_partials('*', '*', method='fd')   # rows=rows, cols=cols)  # TODO: CHECK - is it just diagonal?

    def compute(self, inputs, outputs):
        outputs['ptVec'] = inputs['ptVec3D'].flatten()


class ReshapeDeflections(om.ExplicitComponent):
    """
    Reshape flattened deflections to 2D array of shape (9, n_secs)
    """
    def initialize(self):
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')

    def setup(self):
        n_node = self.options['n_node']
        self.add_input('deflections', shape=(9 * n_node))
        self.add_output('deflections2D', shape=(n_node, 9))

        rows = np.arange(9 * n_node)
        cols = np.arange(9 * n_node).reshape(n_node, 9).flatten()
        self.declare_partials('*', '*', method='fd')   # , rows=rows, cols=cols)  # TODO: CHECK

    def compute(self, inputs, outputs):
        n_node = self.options['n_node']
        outputs['deflections2D'] = inputs['deflections'].reshape(n_node, 9)


class InterpolateDisplacement(om.JaxExplicitComponent):
    """
    Linearly interpolate displacements from FEM nodal values to hydro geometry spanwise sections
    """
    def initialize(self):
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')
        self.options.declare('n_secs', types=int, desc='Number of spanwise sections for geometry definition')

    def setup(self):
        n_node = self.options['n_node']
        n_secs = self.options['n_secs']

        self.add_input('nodes_y', shape=(n_node))  # spanwise coordinates of FEM nodes
        self.add_input('disp', shape=(n_node, 6))  # nodal displacements (x, y, z, rx, ry, rz)
        self.add_input('ptVec_y', shape=(n_secs))  # spanwise coordinates from ptVec at which we interpolate FEM nodal values

        self.add_output('disp_ptVec', shape=(n_secs, 6))

        self.declare_partials('*', '*')

    def compute_primal(self, nodes_y, disp, ptVec_y):
        # sort FEM nodes in spanwise direction
        sort_idx = jnp.argsort(nodes_y)

        # linear interpolation of each displacements
        x = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 0])
        y = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 1])
        z = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 2])
        rx = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 3])
        ry = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 4])
        rz = jnp.interp(ptVec_y, nodes_y[sort_idx], disp[sort_idx, 5])
        
        return jnp.vstack((x, y, z, rx, ry, rz)).T


class LoadTransfer(om.JaxExplicitComponent):
    """
    Load transfer from flow force distribution to FEM nodel loads

    Parameters
    ----------
    collocationPts: ndarray, (3, n_strips)
        Coordinates of force acting points
        Sorted in spanwise direction from -b/2 to b/2
    forces_hydro: ndarray, (3, n_strips)
        Force distribution at each collocation point
        Sorted in spanwise direction from -b/2 to b/2
    nodes: ndarray, (n_node, 3)
        Coordinates of FEM nodes
        Does not need to be sorted

    Returns
    -------
    loads_str: ndarray, (9 * n_node,)
        Nodal force to be applied to FEM
        First 6 entries (forces and moments) are computed, last 3 entries are zero
    """

    def initialize(self):
        self.options.declare('n_strips', types=int, desc='Number of lifting line strips')
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')
        self.options.declare('xMount', types=float, desc='subtract xMount from collocationPts x coordinates')

    def setup(self):
        n_strips = self.options['n_strips']
        n_node = self.options['n_node']

        # NOTE: ordering of declared inputs and outputs must match the compute_primal's signature
        self.add_input('collocationPts', shape=(3, n_strips))
        self.add_input('forces_hydro', shape=(3, n_strips))
        self.add_input('nodes', shape=(n_node, 3))

        self.add_output('loads_str', shape=(9 * n_node,))

        self.declare_partials('loads_str', '*')

    def compute_primal(self, collocationPts, forces_hydro, nodes):
        n_node = self.options['n_node']
        n_strips = self.options['n_strips']

        # shift collocation pts x axis to be consistent with FEM frame
        colloc_pts = collocationPts * 1.
        colloc_pts[0, :] -= self.options['xMount']  # shift x  # TODO: does this work in Jax?

        # nodal load array
        loads = jnp.zeros((9, n_node))  # [9, n_node]

        # TODO: vectorize
        for i in range(n_strips):
            # find adjacent nodes (in spanwise coordinate)
            y_dist = nodes[:, 1] - colloc_pts[1, i]

            # left node: max y_dist for y_dist <= 0
            left_y_ist = jnp.max(y_dist[y_dist <= 0])
            left_node_index = int(jnp.where(y_dist == left_y_ist)[0][0])
            # right node: min y_dist but y_dist > 0
            right_y_dist = jnp.min(y_dist[y_dist >= 0])
            right_node_index = int(jnp.where(y_dist == right_y_dist)[0][0])

            if left_node_index == right_node_index:
                # y-coord of the collocation point is exactly at one of the FEM nodes, so just transfer the loads to that node
                loads = loads.at[:3, left_node_index].add(forces_hydro[:, i])
                moment = jnp.cross(colloc_pts[:, i] - nodes[left_node_index, :], forces_hydro[:, i])
                loads = loads.at[3:6, left_node_index].add(moment)
            else:
                # distribute load to left and right adjacent nodes
                # compute weight factors (inverse of spanwise distance)
                d1 = jnp.abs(colloc_pts[1, i] - nodes[left_node_index, 1])
                d2 = jnp.abs(colloc_pts[1, i] - nodes[right_node_index, 1])
                w1 = d2 / (d1 + d2)
                w2 = d1 / (d1 + d2)
                # check consistency
                # if not jnp.allclose(w1 + w2, 1.0, 1e-8):
                #     raise ValueError("Weight factors do not sum to 1.")

                loads = loads.at[:3, left_node_index].add(forces_hydro[:, i] * w1)
                loads = loads.at[:3, right_node_index].add(forces_hydro[:, i] * w2)
                moment1 = jnp.cross(colloc_pts[:, i] - nodes[left_node_index, :], forces_hydro[:, i] * w1)
                moment2 = jnp.cross(colloc_pts[:, i] - nodes[right_node_index, :], forces_hydro[:, i] * w2)

                loads = loads.at[3:6, left_node_index].add(moment1)
                loads = loads.at[3:6, right_node_index].add(moment2)

        # flatten loads to 1D array
        loads_str = loads.flatten(order='F')
        return (loads_str,)


class OLD_DONOTUSE_LoadTransferInterpolation(om.JaxExplicitComponent):
    """
    Load transfer
    TODO: This does not satisfy load concistency, so do not use!!

    Parameters
    ----------
    forces_hydro: ndarray, (3, n_strips)
        Force distribution from lifting line model
    collocationPts: ndarray, (3, n_strips)
        Coordinates of force acting points
        Sorted in spanwise direction from -b/2 to b/2
    nodes: ndarray, (n_node, 3)
        Coordinates of FEM nodes
        Does not need to be sorted

    Returns
    -------
    loads_str: ndarray, (9 * n_node,)
        Nodal force to be applied to FEM
    """

    def initialize(self):
        self.options.declare('n_strips', types=int, desc='Number of lifting line strips')
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')
        self.options.declare('xMount', types=float, desc='subtract xMount from collocationPts x coordinates')

    def setup(self):
        n_strips = self.options['n_strips']
        n_node = self.options['n_node']

        # NOTE: ordering of declared inputs and outputs must match the compute_primal's signature
        self.add_input('forces_hydro', shape=(3, n_strips))
        self.add_input('collocationPts', shape=(3, n_strips))
        self.add_input('nodes', shape=(n_node, 3))

        self.add_output('loads_str', shape=(9 * n_node,))

        self.declare_partials('loads_str', '*')

    def compute_primal(self, forces_hydro, collocationPts, nodes):
        n_node = self.options['n_node']

        # linear interpolation of forces_hydro to have same spanwise discretization as FEM nodes
        fx_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], forces_hydro[0, :])
        fy_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], forces_hydro[1, :])
        fz_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], forces_hydro[2, :])
        f_interp = jnp.vstack((fx_interp, fy_interp, fz_interp))   # [3, n_node]

        # compute acting points of interpolated force
        # flip x axis of collcation pts and shift by xMount
        x_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], collocationPts[0, :] - self.options['xMount'])  # shift x  # TODO: might beed to modify this for JAX
        y_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], collocationPts[1, :])
        z_interp = jnp.interp(nodes[:, 1], collocationPts[1, :], collocationPts[2, :])
        nodes_along_aero_center = jnp.vstack((x_interp, y_interp, z_interp)).T  # [n_node, 3]

        # compute nodal moments
        r = nodes_along_aero_center - nodes
        moment = jnp.cross(r, f_interp.T).T

        # returns nodal forces and moments. add zero loads for additional 3 DOFs
        loads_str = jnp.vstack((f_interp, moment, jnp.zeros((3, n_node))))  # [9, n_node]
        return (loads_str.flatten(order='F'),)


def test_displacement_transfer():
    n_node = 7

    # FEM nodes and displacements
    nodes = np.zeros((n_node, 3))
    nodes[:, 1] = np.linspace(-0.333, 0.333, n_node)
    # shuffle order of FEM node
    # np.random.shuffle(nodes)
    # print(nodes[0, :])

    # nodal displacements
    disp_x = np.ones(n_node) * 0.5  # np.sin(np.linspace(-np.pi, np.pi, n_node))
    disp_y = np.linspace(-1, 1, n_node) * 0  # np.cos(np.linspace(-np.pi, np.pi, n_node))
    disp_z = np.linspace(-1, 1, n_node) * 0.01
    disp_rx = np.zeros(n_node)
    disp_ry = np.linspace(-1, 1, n_node) * 0.03
    disp_rz = np.zeros(n_node)
    disp_rxrate = np.zeros(n_node)
    disp_ryrate = np.zeros(n_node)
    disp_rzrate = np.zeros(n_node)
    disp = np.vstack((disp_x, disp_y, disp_z, disp_rx, disp_ry, disp_rz, disp_rxrate, disp_ryrate, disp_rzrate)).T.flatten()

    # geometry
    ptVec = np.array([-0.07, 0.0, 0.0, -0.0675, 0.037, 0.0, -0.065, 0.074, 0.0, -0.0625, 0.111, 0.0, -0.06, 0.148, 0.0, -0.0575, 0.185, 0.0, -0.055, 0.222, 0.0, -0.0525, 0.259, 0.0, -0.05, 0.296, 0.0, -0.0475, 0.333, 0.0, -0.0675, -0.037, 0.0, -0.065, -0.074, 0.0, -0.0625, -0.111, 0.0, -0.06, -0.148, 0.0, -0.0575, -0.185, 0.0, -0.055, -0.222, 0.0, -0.0525, -0.259, 0.0, -0.05, -0.296, 0.0, -0.0475, -0.333, 0.0, 0.07, 0.0, 0.0, 0.0675, 0.037, 0.0, 0.065, 0.074, 0.0, 0.0625, 0.111, 0.0, 0.06, 0.148, 0.0, 0.0575, 0.185, 0.0, 0.055, 0.222, 0.0, 0.0525, 0.259, 0.0, 0.05, 0.296, 0.0, 0.0475, 0.333, 0.0, 0.0675, -0.037, 0.0, 0.065, -0.074, 0.0, 0.0625, -0.111, 0.0, 0.06, -0.148, 0.0, 0.0575, -0.185, 0.0, 0.055, -0.222, 0.0, 0.0525, -0.259, 0.0, 0.05, -0.296, 0.0, 0.0475, -0.333, 0.0])
    n_secs = len(ptVec) // 6

    prob = om.Problem()
    prob.model.add_subsystem('disp_transfer', DisplacementTransfer(n_node=n_node, n_secs=n_secs), promotes=['*'])
    prob.setup()
    prob.set_val('nodes', nodes)
    prob.set_val('deflections', disp)
    prob.set_val('ptVec', ptVec)

    prob.run_model()

    # prob.check_partials(compact_print=True)

    om.n2(prob)

    # plot geometries in 3D
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    mesh_orig = prob.get_val('reshape_ptVec.ptVec3D')
    ax.plot(mesh_orig[0, :, 0], mesh_orig[0, :, 1], mesh_orig[0, :, 2], 'o', color='C0', ms=3)  # LE
    ax.plot(mesh_orig[1, :, 0], mesh_orig[1, :, 1], mesh_orig[1, :, 2], 'o', color='C0', ms=3)  # TE
    mesh_def = prob.get_val('ptVec_def')
    ax.plot(mesh_def[0, :, 0], mesh_def[0, :, 1], mesh_def[0, :, 2], 'o', color='C1', ms=3)  # LE
    ax.plot(mesh_def[1, :, 0], mesh_def[1, :, 1], mesh_def[1, :, 2], 'o', color='C1', ms=3)  # TE

    # plot original displacements at FEM nodes
    nodes_FEM = prob.get_val('nodes')
    nodes_y = nodes_FEM[:, 1]
    disp_FEM = prob.get_val('reshape_disp.deflections2D')
    mesh_hydro = prob.get_val('disp_transfer.mesh')
    mesh_y = mesh_hydro[0, :, 1]
    disp_hydro = prob.get_val('disp_transfer.disp')

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    color = 'C0'
    ax[0, 0].plot(nodes_y, disp_FEM[:, 0], 'o', color=color)
    ax[1, 0].plot(nodes_y, disp_FEM[:, 1], 'o', color=color)
    ax[2, 0].plot(nodes_y, disp_FEM[:, 2], 'o', color=color)
    ax[0, 1].plot(nodes_y, disp_FEM[:, 3], 'o', color=color)
    ax[1, 1].plot(nodes_y, disp_FEM[:, 4], 'o', color=color)
    ax[2, 1].plot(nodes_y, disp_FEM[:, 5], 'o', color=color)
    color = 'C1'
    ax[0, 0].plot(mesh_y, disp_hydro[:, 0], '*', color=color)
    ax[1, 0].plot(mesh_y, disp_hydro[:, 1], '*', color=color)
    ax[2, 0].plot(mesh_y, disp_hydro[:, 2], '*', color=color)
    ax[0, 1].plot(mesh_y, disp_hydro[:, 3], '*', color=color)
    ax[1, 1].plot(mesh_y, disp_hydro[:, 4], '*', color=color)
    ax[2, 1].plot(mesh_y, disp_hydro[:, 5], '*', color=color)

    plt.show()


def test_load_transfer():
    n_strips = 20
    n_node = 6   # per one side of beam. Total number of nodes = 2 * n_node - 1

    collocationPts = np.zeros((3, n_strips))
    collocationPts[1, :] = np.linspace(-1, 1, n_strips)
    collocationPts[0, :] = -0.5 + 3.355

    # FEM nodes and connectivity
    nodes_pos = np.linspace(0, 1, n_node)
    nodes_neg = np.linspace(0, -1, n_node)
    nodes_y = np.concatenate((nodes_pos, nodes_neg[1:]))
    nodes = np.zeros((n_node * 2 - 1, 3))
    nodes[:, 1] = nodes_y
    # print(nodes)

    # elem_conn = [[i, i + 1] for i in range(len(nodes) - 1)]
    # elem_conn = np.array(elem_conn) + 1  # index starts at 1 in Julia!!
    # elem_conn[n_node - 1, 0] = 1
    # print(elem_conn)
    
    forces_hydro = np.zeros((3, n_strips))
    forces_hydro[2, :] = np.sin(np.linspace(-np.pi, np.pi, n_strips)) + 0.3  # out of plane force

    prob = om.Problem()
    prob.model.add_subsystem('load_transfer', LoadTransfer(n_strips=n_strips, n_node=n_node * 2 - 1, xMount=3.355), promotes=['*'])
    prob.setup()
    prob.set_val('collocationPts', collocationPts)
    prob.set_val('nodes', nodes)
    prob.set_val('forces_hydro', forces_hydro)
    prob.run_model()

    ### prob.check_partials(compact_print=True, method='cs')

    # plot forces and moments
    col_pts = prob.get_val('collocationPts')
    f_hydro = prob.get_val('forces_hydro')
    loads = prob.get_val('loads_str').reshape(9, n_node * 2 - 1, order='F')
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    # plot original hydro forces
    ax[0, 0].plot(col_pts[1, :], f_hydro[0, :], 'o-', label='hydro')
    ax[1, 0].plot(col_pts[1, :], f_hydro[1, :], 'o-')
    ax[2, 0].plot(col_pts[1, :], f_hydro[2, :], 'o-')
    # plot FEM loads
    ax[0, 0].plot(nodes[:, 1], loads[0, :], 'o', color='C3', label='FEM')
    ax[1, 0].plot(nodes[:, 1], loads[1, :], 'o', color='C3')
    ax[2, 0].plot(nodes[:, 1], loads[2, :], 'o', color='C3')
    ax[0, 1].plot(nodes[:, 1], loads[3, :], 'o', color='C3')
    ax[1, 1].plot(nodes[:, 1], loads[4, :], 'o', color='C3')
    ax[2, 1].plot(nodes[:, 1], loads[5, :], 'o', color='C3')
    # horizontal line at zero
    for axeach in ax.flatten():
        axeach.axhline(0, color='darkgray', lw=0.5)

    ax[0, 0].set_ylabel('Fx')
    ax[1, 0].set_ylabel('Fy')
    ax[2, 0].set_ylabel('Fz')
    ax[0, 1].set_ylabel('Mx')
    ax[1, 1].set_ylabel('My')
    ax[2, 1].set_ylabel('Mz')

    ax[0, 0].legend()

    # check force consistency
    fx_sum_hydro = np.sum(f_hydro[2, :])
    fx_sum_str = np.sum(loads[2, :])
    print('Sum of hydro forces (Fx):', fx_sum_hydro)
    print('Sum of FEM nodal forces (Fx):', fx_sum_str)
    print('Diff:', fx_sum_hydro - fx_sum_str, ' (should be 0)')

    mx_sum_str = np.sum(loads[3, :])
    print('Sum of FEM moments (Mx):', mx_sum_str, ' (should be 0)')

    plt.show()


if __name__ == '__main__':
    test_load_transfer()
    # test_displacement_transfer()
