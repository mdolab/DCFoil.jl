import numpy as np
import jax.numpy as jnp
import openmdao.api as om
import matplotlib.pyplot as plt


class DisplacementTransfer(om.JaxExplicitComponent):
    """
    Displacement transfer

    Parameters
    ----------
    collocationPts: ndarray, (3, n_strips)
        Coordinates of flow collocation points
        Sorted in spanwise direction from -b/2 to b/2
    nodes: ndarray, (n_node, 3)
        Coordinates of FEM nodes
    deflections: ndarray, (9 * n_node)
        Displacement at each FEM nodes (x, y, z, rx, ry, rz, rx_rate, ry_rate, rz_rate)

    Returns
    -------
    disp_colloc: ndarray, (6, n_strips)
        Displacement (x, y, z, rx, ry, rz) at each collocation point
        Sorted in spanwise direction from -b/2 to b/2
    """

    def initialize(self):
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')
        self.options.declare('n_strips', types=int, desc='Number of lifting line strips')
        self.options.declare('xMount', types=float, desc='subtract xMount from collocationPts x coordinates')

    def setup(self):
        n_node = self.options['n_node']
        n_strips = self.options['n_strips']

        # NOTE: ordering of declared inputs and outputs must match the compute_primal's signature
        self.add_input('collocationPts', shape=(3, n_strips))
        self.add_input('nodes', shape=(n_node, 3))
        self.add_input('deflections', shape=(9 * n_node))

        self.add_output('disp_colloc', shape=(6, n_strips))

        self.declare_partials('*', '*')

    def compute_primal(self, collocationPts, nodes, deflections):
        n_node = self.options['n_node']
        n_strips = self.options['n_strips']

        # shift collocation pts x axis to be consistent with FEM frame
        colloc_pts = collocationPts * 1.
        colloc_pts[0, :] -= self.options['xMount']  # shift x  # TODO: does this work in Jax?

        # reshape deflections to 2D array of shape (n_node, 9)
        disp = deflections.reshape(n_node, 9)
        disp_trans = disp[:, :3]
        disp_rot = disp[:, 3:6]

        # displacement at each collocation point
        disp_colloc = jnp.zeros((6, n_strips))

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
                # y-coord of the collocation point is exactly at one of the FEM nodes
                r = colloc_pts[:, i] - nodes[left_node_index, :]
                disp_translation = disp_trans[left_node_index, :] + jnp.cross(r, disp_rot[left_node_index, :])
                disp_rotation = disp_rot[left_node_index, :]
                disp_colloc = disp_colloc.at[:3, i].set(disp_translation)
                disp_colloc = disp_colloc.at[3:6, i].set(disp_rotation)
            else:
                # weighted sum of displacement from left and right adjacent nodes

                # compute weight factors (inverse of spanwise distance)
                d1 = jnp.abs(colloc_pts[1, i] - nodes[left_node_index, 1])
                d2 = jnp.abs(colloc_pts[1, i] - nodes[right_node_index, 1])
                w1 = d2 / (d1 + d2)
                w2 = d1 / (d1 + d2)
                # check consistency
                # if not jnp.allclose(w1 + w2, 1.0, 1e-8):
                #     raise ValueError("Weight factors do not sum to 1.")

                # translational and rotational displacements from left and right nodes
                r1 = colloc_pts[:, i] - nodes[left_node_index, :]
                disp_trans_1 = disp_trans[left_node_index, :] + jnp.cross(r1, disp_rot[left_node_index, :])
                disp_rot_1 = disp_rot[left_node_index, :]

                r2 = colloc_pts[:, i] - nodes[right_node_index, :]
                disp_trans_2 = disp_trans[right_node_index, :] + jnp.cross(r2, disp_rot[right_node_index, :])
                disp_rot_2 = disp_rot[right_node_index, :]

                # weighted sum of displacements
                disp_colloc = disp_colloc.at[:3, i].set(w1 * disp_trans_1 + w2 * disp_trans_2)
                disp_colloc = disp_colloc.at[3:6, i].set(w1 * disp_rot_1 + w2 * disp_rot_2)

        return (disp_colloc,)


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


class CLAlphaInterpolation(om.JaxExplicitComponent):
    """
    Interpolate CL_alpha from flow collocation points to FEM nodes

    Parameters
    ----------
    CL_alpha: ndarray, (n_strips)
        CL_alpha at each collocation point
    collocationPts: ndarray, (3, n_strips)
        Coordinates of force acting points
        Sorted in spanwise direction from -b/2 to b/2
    nodes: ndarray, (n_node, 3)
        Coordinates of FEM nodes
        Does not need to be sorted

    Returns
    -------
    CL_alpha_node: ndarray, (n_node,)
        CL_alpha at each FEM node
    """

    def initialize(self):
        self.options.declare('n_strips', types=int, desc='Number of lifting line strips')
        self.options.declare('n_node', types=int, desc='Number of FEM nodes')

    def setup(self):
        n_strips = self.options['n_strips']
        n_node = self.options['n_node']

        # NOTE: ordering of declared inputs and outputs must match the compute_primal's signature
        self.add_input('CL_alpha', shape=(n_strips,))
        self.add_input('collocationPts', shape=(3, n_strips))
        self.add_input('nodes', shape=(n_node, 3))

        self.add_output('CL_alpha_node', shape=(n_node,))

        self.declare_partials('CL_alpha_node', '*')

    def compute_primal(self, CL_alpha, collocationPts, nodes):
        # spanwise linear interpolation of CL_alpha
        CL_alpha_node = jnp.interp(nodes[:, 1], collocationPts[1, :], CL_alpha)
        return (CL_alpha_node,)


def test_displacement_transfer():
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

    # nodal displacements
    n_node_full = n_node * 2 - 1
    disp_x = np.concatenate((np.linspace(0., 0.01, n_node), np.linspace(0, 0.01, n_node)[1:]))  # np.sin(np.linspace(-np.pi, np.pi, n_node))
    disp_y = np.linspace(-1, 1, n_node_full) * 0  # np.cos(np.linspace(-np.pi, np.pi, n_node))
    disp_z = np.concatenate((np.linspace(0., 0.1, n_node), np.linspace(0, 0.1, n_node)[1:]))
    disp_rx = np.concatenate((np.linspace(0., 0.1, n_node), np.linspace(0, 0.1, n_node)[1:])) * 0
    disp_ry = np.concatenate((np.linspace(0., 0.1, n_node), np.linspace(0, 0.1, n_node)[1:])) * 1
    disp_rz = np.concatenate((np.linspace(0., 0.01, n_node), np.linspace(0, 0.01, n_node)[1:])) * 0
    disp_rxrate = np.zeros(n_node_full)
    disp_ryrate = np.zeros(n_node_full)
    disp_rzrate = np.zeros(n_node_full)
    disp = np.vstack((disp_x, disp_y, disp_z, disp_rx, disp_ry, disp_rz, disp_rxrate, disp_ryrate, disp_rzrate)).T.flatten()

    prob = om.Problem()
    prob.model.add_subsystem('disp_transfer', DisplacementTransfer(n_node=n_node_full, n_strips=n_strips, xMount=3.355), promotes=['*'])
    prob.setup()
    prob.set_val('nodes', nodes)
    prob.set_val('deflections', disp)
    prob.set_val('collocationPts', collocationPts)  # collocation points are sorted in spanwise direction

    prob.run_model()
    # prob.check_partials(compact_print=True)
    # om.n2(prob)

    # plot results
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    # FEM nodal discplacements
    nodes_FEM = prob.get_val('nodes')
    nodes_y = nodes_FEM[:, 1]
    disp_FEM = prob.get_val('deflections').reshape(n_node_full, 9)
    color = 'C0'
    ax[0, 0].plot(nodes_y, disp_FEM[:, 0], 'o', color=color)
    ax[1, 0].plot(nodes_y, disp_FEM[:, 1], 'o', color=color)
    ax[2, 0].plot(nodes_y, disp_FEM[:, 2], 'o', color=color)
    ax[0, 1].plot(nodes_y, disp_FEM[:, 3], 'o', color=color)
    ax[1, 1].plot(nodes_y, disp_FEM[:, 4], 'o', color=color)
    ax[2, 1].plot(nodes_y, disp_FEM[:, 5], 'o', color=color)

    # collocation points displacements
    colloc_y = prob.get_val('collocationPts')[1, :]
    disp_colloc = prob.get_val('disp_colloc')
    color = 'C1'
    ax[0, 0].plot(colloc_y, disp_colloc[0, :], 's', ms=4, color=color)
    ax[1, 0].plot(colloc_y, disp_colloc[1, :], 's', ms=4, color=color)
    ax[2, 0].plot(colloc_y, disp_colloc[2, :], 's', ms=4, color=color)
    ax[0, 1].plot(colloc_y, disp_colloc[3, :], 's', ms=4, color=color)
    ax[1, 1].plot(colloc_y, disp_colloc[4, :], 's', ms=4, color=color)
    ax[2, 1].plot(colloc_y, disp_colloc[5, :], 's', ms=4, color=color)

    ax[0, 0].set_ylabel('x')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('z')
    ax[0, 1].set_ylabel('rx')
    ax[1, 1].set_ylabel('ry')
    ax[2, 1].set_ylabel('rz')

    plt.tight_layout()
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


def test_CLalpha_transfer():
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

    # CL alpha values
    CL_alpha = np.random.random(n_strips)

    prob = om.Problem()
    prob.model.add_subsystem('CL_alpha_transfer', CLAlphaInterpolation(n_strips=n_strips, n_node=n_node * 2 - 1), promotes=['*'])
    prob.setup()
    prob.set_val('CL_alpha', CL_alpha)
    prob.set_val('collocationPts', collocationPts)
    prob.set_val('nodes', nodes)

    prob.run_model()

    # prob.check_partials(compact_print=True)

    om.n2(prob)

    # plot intepolated CL_alpha at FEM nodes
    plt.figure()
    plt.plot(collocationPts[1, :], CL_alpha, 'o-', ms=5, lw=1)
    CL_alpha_node = prob.get_val('CL_alpha_node')
    plt.plot(nodes[:, 1], CL_alpha_node, 's', ms=5, color='C1')
    plt.show()


if __name__ == '__main__':
    test_displacement_transfer()
    # test_load_transfer()
    # test_CLalpha_transfer()
