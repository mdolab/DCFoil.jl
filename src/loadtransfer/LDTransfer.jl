# --- Julia 1.11---
"""
@File          :   LDTransfer.jl
@Date created  :   2025-01-30
@Last modified :   2026-10-13
@Author        :   Galen Ng
@Desc          :   Load and displacement transfer routines between the lifting line collocation points and the FEM beam nodes.
                   Ported from the JAX implementation in dcfoil/transfer.py
"""


module LDTransfer

using LinearAlgebra: cross

export transfer_displacements, transfer_loads, interp_cla

function find_adjacent_nodes(nodesY, y)
    """
    Find the FEM nodes adjacent (in spanwise coordinate) to the point y

    Returns
    -------
        iL : index of node with the max y_dist where y_dist <= 0
        iR : index of node with the min y_dist where y_dist >= 0
    If y is exactly at a node, iL == iR
    """
    iL = 0
    iR = 0
    distL = -Inf
    distR = Inf
    for (ii, yNode) in enumerate(nodesY)
        y_dist = yNode - y
        # strict inequalities so ties go to the first index (like jnp.argmax/argmin)
        if y_dist <= 0 && y_dist > distL
            distL = y_dist
            iL = ii
        end
        if y_dist >= 0 && y_dist < distR
            distR = y_dist
            iR = ii
        end
    end

    # mimic jnp.argmax/argmin returning the first index when everything is masked out
    iL = iL == 0 ? 1 : iL
    iR = iR == 0 ? 1 : iR

    return iL, iR
end

function compute_weights(y, yL, yR)
    """
    Weight factors (inverse of spanwise distance) for the left and right nodes
    """
    d1 = abs(y - yL)
    d2 = abs(y - yR)
    w1 = d2 / (d1 + d2)
    w2 = d1 / (d1 + d2)
    return w1, w2
end

function shift_colloc(collocationPts, xMount)
    """
    Shift collocation pts x axis to be consistent with the FEM frame
    """
    colloc_pts = copy(collocationPts)
    colloc_pts[1, :] .-= xMount
    return colloc_pts
end

function transfer_displacements(collocationPts, nodes, deflections; xMount=0.0, config="full-wing", hack_rot_X=true)
    """
    Displacement transfer from FEM nodes to flow collocation points

    Inputs
    ------
        collocationPts : (3, n_strips) coordinates of flow collocation points, sorted from -b/2 to b/2
        nodes : (n_node, 3) coordinates of FEM nodes
        deflections : (9 * n_node) displacements at each FEM node (x, y, z, rx, ry, rz, rx_rate, ry_rate, rz_rate)

    Returns
    -------
        disp_colloc : (6, n_strips_full) displacements (x, y, z, rx, ry, rz) at each collocation point
            Always the full span, even if config is 'wing'
    """
    n_node = size(nodes, 1)
    n_strips = size(collocationPts, 2)

    colloc_pts = shift_colloc(collocationPts, xMount)

    # (9, n_node) in column-major is the transpose of python's reshape(n_node, 9)
    disp = reshape(deflections, 9, n_node)
    disp_trans = disp[1:3, :]
    disp_rot = disp[4:6, :]

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # HACK: flip sign of X rotation here because the beam model seems to flip it internally
    # TODO: undo this change once the beam model is fixed
    # NOTE: This hack fails work conservation unit test, so when computing virtual work, hack_rot_X should be turned off
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if hack_rot_X
        disp_rot[1, :] .*= -1
    end

    dtype = promote_type(eltype(colloc_pts), eltype(nodes), eltype(deflections))
    disp_colloc = zeros(dtype, 6, n_strips)

    for ii in 1:n_strips
        iL, iR = find_adjacent_nodes(view(nodes, :, 2), colloc_pts[2, ii])

        if iL == iR
            # y-coord of the collocation point is exactly at one of the FEM nodes
            r = colloc_pts[:, ii] - nodes[iL, :]
            disp_colloc[1:3, ii] = disp_trans[:, iL] + cross(disp_rot[:, iL], r)
            disp_colloc[4:6, ii] = disp_rot[:, iL]
        else
            # weighted sum of displacement from left and right adjacent nodes
            w1, w2 = compute_weights(colloc_pts[2, ii], nodes[iL, 2], nodes[iR, 2])

            r1 = colloc_pts[:, ii] - nodes[iL, :]
            disp_trans_1 = disp_trans[:, iL] + cross(disp_rot[:, iL], r1)

            r2 = colloc_pts[:, ii] - nodes[iR, :]
            disp_trans_2 = disp_trans[:, iR] + cross(disp_rot[:, iR], r2)

            disp_colloc[1:3, ii] = w1 * disp_trans_1 + w2 * disp_trans_2
            disp_colloc[4:6, ii] = w1 * disp_rot[:, iL] + w2 * disp_rot[:, iR]
        end
    end

    if config == "wing"
        # for symmetry case, we still need to return disp_colloc for the full span
        disp_colloc = hcat(reverse(disp_colloc, dims=2), disp_colloc)
    end

    return disp_colloc
end

function transfer_loads(collocationPts, forces_hydro, nodes; xMount=0.0, config="full-wing")
    """
    Load transfer from flow force distribution to FEM nodal loads

    Inputs
    ------
        collocationPts : (3, n_strips) coordinates of force acting points, sorted from -b/2 to b/2
        forces_hydro : (3, n_strips) force at each collocation point
        nodes : (n_node, 3) coordinates of FEM nodes, does not need to be sorted

    Returns
    -------
        loads_str : (9 * n_node) nodal loads to be applied to FEM
            First 6 entries per node (forces and moments) are computed, last 3 entries are zero
    """
    n_node = size(nodes, 1)
    n_strips = size(collocationPts, 2)

    colloc_pts = shift_colloc(collocationPts, xMount)

    dtype = promote_type(eltype(colloc_pts), eltype(forces_hydro), eltype(nodes))
    loads = zeros(dtype, 9, n_node)

    # x and z moments have opposite sign on the mirrored side so they cancel out
    symFlip = [-1, 1, -1]

    for ii in 1:n_strips
        iL, iR = find_adjacent_nodes(view(nodes, :, 2), colloc_pts[2, ii])
        force = forces_hydro[:, ii]

        if iL == iR
            # y-coord of the collocation point is exactly at one of the FEM nodes, so just transfer the loads to that node
            loads[1:3, iL] .+= force
            loads[4:6, iL] .+= cross(colloc_pts[:, ii] - nodes[iL, :], force)
        else
            # distribute load to left and right adjacent nodes
            w1, w2 = compute_weights(colloc_pts[2, ii], nodes[iL, 2], nodes[iR, 2])

            moment1 = cross(colloc_pts[:, ii] - nodes[iL, :], force * w1)
            moment2 = cross(colloc_pts[:, ii] - nodes[iR, :], force * w2)

            loads[1:3, iL] .+= force * w1
            loads[1:3, iR] .+= force * w2
            loads[4:6, iL] .+= moment1
            loads[4:6, iR] .+= moment2

            # for symmetry case, we need to add loads from the other side to the beam root node
            if config == "wing"
                if iL == 1
                    loads[1:3, iL] .+= force * w1
                    loads[4:6, iL] .+= moment1 .* symFlip
                end
                if iR == 1
                    loads[1:3, iR] .+= force * w2
                    loads[4:6, iR] .+= moment2 .* symFlip
                end
            end
        end
    end

    # same as python's flatten(order="F")
    return vec(loads)
end

function interp_linear_extrap(xq, xp, fp)
    """
    Linear interpolation with linear extrapolation off both ends (like jnp.interp with left/right="extrapolate")
    xp must be increasing
    """
    nP = length(xp)
    # segment index: clamp to the first/last segment for extrapolation
    jj = clamp(searchsortedlast(xp, xq), 1, nP - 1)
    slope = (fp[jj+1] - fp[jj]) / (xp[jj+1] - xp[jj])
    return fp[jj] + slope * (xq - xp[jj])
end

function interp_cla(CL_alpha, collocationPts, nodes)
    """
    Spanwise linear interpolation of CL_alpha from flow collocation points to FEM nodes

    Inputs
    ------
        CL_alpha : (n_strips) CL_alpha at each collocation point
        collocationPts : (3, n_strips) sorted from -b/2 to b/2
        nodes : (n_node, 3) coordinates of FEM nodes, does not need to be sorted

    Returns
    -------
        CL_alpha_node : (n_node) CL_alpha at each FEM node
    """
    colloc_y = collocationPts[2, :]
    return [interp_linear_extrap(yNode, colloc_y, CL_alpha) for yNode in nodes[:, 2]]
end

end
