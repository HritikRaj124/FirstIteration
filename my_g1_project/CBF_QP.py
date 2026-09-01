import numpy as np
import proxsuite

def solve_safety_qp(qdot_des, h_list, grad_h_list, alpha=5.0, qdot_max=2.0):
    """
    qdot_des : (7,) desired joint velocity from the nominal controller
    h_val    : scalar, h(q) at the current joint state
    grad_h   : (7,) gradient of h w.r.t. arm_q, from jax.grad(h_fn)(arm_q)
    alpha    : CBF class-K gain (linear case: alpha(h) = alpha * h)
    qdot_max : scalar or (7,) placeholder joint velocity limit (rad/s)

    Returns qdot_safe : (7,) filtered joint velocity command
    """
    n = qdot_des.shape[0]  # 7 arm joints
    n_rows = len(h_list)

    # cost: ||qdot - qdot_des||^2 -> H = 2I, g = -2*qdot_des
    H = 2.0 * np.eye(n)
    g = -2.0 * np.asarray(qdot_des)

    # CBF constraint row: grad_h(q)^T qdot >= -alpha * h(q)
    C = np.stack([np.asarray(gh) for gh in grad_h_list], axis=0)
    l = np.array([-alpha * h for h in h_list])
    u = np.full(n_rows, np.inf)

    # box constraint: qdot_min <= qdot <= qdot_max
    qdot_max_vec = np.full(n, qdot_max) if np.isscalar(qdot_max) else np.asarray(qdot_max)
    l_box = -qdot_max_vec
    u_box = qdot_max_vec

    qp = proxsuite.proxqp.dense.QP(n, 0, n_rows, box_constraints=True)
    qp.init(H, g, None, None, C, l, u, l_box=l_box, u_box=u_box)
    qp.solve()

    if qp.results.info.status != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
        # Fail safe: command zero velocity rather than a possibly-unsafe result.
        print(f"  [WARNING] QP did not solve cleanly: {qp.results.info.status}")
        return np.zeros(n)

    return qp.results.x

