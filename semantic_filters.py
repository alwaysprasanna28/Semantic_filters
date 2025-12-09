import numpy as np
import cvxpy as cp
import robosuite.utils.transform_utils as T


class SemanticSafetyFilter:
  

    def __init__(self, env, robot_idx: int = 0, dt: float = 0.05):
        self.env = env
        self.robot_idx = robot_idx
        self.dt = dt

        # Optimization variables / parameters
        self.u = cp.Variable(6)          # [v_x, v_y, v_z, w_x, w_y, w_z]
        self.u_cmd = cp.Parameter(6)     # commanded twist

        # Optional orientation regularization weights (set to zero by default)
        self.w_rot_err = cp.Parameter(nonneg=True, value=0.0)
        self.w_rot_vel = cp.Parameter(nonneg=True, value=0.0)

        # Cached IDs and pose memory
        self.body_ids = {}
        self.R_des = None                # desired EEF orientation when grasping begins
        self.is_grasping_prev = False

    
    def _get_body_state(self, body_name):
        """
        Return (body_id, position, orientation_matrix) for a named body.
        If the body does not exist, returns (None, None, None).
        """
        sim = self.env.sim

        if body_name not in self.body_ids:
            try:
                self.body_ids[body_name] = sim.model.body_name2id(body_name)
            except ValueError:
                return None, None, None

        bid = self.body_ids[body_name]
        pos = np.array(sim.data.body_xpos[bid])
        rot_mat = np.array(sim.data.body_xmat[bid]).reshape(3, 3)
        return bid, pos, rot_mat

    def _get_body_dimensions(self, body_id):
        """
        Approximate object's size from its first geometry.  For mesh
        geoms (very small default size) we fall back to a reasonable
        default.
        """
        sim = self.env.sim
        geom_adr = sim.model.body_geomadr[body_id]

        # Default fallback in case dimensions are not informative
        default_size = np.array([0.15, 0.10, 0.02])

        if geom_adr == -1:
            return default_size

        dim = np.array(sim.model.geom_size[geom_adr])
        if np.max(dim) < 0.01:
            return default_size

        return dim

    def get_robot_state(self):
        robot = self.env.robots[self.robot_idx]
        grip_site_name = robot.gripper["right"].important_sites["grip_site"]
        site_id = self.env.sim.model.site_name2id(grip_site_name)

        eef_pos = np.array(self.env.sim.data.site_xpos[site_id])
        eef_rot = np.array(self.env.sim.data.site_xmat[site_id]).reshape(3, 3)
        return eef_pos, eef_rot


    def get_semantic_context(self, is_grasping: bool):
        """
        Encodes which semantic constraints are active for the current
        manipulated object.  For the demo scene we consider:

            manipulated object: "coke_main"
            unsafe spatial relation: ( "bluemug_main", above)
            unsafe behaviors: caution for both coke and mug
            pose constraint: constrained_rotation when grasping
        """
        semantic_context = {
            "spatial": {
                "bluemug_main": "above",
            },
            "behavior": {},
            "pose": "free_rotation",
        }

        # # "caution" behavior -> slower approach in alpha(h) if desired
        # semantic_context["behavior"]["laptop_main"] = "caution"
        # semantic_context["behavior"]["bluemug_main"] = "caution"

        if is_grasping:
            semantic_context["pose"] = "constrained_rotation"

        return semantic_context

    # ------------------------------------------------------------------
    # Superquadric helper functions
    # ------------------------------------------------------------------
    def get_laptop_column_superquadric(self, laptop_pos, laptop_dim,
                                       workspace_top_z: float = None):
     
        # Laptop top surface
        z_laptop_top = laptop_pos[2] + laptop_dim[2]

      
        if workspace_top_z is None:
            workspace_top_z = z_laptop_top + 0.6

        # Center of the column and its half-height
        z_center = 0.5 * (z_laptop_top + workspace_top_z)
        a_z = 0.5 * (workspace_top_z - z_laptop_top)

        # Slightly inflate footprint in x,y so we get clearance
        margin_xy = 0.03
        a_x = laptop_dim[0] + margin_xy
        a_y = laptop_dim[1] + margin_xy

        center = np.array([laptop_pos[0], laptop_pos[1], z_center])
        scale = np.array([a_x, a_y, a_z])

        # eps1 = eps2 = 0.5 -> superellipsoid with almost cylindrical sides
        epsilon = np.array([0.5, 0.5])

        return center, scale, epsilon

    def superquadric_value_and_gradient(self, x_ee, center, rot, scale, epsilon):
        """
        Compute value and gradient of the superquadric

            g(x) = ( (|τ1|/a_x)^(2/ε2) + (|τ2|/a_y)^(2/ε2) )^(ε2/ε1)
                   + (|τ3|/a_z)^(2/ε1),

        where τ = R^T (x - center).

        Inside:  g(x) < 1
        On surf: g(x) = 1
        Outside:g(x) > 1

        Returns:
            g_val : scalar
            grad_g_world : 3-vector (∂g/∂x) in world frame
        """
        rel_pos = rot.T @ (x_ee - center)  # local object frame
        x, y, z = rel_pos
        ax, ay, az = scale
        e1, e2 = epsilon

        # Normalised coordinates
        tx_base = np.abs(x) / (ax + 1e-6)
        ty_base = np.abs(y) / (ay + 1e-6)
        tz_base = np.abs(z) / (az + 1e-6)

        # Superquadric value
        T_xy = (tx_base ** (2.0 / e2)) + (ty_base ** (2.0 / e2))
        g_val = (T_xy ** (e2 / e1)) + (tz_base ** (2.0 / e1))

        # Gradient in local coordinates
        if T_xy > 1e-9:
            dg_dTxy = (e2 / e1) * (T_xy ** ((e2 / e1) - 1.0))
        else:
            dg_dTxy = 0.0

        # ∂T_xy / ∂x, ∂T_xy / ∂y
        ax_safe = ax + 1e-6
        ay_safe = ay + 1e-6
        az_safe = az + 1e-6

        # Note: for |t|^(2/ε), derivative is (2/ε)*|t|^{(2/ε)-1}*sign(t)
        dx_dtx = (2.0 / e2) * (1.0 / ax_safe) * (tx_base ** ((2.0 / e2) - 1.0)) * np.sign(x)
        dy_dty = (2.0 / e2) * (1.0 / ay_safe) * (ty_base ** ((2.0 / e2) - 1.0)) * np.sign(y)
        dz_dtz = (2.0 / e1) * (1.0 / az_safe) * (tz_base ** ((2.0 / e1) - 1.0)) * np.sign(z)

        grad_local = np.array([dg_dTxy * dx_dtx, dg_dTxy * dy_dty, dz_dtz])

        # Transform gradient back to world frame
        grad_world = rot @ grad_local
        return g_val, grad_world

    # ------------------------------------------------------------------
    # Grasp pose memory (for pose constraints if needed)
    # ------------------------------------------------------------------
    def update_grasp_state(self, is_grasping, current_rot):
        if is_grasping and not self.is_grasping_prev:
            # Lock desired orientation when grasp is first detected
            self.R_des = current_rot
        self.is_grasping_prev = is_grasping

    # ------------------------------------------------------------------
    # Main QP solve
    # ------------------------------------------------------------------
    def solve(self, user_action_vector, is_grasping: bool = False):
        
        self.u_cmd.value = user_action_vector[:6]
        eef_pos, eef_rot = self.get_robot_state()

        # Update grasp orientation memory
        self.update_grasp_state(is_grasping, eef_rot)
        semantic_context = self.get_semantic_context(is_grasping)
        # print(f"is_grasping:{is_grasping}")
        constraints = []
        active_violations = []

    
        pose_cost_expr = 0.0
        if semantic_context["pose"] == "constrained_rotation" and self.R_des is not None:
            # Desired orientation is the one at grasp time
            R_diff = self.R_des @ eef_rot.T
            axis_angle = T.quat2axisangle(T.mat2quat(R_diff))
            target_omega = axis_angle / max(self.dt, 1e-3)

            self.w_rot_err.value = 10.0
            self.w_rot_vel.value = 1.0

            pose_cost_expr = (
                self.w_rot_err * cp.sum_squares(self.u[3:] - target_omega)
                + self.w_rot_vel * cp.sum_squares(self.u[3:])
            )

        bluemug_bid, bluemug_pos, bluemug_rot = self._get_body_state("bluemug_main")
        bluemug_dim = self._get_body_dimensions(bluemug_bid) if bluemug_bid is not None else None

        
        enable_semantic_spatial = is_grasping

       
        if enable_semantic_spatial:
            relation = semantic_context["spatial"]["bluemug_main"]

            if relation == "above":
                # Superquadric column: unsafe interior above the laptop
                center, scale, epsilon = self.get_laptop_column_superquadric(
                    bluemug_pos, bluemug_dim
                )
                # print(f"center:{center}, scale:{scale}, epsilon:{epsilon}")
                g_val, grad_g = self.superquadric_value_and_gradient(
                    eef_pos, center, bluemug_rot, scale, epsilon
                )
                print(f"g_val: {g_val}")
                # CBF: h(x) = g(x) - 1 >= 0
                h_sem = g_val - 1.0
                alpha_sem = 1.0  

                # Only include constraint when numerically meaningful
                if g_val < 50.0:
                    constraints.append(grad_g @ self.u[:3] >= -alpha_sem * h_sem)

                    if g_val < 1.0:
                        active_violations.append("HIT: laptop_above_column")
                    elif g_val < 2.0:
                        active_violations.append("Near laptop_above_column")

        
        z_table_limit = 0.82
        h_table = eef_pos[2] - z_table_limit
        constraints.append(self.u[2] >= -5.0 * h_table)

        if h_table < 0.02:
            active_violations.append("Table")

        
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_cmd) + pose_cost_expr)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            return np.zeros(6), "QP Error"

        if prob.status not in ["optimal", "optimal_inaccurate"] or self.u.value is None:
            return np.zeros(6), "Blocked (Infeasible)"

        status_str = " | ".join(active_violations) if active_violations else "Free"
        return self.u.value, status_str
