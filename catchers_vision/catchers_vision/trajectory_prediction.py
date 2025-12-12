import matplotlib.pyplot as plt
import modern_robotics as mr
import numpy as np
import tf_transformations as tf


def angle_between(a, b):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    a : array-like
        First input vector.
    b : array-like
        Second input vector.

    Returns
    -------
    float
        Angle between the two vectors in radians.

    """
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)


class LSMADParabola:
    """Initialize nessesary parameters for parabola fitting."""

    def __init__(
        self,
        x_bounds,
        y_bounds,
        z_bounds,
        N=5,
        N_best=3,
        v_gate=10,
        window_size=15,
        gate_residual_thresh=0.2,
        min_inliers_for_gate=5
    ):
        self.theta_i = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.theta = self.theta_i.copy()
        self.t_i = None
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.t_list = []
        self.N = N
        self.N_best = N_best
        self.bounds = np.array(x_bounds + y_bounds + z_bounds)
        self.meas_prev = None
        self.window_size = window_size
        self.v_gate = v_gate
        self.v_gate_active = False
        self.gate_residual_thresh = gate_residual_thresh
        self.min_inliers_for_gate = min_inliers_for_gate

    def LS_MAD(self, t, x, y, z, N_best=3, k=3, use_recency_weights=True):
        """
        Perform LS-MAD fitting of a 3D parabolic trajectory using optional \
        recency-weighted least squares.

        Parameters
        ----------
        t : array-like
            Time stamps corresponding to each measurement.
        x : array-like
            Measured x positions.
        y : array-like
            Measured y positions.
        z : array-like
            Measured z positions.
        N_best : int, default=3
            Number of lowest-residual points retained for the final refit.
        k : float, default=3
            Scaling factor for the MAD-based inlier threshold.
        use_recency_weights : bool, default=True
            If True, weight recent measurements more heavily in the least-squares fit.

        Returns
        -------
        None
            Updates internal trajectory parameters and gating state.

        """
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        t = np.array(t)
        H_list = []

        if self.t_i is None:
            self.t_i = t[0]
        t = t - self.t_i
        n_pts = len(t)

        weights = np.linspace(1.0, 7.0, n_pts)

        for ti in t:
            H_i = np.array(
                [
                    [ti, 1, 0, 0, 0, 0, 0],
                    [0, 0, ti, 1, 0, 0, 0],
                    [0, 0, 0, 0, ti**2, ti, 1],
                ]
            )
            H_list.append(H_i)
        H = np.vstack(H_list)
        y_full = np.vstack([x, y, z]).T.reshape(-1)

        if use_recency_weights:
            w_rows = np.repeat(weights, 3)
            w_sqrt = np.sqrt(w_rows)
            H_w = H * w_sqrt[:, None]
            y_w = y_full * w_sqrt
            theta_ls = np.linalg.lstsq(H_w, y_w, rcond=None)[0]
        else:
            theta_ls = np.linalg.lstsq(H, y_full, rcond=None)[0]

        r_full = y_full - H @ theta_ls
        r_xyz = r_full.reshape(-1, 3)
        residuals = np.linalg.norm(r_xyz, axis=1)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))

        if mad == 0:
            mask = np.ones(n_pts, dtype=bool)
        else:
            thresh = med + k * mad
            mask = residuals <= thresh

        inlier_idx = np.nonzero(mask)[0]
        if len(inlier_idx) <= N_best:
            best_idx = np.argsort(residuals)[:N_best]
        else:
            inlier_res = residuals[inlier_idx]
            order = np.argsort(inlier_res)
            best_idx = inlier_idx[order[:N_best]]

        t_best = t[best_idx]
        x_best = x[best_idx]
        y_best = y[best_idx]
        z_best = z[best_idx]

        H_best = []
        for ti in t_best:
            H_i = np.array(
                [
                    [ti, 1, 0, 0, 0, 0, 0],
                    [0, 0, ti, 1, 0, 0, 0],
                    [0, 0, 0, 0, ti**2, ti, 1],
                ]
            )
            H_best.append(H_i)
        H_best = np.vstack(H_best)
        y_best_full = np.vstack([x_best, y_best, z_best]).T.reshape(-1)

        if use_recency_weights:
            w_best = weights[best_idx]
            w_best_rows = np.repeat(w_best, 3)
            w_best_sqrt = np.sqrt(w_best_rows)
            H_best_w = H_best * w_best_sqrt[:, None]
            y_best_w = y_best_full * w_best_sqrt
            theta_best = np.linalg.lstsq(H_best_w, y_best_w, rcond=None)[0]
        else:
            theta_best = np.linalg.lstsq(H_best, y_best_full, rcond=None)[0]

        self.theta = theta_best.copy()
        self.meas_prev = np.array([t_best[-1], x_best[-1], y_best[-1], z_best[-1]])

        inlier_res = residuals[best_idx]
        rms_inlier = np.sqrt(np.mean(inlier_res**2))

        if (len(best_idx) >= self.min_inliers_for_gate and
                rms_inlier < self.gate_residual_thresh and
                not np.isnan(self.theta).any()):
            self.v_gate_active = True

    def update(self, x, y, z, t):
        """
        Update the trajectory estimate with a new 3D measurement LS-MAD fitting.

        Parameters
        ----------
        x : float
            Measured x position.
        y : float
            Measured y position.
        z : float
            Measured z position.
        t : float
            Time stamp of the measurement.

        Returns
        -------
        ndarray
            Current trajectory parameter estimate, or NaNs if insufficient data
            is available.

        """
        pos = np.array([x, y, z])
        if (
            self.v_gate is not None
            and self.v_gate_active
            and self.meas_prev is not None
            and self.t_i is not None
        ):
            t_s = t - self.t_i
            dt = t_s - self.meas_prev[0]
            if dt <= 0:
                return self.theta
            v_mag = np.linalg.norm((pos - self.meas_prev[1:]) / dt)
            if v_mag > self.v_gate:
                return self.theta
        self.x_list.append(x)
        self.y_list.append(y)
        self.z_list.append(z)
        self.t_list.append(t)
        W = self.window_size
        if len(self.t_list) > W:
            self.x_list = self.x_list[-W:]
            self.y_list = self.y_list[-W:]
            self.z_list = self.z_list[-W:]
            self.t_list = self.t_list[-W:]
        if len(self.t_list) < self.N:
            return np.full(self.theta.shape, np.nan)
        elif len(self.t_list) == self.N:
            self.LS_MAD(
                self.t_list, self.x_list, self.y_list, self.z_list, N_best=self.N_best
            )
            return self.theta
        self.LS_MAD(
            self.t_list,
            self.x_list,
            self.y_list,
            self.z_list,
            N_best=5,
        )
        return self.theta

    def reset(self):
        """Reset function to reset field for next throw."""
        self.t_i = None
        self.meas_prev = None
        self.theta = self.theta_i.copy()
        self.x_list.clear()
        self.y_list.clear()
        self.z_list.clear()
        self.t_list.clear()
        self.v_gate_active = False

    def pos_at(self, t):
        """
        Evaluate the estimated 3D trajectory position at a given time.

        Parameters
        ----------
        t : float or array-like
            Time value(s) at which to evaluate the trajectory.

        Returns
        -------
        ndarray
            3D position(s) [x, y, z] corresponding to the given time value(s).

        """
        x = self.theta[0] * t + self.theta[1]
        y = self.theta[2] * t + self.theta[3]
        z = self.theta[4] * t**2 + self.theta[5] * t + self.theta[6]
        return np.stack((x, y, z), axis=-1)

    def inside_box(self, pts, eps=1e-9):
        """
        Check whether 3D point(s) lie within the bounding box.

        Parameters
        ----------
        pts : ndarray
            Array of 3D point(s) with last dimension [x, y, z].
        eps : float, default=1e-9
            Numerical tolerance applied to the bounding box limits.

        Returns
        -------
        ndarray of bool
            Boolean mask indicating which points lie inside the bounding box.

        """
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return (
            (self.bounds[0] - eps <= x)
            & (x <= self.bounds[1] + eps)
            & (self.bounds[2] - eps <= y)
            & (y <= self.bounds[3] + eps)
            & (self.bounds[4] - eps <= z)
            & (z <= self.bounds[5] + eps)
        )

    def find_t_linear(self, val, a, b):
        """
        Solve for time t from a linear relation a * t + b = val with nonnegative constraint.

        Parameters
        ----------
        val : ndarray
            Target value(s) of the linear function.
        a : float
            Linear coefficient multiplying time.
        b : float
            Constant offset.

        Returns
        -------
        ndarray
            Time value(s) t satisfying the equation, or NaN where undefined or negative.

        """
        if np.isclose(a, 0.0):
            return np.full(val.shape, np.nan)
        t = (val - b) / a
        t = np.where(t < 0, np.nan, t)
        return t

    def find_t_quad(self, val):
        """
        Solve for time t at which the quadratic z(t) equals a specified value.

        Parameters
        ----------
        val : ndarray
            Target value(s) of the quadratic function.

        Returns
        -------
        ndarray
            Nonnegative solution(s) for time t, or NaN where no valid solution exists.

        """
        a = self.theta[4]
        b = self.theta[5]
        c = self.theta[6] - val
        if np.isclose(a, 0.0):
            if np.isclose(b, 0.0):
                return np.full(val.shape, np.nan)
            t = -c / b
        else:
            t_1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            t_2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            t = np.concatenate((t_1, t_2))
        t = np.where(t < 0, np.nan, t)
        return t

    def calc_goal(self, eff_quat, eff_pos):
        """
        Compute an intercept goal position inside the bounding box and a goal quaternion\
        that minimally rotates the current end-effector x-axis toward the goal direction.

        Parameters
        ----------
        eff_quat : array-like
            Current end-effector orientation as a quaternion [x, y, z, w].
        eff_pos : array-like
            Current end-effector position as [x, y, z].

        Returns
        -------
        tuple[ndarray, ndarray]
            (goal_pos, goal_quat) where goal_pos is the computed 3D intercept position
            (or NaNs if no valid intercept exists) and goal_quat is the corresponding
            goal orientation quaternion.

        """
        if np.any(np.isnan(self.theta)):
            return np.array([np.nan, np.nan, np.nan]), eff_quat
        eff_R = tf.quaternion_matrix(eff_quat)[:3, :3]
        t_int_x = self.find_t_linear(self.bounds[:2], self.theta[0], self.theta[1])
        t_int_y = self.find_t_linear(self.bounds[2:4], self.theta[2], self.theta[3])
        t_int_z = self.find_t_quad(self.bounds[4:])
        t_cand = np.concatenate((t_int_x, t_int_y, t_int_z))
        t_cand = t_cand[np.isfinite(t_cand)]
        if t_cand.size == 0:
            return np.array([np.nan, np.nan, np.nan]), eff_quat
        pts = self.pos_at(t_cand)
        mask_inside = self.inside_box(pts)
        if not np.any(mask_inside):
            return np.array([np.nan, np.nan, np.nan]), eff_quat
        t_valid = t_cand[mask_inside]
        t_hit = np.min(t_valid)

        goal_pos = self.pos_at(t_hit)

        dir_vec = goal_pos - eff_pos
        dir_mag = np.linalg.norm(dir_vec)
        if dir_mag < 1e-9:
            return goal_pos, eff_quat

        x_new = dir_vec / dir_mag

        x_cur = eff_R[:, 0]

        w = np.cross(x_cur, x_new)
        w_norm = np.linalg.norm(w)
        R_rot = np.eye(3)
        if w_norm > 1e-9:
            th = angle_between(x_cur, x_new)
            w_brac = mr.VecToso3(w / w_norm * th)
            R_rot = mr.MatrixExp3(w_brac)
        else:
            dot = np.dot(x_cur, x_new)
            if dot < 0:
                tmp = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(tmp, x_cur)) > 0.9:
                    tmp = np.array([0.0, 0.0, 1.0])
                axis = np.cross(x_cur, tmp)
                axis /= np.linalg.norm(axis)
                w_brac = mr.VecToso3(axis * np.pi)
                R_rot = mr.MatrixExp3(w_brac)
        R_g = R_rot @ eff_R
        T_g = np.eye(4)
        T_g[:3, :3] = R_g
        return self.pos_at(t_hit), tf.quaternion_from_matrix(T_g)


if __name__ == '__main__':
    t = np.linspace(0, 1, 10)
    vx = -2
    vy = 0
    x_vals = vx * t + 0.4
    y_vals = vy * t
    z_vals = 1 - 4.9 * t**2
    noise_std = 0.01

    x_n = x_vals + np.random.randn(*x_vals.shape) * noise_std
    y_n = y_vals + np.random.randn(*y_vals.shape) * noise_std
    z_n = z_vals + np.random.randn(*z_vals.shape) * noise_std

    x_n[0] += 2
    x_n[6] += 5
    x_n[9] -= 3

    rls = LSMADParabola([-0.4, 0.4], [-0.4, 0.4], [-0.4, 0.4])

    for i in range(7):
        model = rls.update(x_n[i], y_n[i], z_n[i], t[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_pred = model[0] * t + model[1]
    y_pred = model[2] * t + model[3]
    z_pred = model[4] * (t**2) + model[5] * t + model[6]

    goal, quat = rls.calc_goal([0.0, 0.0, 0.0, 1.0], [0, 0, 0])
    print(quat)
    print(goal)

    ax.plot(x_vals, y_vals, z_vals, linewidth=2)
    ax.plot(x_pred, y_pred, z_pred, linewidth=2)
    ax.scatter(x_n, y_n, z_n)
    ax.scatter(goal[0], goal[1], goal[2])
    plt.axis('equal')
    plt.legend(['true', 'pred'])
    plt.show()
