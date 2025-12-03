import matplotlib.pyplot as plt
import modern_robotics as mr
import numpy as np
import tf_transformations as tf


def angle_between(a, b):
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.arccos(dot)


class RLSParabola:

    def __init__(self, x_bounds, y_bounds, z_bounds, warm_start=True, N = 5, N_best = 3, lam=0.99, v_gate = 10):
        self.theta_i = np.array([0, 0, 0, 0, -4.9, 0, 0])
        self.P_i = np.eye(7) * 1e6
        self.P_i[4,4] = 1e2
        self.theta = self.theta_i.copy()
        self.P = self.P_i.copy()
        self.lam = lam
        self.t_i = None
        self.x_warm = np.zeros(N)
        self.y_warm = np.zeros(N)
        self.z_warm = np.zeros(N)
        self.t_warm = np.zeros(N)
        self.N = N
        self.N_best = N_best
        self.ws = warm_start
        self.counter = 0
        self.bounds = np.array(x_bounds + y_bounds + z_bounds)
        self.meas_prev = None
        self.v_gate = v_gate
    
    def warm_start(self, t, x, y, z, N_best = 3, k = 3):
        H_list = []
        t_0 = t[0]
        t = t - t_0
        N = len(t)
        for ti in t:
            H_i = np.array([
                [ti, 1, 0,  0,   0,     0,     0],
                [0,  0, ti, 1,   0,     0,     0],
                [0,  0, 0,  0, ti**2,  ti,     1],
            ])
            H_list.append(H_i)
        H = np.vstack(H_list)
        y_full = np.vstack([x, y, z]).T.reshape(-1)
        theta_ls = np.linalg.lstsq(H, y_full, rcond=None)[0]
        r_full = y_full - H @ theta_ls
        r_xyz = r_full.reshape(-1, 3)           
        residuals = np.linalg.norm(r_xyz, axis=1)  
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad == 0:
            mask = np.ones(N, dtype=bool)
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
            H_i = np.array([
                [ti, 1, 0, 0, 0, 0, 0],
                [0, 0, ti, 1, 0, 0, 0],
                [0, 0, 0, 0, ti**2, ti, 1],
            ])
            H_best.append(H_i)

        H_best = np.vstack(H_best)
        y_best_full = np.vstack([x_best, y_best, z_best]).T.reshape(-1)
        theta_best = np.linalg.lstsq(H_best, y_best_full, rcond=None)[0]
        self.theta = theta_best.copy()
        self.P = np.eye(7) * 100.0
        self.t_i = t_0
        self.meas_prev = np.array([t_best[-1], x_best[-1], y_best[-1], z_best[-1]])

    def update(self, x, y, z, t):
        if self.ws:
            if self.counter<self.N:
                self.x_warm[self.counter] = x
                self.y_warm[self.counter] = y
                self.z_warm[self.counter] = z
                self.t_warm[self.counter] = t
                self.counter += 1
                return np.full(self.theta.shape, np.nan)
            elif self.counter == self.N:
                self.warm_start(self.t_warm,
                                self.x_warm,
                                self.y_warm,
                                self.z_warm,
                                N_best=self.N_best)
                self.counter += 1
                return self.theta
        t_s = 0
        if self.t_i is None:
            self.t_i = t
        else:
            t_s = t - self.t_i
        pos = np.array([x, y, z])
        if self.v_gate is not None and self.meas_prev is not None:
            v_mag = np.linalg.norm((pos-self.meas_prev[1:])/(t_s - self.meas_prev[0]))
            if v_mag > self.v_gate:
                return self.theta
        H = np.array([
            [t_s, 1.0, 0.0, 0.0, 0.0,      0.0,    0.0],
            [0.0, 0.0, t_s, 1.0, 0.0,      0.0,    0.0],
            [0.0, 0.0, 0.0, 0.0, t_s**2,   t_s,    1.0],
        ])
        S = self.lam * np.eye(3) + H @ self.P @ H.T
        K = self.P @ H.T @ np.linalg.inv(S)
        r = pos - H @ self.theta
        self.theta = self.theta + K @ r
        self.P = (self.P - K @ H @ self.P) / self.lam
        self.meas_prev = np.array([t_s,x,y,z])
        return self.theta

    def reset(self):
        self.theta = self.theta_i.copy()
        self.P = self.P_i.copy()
        self.counter = 0
        self.t_i = None
        self.meas_prev = None

    def pos_at(self, t):
        x = self.theta[0] * t + self.theta[1]
        y = self.theta[2] * t + self.theta[3]
        z = self.theta[4] * t**2 + self.theta[5] * t + self.theta[6]
        return np.stack((x, y, z), axis=-1)

    def inside_box(self, pts, eps=1e-9):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        return ((self.bounds[0] - eps <= x) & (x <= self.bounds[1] + eps) &
                (self.bounds[2] - eps <= y) & (y <= self.bounds[3] + eps) &
                (self.bounds[4] - eps <= z) & (z <= self.bounds[5] + eps))

    def find_t_linear(self, val, a, b):
        if np.isclose(a, 0.0):
            return np.full(val.shape, np.nan)
        t = (val - b) / a
        t = np.where(t < 0, np.nan, t)
        return t

    def find_t_quad(self, val):
        a = self.theta[4]
        b = self.theta[5]
        c = self.theta[6] - val
        if np.isclose(a, 0.0):
            if np.isclose(b, 0.0):
                return np.full(val.shape, np.nan)
            t = -c / b
        else:
            t_1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            t_2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            t = np.concatenate((t_1, t_2))
        t = np.where(t < 0, np.nan, t)
        return t

    def calc_goal(self, eff_quat):
        """
        Calc goal.

        returns pose and quat (np array) orientation of basket to catch ball,
        eff_quat is the current orientation of the basket
        """
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
        vx = self.theta[0]
        vy = self.theta[2]
        vz = 2*self.theta[4]*t_hit + self.theta[5]
        vel_vec = np.array([vx, vy, vz])
        vel_mag = np.linalg.norm(vel_vec)
        if (vel_mag < 1e-9):
            return self.pos_at(t_hit),eff_quat
        z_new = -vel_vec/vel_mag
        z_cur = eff_R[:,2]
        w = np.cross(z_cur, z_new)
        w_norm = np.linalg.norm(w)
        R_rot = np.eye(3)
        if w_norm > 1e-9:
            th = angle_between(z_cur, z_new)
            w_brac = mr.VecToso3(w/w_norm*th)
            R_rot = mr.MatrixExp3(w_brac)
        else:
            if np.dot(z_cur,z_new) < 0:
                tmp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(tmp, z_cur)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                axis = np.cross(z_cur, tmp)
                axis /= np.linalg.norm(axis)
                w_brac = mr.VecToso3(axis * np.pi)
                R_rot = mr.MatrixExp3(w_brac)
        R_g = R_rot @ eff_R
        T_g = np.eye(4)
        T_g[:3, :3] = R_g
        return self.pos_at(t_hit),tf.quaternion_from_matrix(T_g)


    
if __name__ == '__main__':
    t = np.linspace(0,1,10)
    vx = -2
    vy = 0
    x_vals = vx*t + 0.4
    y_vals = vy*t 
    z_vals = 1 - 4.9*t**2
    noise_std = 0.01

    x_n = x_vals + np.random.randn(*x_vals.shape) * noise_std
    y_n = y_vals + np.random.randn(*y_vals.shape) * noise_std
    z_n = z_vals + np.random.randn(*z_vals.shape) * noise_std
    
    x_n[0] += 2
    x_n[6] += 5
    x_n[9] -= 3

    rls = RLSParabola([-0.4,0.4],[-0.4,0.4],[-0.4,0.4])

    for i in range(7):
        model = rls.update(x_n[i],y_n[i],z_n[i],t[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_pred = model[0]*t + model[1]
    y_pred = model[2]*t + model[3]
    z_pred = model[4]*(t**2)+ model[5]*t + model[6]

    goal,quat = rls.calc_goal([0.0, 0.0, 0.0, 1.0])
    print(quat)
    print(goal)

    ax.plot(x_vals, y_vals, z_vals, linewidth=2)
    ax.plot(x_pred, y_pred, z_pred, linewidth=2)
    ax.scatter(x_n,y_n,z_n)
    ax.scatter(goal[0],goal[1],goal[2])
    plt.axis('equal')
    plt.legend(['true','pred'])
    plt.show()
