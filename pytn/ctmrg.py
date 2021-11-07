import tensorflow as tf
import numpy as np

import pytn.utils


class CTMRG():
    N: int
    W_abcd: tf.Tensor
    chi: int

    def __init__(self, N: int, J_kT: float, chi: int,
                 boundary_cond="fixed", rescale=False, disp=False):
        self.N = N
        self.W_abcd = pytn.utils.ising_tensor_W(J_kT, N)
        self.chi = chi
        self.s_sigma_ab = [[(2*a-1)*(2*b-1) for a in [0, 1]] for b in [0, 1]]
        self.s_sigma_ab = tf.constant(self.s_sigma_ab, dtype=tf.float32)
        self.boundary_cond = boundary_cond
        self.rescale = rescale
        self.disp = disp

    def eff_num(self, nw: int):
        if self.chi is None:
            return nw
        if nw > self.chi:
            return self.chi
        else:
            return nw

    def measure(self, L, C_xy, P_xdy):
        s_a = tf.constant([2*a-1 for a in [0, 1]], dtype=tf.float32)

        # Z
        Z = tf.einsum("xy,yz,zw,wx->", C_xy, C_xy, C_xy, C_xy).numpy()
        # spin_corr
        CP_xay = tf.einsum("xy,ydz->xdz", C_xy, P_xdy)
        G_abcd = tf.einsum("xay,ybz,zcw,wdx->abcd",
                           CP_xay, CP_xay, CP_xay, CP_xay)
        OL = tf.einsum("abcd,abcd->", G_abcd, self.W_abcd)
        s_sigma = tf.einsum("abcd,abcd,ab",
                            G_abcd, self.W_abcd, self.s_sigma_ab)
        spin_corr = (s_sigma / OL).numpy()
        # spin
        phi_xy = tf.einsum("xy,yz->xz", C_xy, C_xy)
        QL = tf.einsum("xy,zw,xaz,yaw->", phi_xy, phi_xy, P_xdy, P_xdy)
        s = tf.einsum("xy,zw,xaz,yaw,a->",
                      phi_xy, phi_xy, P_xdy, P_xdy, s_a)
        spin = (s / QL).numpy()
        # output
        if self.disp:
            print(f"L={L}, Z={Z}, spin={spin}, spin_corr={spin_corr}")
        return {"L": L, "Z": Z, "spin": spin, "spin_corr": spin_corr}

    def run_directory(self):
        if self.boundary_cond == "free":
            P1_abc = tf.reduce_sum(self.W_abcd, axis=0)
            C1_ab = tf.reduce_sum(self.W_abcd, axis=[0, 1])
        elif self.boundary_cond == "fixed":
            P1_abc = self.W_abcd[1, :, :, :]
            C1_ab = self.W_abcd[1, 1, :, :]
        else:
            msg = f"invalid argument. boudanry_cond={self.boundary_cond}"
            raise Exception(msg)

        li = []
        P_xdy = P1_abc
        C_xy = C1_ab
        for L in range(1, self.N//2+1):
            # 1. measure
            res_measurement = self.measure(L, C_xy, P_xdy)
            li.append(res_measurement)

            # 2. extention
            P_x_a_b_y_d = tf.einsum("xcy,abcd->xabyd",
                                    P_xdy, self.W_abcd)
            C_x_b_y_d = tf.einsum("xy,zax,ycw,abcd->zbwd",
                                  C_xy, P_xdy, P_xdy, self.W_abcd)

            # 3. update
            (x, a, b, y, d) = P_x_a_b_y_d.shape
            P_xdy = tf.reshape(P_x_a_b_y_d, shape=(x*a, b, y*d))
            (x, b, y, d) = C_x_b_y_d.shape
            C_xy = tf.reshape(C_x_b_y_d, shape=(x*b, y*d))
            if self.rescale:
                scaling_coef = np.max(
                    [P_xdy.numpy().max(), C_xy.numpy().max()])
                P_xdy = P_xdy / scaling_coef
                C_xy = C_xy / scaling_coef
        return li

    def run(self):
        if self.boundary_cond == "free":
            P1_abc = tf.reduce_sum(self.W_abcd, axis=0)
            C1_ab = tf.reduce_sum(self.W_abcd, axis=[0, 1])
        elif self.boundary_cond == "fixed":
            P1_abc = self.W_abcd[1, :, :, :]
            C1_ab = self.W_abcd[1, 1, :, :]
        else:
            msg = f"invalid argument. boudanry_cond={self.boundary_cond}"
            raise Exception(msg)

        li = []
        P_xdy = P1_abc
        C_xy = C1_ab
        for L in range(1, self.N//2+1):
            # measumement
            res_measurement = self.measure(L, C_xy, P_xdy)
            li.append(res_measurement)

            # extention
            P_x_a_b_y_d = tf.einsum("xcy,abcd->xabyd",
                                    P_xdy, self.W_abcd)
            C_x_b_y_d = tf.einsum("xy,zax,ycw,abcd->zbwd",
                                  C_xy, P_xdy, P_xdy, self.W_abcd)

            # diagonalize
            (x, a, y, b) = C_x_b_y_d.shape
            C = tf.reshape(C_x_b_y_d, shape=(x*a, y*b))
            print("C", C.shape)
            (lam_i, U_xa_i) = tf.linalg.eigh(C)
            num_lam = lam_i.shape[0]
            nw = self.eff_num(num_lam)
            lam_i = lam_i[num_lam-nw:]
            U_xa_i = U_xa_i[:, num_lam-nw:]

            # compression
            # print([t.shape for t in [lam_w, U_xa_w, U_xa_w]])
            C_i_j = tf.linalg.diag(lam_i)
            (x, a, b, y, c) = P_x_a_b_y_d.shape
            P_xa_d_yb = tf.reshape(P_x_a_b_y_d, shape=(x*a, b, y*c))
            P_i_d_j = tf.einsum("xi,yj,xdy->idj", U_xa_i, U_xa_i, P_xa_d_yb)

            # update
            P_xdy = P_i_d_j
            C_xy = C_i_j
            if self.rescale:
                scaling_coef = np.max(
                    [P_xdy.numpy().max(), C_xy.numpy().max()])
                P_xdy = P_xdy / scaling_coef
                C_xy = C_xy / scaling_coef
        return li
