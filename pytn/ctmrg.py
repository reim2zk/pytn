import tensorflow as tf
import numpy as np

import pytn.utils


class CTMRG():
    N: int
    W_abcd: tf.Tensor
    chi: int

    def __init__(self, N: int, J_kT: float, chi: int, 
                 boundary_cond="fixed", rescale=False):
        self.N = N
        self.W_abcd = pytn.utils.ising_tensor_W(J_kT, N)
        self.chi = chi
        self.s_sigma_ab = [[(2*a-1)*(2*b-1) for a in [0, 1]] for b in [0, 1]]
        self.s_sigma_ab = tf.constant(self.s_sigma_ab, dtype=tf.float32)
        self.boundary_cond = boundary_cond
        self.rescale = rescale

    def eff_num(self, nw: int):
        if self.chi is None:
            return nw
        if nw > self.chi:
            return nw
        else:
            return self.chi

    def run_directory(self):
        s_a = tf.constant([2*a-1 for a in [0, 1]], dtype=tf.float32)

        # initialize (free boundary condition is applied)
        if self.boundary_cond == "free":
            P1_abc = tf.reduce_sum(self.W_abcd, axis=0)
            C1_ab = tf.reduce_sum(self.W_abcd, axis=[0, 1])
        elif self.boundary_cond == "fixed":
            P1_abc = self.W_abcd[1, :, :, :]
            C1_ab = self.W_abcd[1, 1, :, :]
        else:
            msg = f"invalid argument. boudanry_cond={self.boundary_cond}"
            raise Exception(msg)

        P_xdy = P1_abc
        C_xy = C1_ab
        for L in range(1, self.N//2+1):
            # 1. measumement
            Z = tf.einsum("xy,yz,zw,wx->", C_xy, C_xy, C_xy, C_xy)
            # 1.1 spin_corr
            CP_xay = tf.einsum("xy,ydz->xdz", C_xy, P_xdy)
            G_abcd = tf.einsum("xay,ybz,zcw,wdx->abcd",
                               CP_xay, CP_xay, CP_xay, CP_xay)
            OL = tf.einsum("abcd,abcd->", G_abcd, self.W_abcd)
            s_sigma = tf.einsum("abcd,abcd,ab",
                                G_abcd, self.W_abcd, self.s_sigma_ab)
            spin_corr = s_sigma / OL
            # 1.2 spin
            phi_xy = tf.einsum("xy,yz->xz", C_xy, C_xy)
            QL = tf.einsum("xy,zw,xaz,yaw->", phi_xy, phi_xy, P_xdy, P_xdy)
            s = tf.einsum("xy,zw,xaz,yaw,a->",
                          phi_xy, phi_xy, P_xdy, P_xdy, s_a)
            spin = s / QL
            # 1.3 output
            print(f"L={L}, Z={Z}, spin={spin}, spin_corr={spin_corr}")

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

    def run(self):
        # initialize (free boundary condition is applied)
        P1_abc = tf.reduce_sum(self.W_abcd, axis=0)
        C1_ab = tf.reduce_sum(self.W_abcd, axis=[0, 1])

        P_xdy = P1_abc
        C_xy = C1_ab
        for L in range(1, self.N//2+1):
            # measumement
            Z = tf.einsum("xy,yz,zw,wx", C_xy, C_xy, C_xy, C_xy)
            print(f"L={L}, Z={Z}")

            # extention
            P_x_a_b_y_c = tf.einsum("xdy,abcd->xabyc",
                                    P_xdy, self.W_abcd)
            C_x_a_y_b = tf.einsum("xy,zax,ydw,abcd->zbwc",
                                  C_xy, P_xdy, P_xdy, self.W_abcd)

            # diagonalize
            (x, a, y, b) = C_x_a_y_b.shape
            C = tf.reshape(C_x_a_y_b, shape=(x*a, y*b))
            (lam_w, U_xa_w) = tf.linalg.eigh(C)
            nw = self.eff_num(lam_w.shape[0])
            lam_w = lam_w[:nw]
            U_xa_w = U_xa_w[:, :nw]

            # compression
            C_w_z = tf.einsum("w,xw,yw->xy", lam_w, U_xa_w, U_xa_w)
            (x, a, b, y, c) = P_x_a_b_y_c.shape
            P_xa_d_yb = tf.reshape(P_x_a_b_y_c, shape=(x*a, b, y*c))
            P_w_d_z = tf.einsum("xw,yz,xdy->wdz", U_xa_w, U_xa_w, P_xa_d_yb)

            # update
            P_xdy = P_w_d_z
            C_xy = C_w_z
