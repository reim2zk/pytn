from typing import Union
import dataclasses
import numpy as np
import tensorflow as tf


def svd_with_reduction(psi_i_j: tf.Tensor, chi: Union[int, None]):
    (lam_w, U_i_w, V_j_w) = tf.linalg.svd(psi_i_j)
    nw = lam_w.shape[0]
    if chi is None:
        idx = 0
    elif nw > chi:
        idx = nw-chi
    else:
        idx = 0
    lam_w = lam_w[idx:]
    U_i_w = U_i_w[:, idx:]
    V_j_w = V_j_w[:, idx:]
    return (lam_w, U_i_w, V_j_w)


def canonical_mps(psi: tf.Tensor, chi: Union[int, None] = None):
    dim_a_li = psi.shape
    dim_a_b_li = zip(dim_a_li[:-1], dim_a_li[1:])

    psi_xa_bcd = tf.reshape(psi, (dim_a_li[0], -1))
    dim_x = 1
    A_x_a_y_li = []
    for (dim_a, dim_b) in dim_a_b_li:
        # (lam_y, U_xa_y, V_bcd_y) = tf.linalg.svd(psi_xa_bcd)
        (lam_y, U_xa_y, V_bcd_y) = svd_with_reduction(psi_xa_bcd, chi)
        dim_y = lam_y.shape[0]
        A_x_a_y = tf.reshape(U_xa_y, (dim_x, dim_a, dim_y))
        A_x_a_y_li.append(A_x_a_y)

        lamV_b_cd_y = tf.reshape(
            lam_y[tf.newaxis, :] * V_bcd_y, (dim_b, -1, dim_y))
        lamV_y_b_cd = tf.transpose(lamV_b_cd_y, perm=[2, 0, 1])
        lamV_yb_cd = tf.reshape(lamV_y_b_cd, (dim_y*dim_b, -1))
        psi_xa_bcd = lamV_yb_cd
        dim_x = dim_y
    B_x_a_y_li = [tf.reshape(tf.transpose(V_bcd_y), (dim_y, dim_a, 1))]
    return MPS(A_x_a_y_li, lam_y, B_x_a_y_li)


def up_spin_mps(num, chi: Union[int, None] = None):
    xs = np.zeros(shape=[2**num], dtype=np.float32)
    xs[-1] = 1
    t = tf.reshape(xs, [2]*num)
    return canonical_mps(t, chi)


@dataclasses.dataclass
class MPS:
    A_li: tf.Tensor  # list of tensors A_x^a_y
    lam: tf.Tensor   # eigenvalues lambda_x
    B_li: tf.Tensor  # list of tensors B_x^a_y

    def check(self):
        eps = 0.0001
        success = True
        for A_x_a_y in self.A_li:
            n_y = A_x_a_y.shape[-1]
            d_yz = tf.einsum("xay,xaz->yz", A_x_a_y, A_x_a_y)
            expect = tf.eye(n_y, n_y)
            diff_max = tf.abs(d_yz-expect).numpy().max()
            if diff_max > eps:
                success = False
                print("A is not orthogonal")
        for B_x_a_y in self.B_li:
            n_x = B_x_a_y.shape[0]
            d_yz = tf.einsum("xaz,yaz->xy", B_x_a_y, B_x_a_y)
            expect = tf.eye(n_x, n_x)
            diff_max = tf.abs(d_yz-expect).numpy().max()
            if diff_max > eps:
                success = False
                print("B is not orthogonal")
        return success

    def summary(self):
        return "A" * self.num_A() + "B" * self.num_B()

    def norm2(self):
        return tf.reduce_sum(self.lam * self.lam).numpy()

    def num_A(self):
        return len(self.A_li)

    def num_B(self):
        return len(self.B_li)

    def full_contract(self):
        na_li = [t.shape[1] for t in self.A_li + self.B_li]
        P_abc_y = None
        for A_y_d_z in self.A_li:
            if P_abc_y is None:
                P_abc_y = tf.einsum("ydz->dz", A_y_d_z)
            else:
                (_, _, nz) = A_y_d_z.shape
                P_adz = tf.einsum("ay,ydz->adz", P_abc_y, A_y_d_z)
                P_abc_y = tf.reshape(P_adz, shape=(-1, nz))
        Q_x_abc = None
        for B_y_d_x in reversed(self.B_li):
            if Q_x_abc is None:
                Q_x_abc = tf.einsum("ydx->yd", B_y_d_x)
            else:
                (nx, _, _) = B_y_d_x.shape
                Q_y_d_abc = tf.einsum("ydx,xa->yda", B_y_d_x, Q_x_abc)
                Q_x_abc = tf.reshape(Q_y_d_abc, shape=(nx, -1))
        lam_x = self.lam
        t = tf.einsum('ax,x,xb->ab', P_abc_y, lam_x, Q_x_abc)
        return tf.reshape(t, na_li)

    def normalize(self):
        n = np.sqrt(self.norm2())
        self.lam = self.lam / n

    def move_left(self, phi_xazb=None):
        if self.num_A() <= 1:
            raise Exception(f"number of A is not enough. A={self.num_A()}")
        if phi_xazb is None:
            A_xay = self.A_li[-2]
            A_ybz = self.A_li[-1]
            lam_z = self.lam
            phi_xazb = tf.einsum('xay,ybz,z->xazb', A_xay, A_ybz, lam_z)
        (nx, na, nz, nb) = phi_xazb.shape
        phi_xa_zb = tf.reshape(phi_xazb, (nx*na, nb*nz))
        (lam_w, U_xa_w, V_zb_w) = tf.linalg.svd(phi_xa_zb)
        nw = lam_w.shape[0]
        A_xaw = tf.reshape(U_xa_w, (nx, na, nw))
        B_wbz = tf.transpose(tf.reshape(V_zb_w, (nz, nb, nw)), perm=[2, 1, 0])

        self.A_li = self.A_li[:-2] + [A_xaw]
        self.lam = lam_w
        self.B_li = [B_wbz] + self.B_li

    def move_right(self, phi_xazb=None):
        if self.num_B() <= 1:
            raise Exception(f"number of B is not enough. A={self.num_B()}")
        if phi_xazb is None:
            B_xay = self.B_li[0]
            B_ybz = self.B_li[1]
            lam_x = self.lam
            phi_xazb = tf.einsum('x,xay,ybz->xazb', lam_x, B_xay, B_ybz)
        (nx, na, nz, nb) = phi_xazb.shape
        phi_xa_zb = tf.reshape(phi_xazb, (na*nx, nb*nz))
        (lam_w, U_xa_w, V_zb_w) = tf.linalg.svd(phi_xa_zb)
        nw = lam_w.shape[0]
        A_xaw = tf.reshape(U_xa_w, (nx, na, nw))
        B_wbz = tf.transpose(tf.reshape(V_zb_w, (nz, nb, nw)), perm=[2, 1, 0])

        self.A_li = self.A_li + [A_xaw]
        self.lam = lam_w
        self.B_li = [B_wbz] + self.B_li[2:]

    def move_left_edge(self):
        for _ in range(self.num_A()-1):
            self.move_left()

    def move_right_edge(self):
        for _ in range(self.num_B()-1):
            self.move_right()

    def reduce(self, chi: int):
        for idx in range(len(self.A_li)):
            A_x_a_y = self.A_li[idx]
            (nx0, _, ny0) = A_x_a_y.shape
            nx = np.min([chi, nx0])
            ny = np.min([chi, ny0])
            self.A_li[idx] = A_x_a_y[nx0-nx:, :, ny0-ny:]
        for idx in range(len(self.B_li)):
            B_x_a_y = self.B_li[idx]
            (nx0, _, ny0) = self.B_li[idx].shape
            nx = np.min([chi, nx0])
            ny = np.min([chi, ny0])
            self.B_li[idx] = B_x_a_y[nx0-nx:, :, ny0-ny:]
        (nx0) = self.lam.shape[0]
        nx = np.min([nx0, chi])
        self.lam = self.lam[nx0-nx:]
