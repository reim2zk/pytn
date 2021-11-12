import dataclasses
import tensorflow as tf

from pytn.mps import MPS
from pytn.mpo import MPO


@dataclasses.dataclass
class DMRG:
    mpo: MPO  # mpo.w_li
    mps: MPS  # A_li, lam, B_li
    disp: bool = False

    def check(self):
        nw = len(self.mpo.W_li)
        na = len(self.mps.A_li)
        nb = len(self.mps.B_li)
        if (nw != na + nb):
            print(f"invalid length. len(W)={nw}, len(A)={na}, len(B)={nb}")

    def init(self):
        self.mps.normalize()
        self.mps.move_left_edge()

    def step(self, direction: str):
        # build env matrix
        Y_xayb = self.env_matrix(direction)
        [x, a, y, b] = Y_xayb.shape[:4]
        n = x*a*y*b
        Y_mat = tf.reshape(Y_xayb, shape=(n, n))

        # diagonalize env matrix
        (lam_w, U_xaby_w) = tf.linalg.eigh(Y_mat)
        phi_xayb = tf.transpose(U_xaby_w)[0]
        phi_xayb = tf.reshape(phi_xayb, shape=[x, a, y, b])
        lam = lam_w[0].numpy()
        lam2 = tf.reduce_sum(Y_xayb * phi_xayb).numpy()

        # update MPS
        if direction == "left":
            self.mps.move_left(phi_xayb)
        else:
            self.mps.move_right(phi_xayb)

        # result
        return {
            "direction": direction,
            "min_eigval": lam,
            "Y.phi": lam2
        }

    def run(self, max_iter=10):
        self.init()
        li = []
        for it in range(max_iter):
            num = self.mps.num_B()-1
            for _ in range(num):
                el = self.step("right")
                el["iter"] = it
                li.append(el)
                if self.disp:
                    print(el)

            num = self.mps.num_A()-1
            for _ in range(num):
                el = self.step("left")
                el["iter"] = it
                li.append(el)
                if self.disp:
                    print(el)
        return li

    def expectation_value(self):
        P_z_h_x = self.left_tensor_P()
        Q_z_h_x = self.right_tensor_Q()
        lam_x = self.mps.lam
        return tf.einsum("zhx,x,z,zhx", P_z_h_x, lam_x, lam_x, Q_z_h_x)

    def env_matrix(self, direction: str):
        num_A = self.mps.num_A()
        if direction == "right":
            W_g_m_a_l = self.mpo.W_li[num_A]
            W_l_n_b_h = self.mpo.W_li[num_A+1]
            P_z_g_x = self.left_tensor_P(skip=0)
            Q_w_h_y = self.right_tensor_Q(skip=2)
        else:
            W_g_m_a_l = self.mpo.W_li[num_A-2]
            W_l_n_b_h = self.mpo.W_li[num_A-1]
            P_z_g_x = self.left_tensor_P(skip=2)
            Q_w_h_y = self.right_tensor_Q(skip=0)
        Y = tf.einsum("zgx,why,gmal,lnbh->xaybzmwn",
                      P_z_g_x, Q_w_h_y, W_g_m_a_l, W_l_n_b_h)
        return Y

    def left_tensor_P(self, skip=0):
        P_z_g_x = tf.constant(1.0, shape=(1, self.mpo.W_li[0].shape[0], 1))
        for idx in range(len(self.mps.A_li)-skip):
            A_x_a_y = self.mps.A_li[idx]
            A_z_m_w = A_x_a_y
            W_g_m_a_h = self.mpo.W_li[idx]
            P_w_h_y = tf.einsum("zgx,xay,zmw,gmah->why",
                                P_z_g_x, A_x_a_y, A_z_m_w, W_g_m_a_h)
            P_z_g_x = P_w_h_y
        return P_z_g_x

    def right_tensor_Q(self, skip=0):
        Q_z_g_x = tf.constant(1.0, shape=(1, self.mpo.W_li[-1].shape[-1], 1))
        for idx in range(len(self.mps.B_li)-skip):
            B_y_a_x = self.mps.B_li[len(self.mps.B_li)-idx-1]
            B_w_m_z = B_y_a_x
            W_h_m_a_g = self.mpo.W_li[len(self.mpo.W_li)-idx-1]
            Q_w_h_y = tf.einsum("zgx,yax,wmz,hmag->why",
                                Q_z_g_x, B_y_a_x, B_w_m_z, W_h_m_a_g)
            Q_z_g_x = Q_w_h_y
        return Q_z_g_x

    def environmental_vector(self):
        lam_x = self.mps.lam
        B_x_a_y = self.mps.B_li[0]
        B_y_b_z = self.mps.B_li[1]
        psi_x_a_z_b = tf.einsum("x,xay,ybz->xazb", lam_x, B_x_a_y, B_y_b_z)
        return tf.reshape(psi_x_a_z_b, shape=[-1])
