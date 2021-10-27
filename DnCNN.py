# coding=utf-8

import torch
import torch.nn as nn
import numpy as np


torch.manual_seed(123456)
np.random.seed(123456)
alph = 0.5
eta = 0.1
lmd = 0.01
gemma = 0.01
rho = 1
epsilon = 0.5
gamma = 1
H_o = 0.1
k = 6 * np.pi



def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)
class PRE(nn.Module):
    def __init__(self):
        super(PRE, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        return x


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=0):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(PRE())
        layers.append(nn.Conv1d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_channels=features, out_channels=4, kernel_size=kernel_size, padding=padding, bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out

    def loss_pde(self, x):
        y = self.net(x)
        # 输入 torch.Size([500, 3])
        # resize torch.Size([500, 3, 1])， 得到torch.Size([500, 4, 1])
        u = y[:, 0]
        u_g = gradients(u, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        u_xx, u_xy, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_x, x)[0][:, 2], gradients(u_y, x)[0][:, 2]

        kappa = (u_xx * (1 + u_y ** 2) + u_yy * (1 + u_x ** 2) - 2 * u_x * u_y * u_xy) / (1 + u_x ** 2 + u_y ** 2) ** (
                3 / 2)

        kappa_g = gradients(kappa, x)[0]
        kappa_t, kappa_x, kappa_y = kappa_g[:, 0], kappa_g[:, 1], kappa_g[:, 2]
        kappa_xx, kappa_xy, kappa_yy = gradients(kappa_x, x)[0][:, 1], \
                                       gradients(kappa_x, x)[0][:, 2], gradients(kappa_y, x)[0][:, 2]
        kappa_Laplacian = (kappa_xx * (1 + kappa_y ** 2) + kappa_yy * (
                1 + kappa_x ** 2) - 2 * kappa_x * kappa_y * kappa_xy) / (1 + kappa_x ** 2 + kappa_y ** 2) ** (3 / 2)
        delt_u = 1 / np.pi * epsilon / (epsilon ** 2 + u ** 2)
        W = 0.5 * u ** 2 * (1 - u) ** 2
        M = W + gamma * epsilon ** 2
        N = torch.sqrt(M)
        W_g = gradients(W, u)[0]
        mu = 1 / epsilon ** 2 * W_g - kappa_Laplacian * (u_x ** 2 + u_y ** 2 + 1) * delt_u

        Q = N * mu
        Q_g = gradients(Q, x)[0]
        Q_t, Q_x, Q_y = Q_g[:, 0], Q_g[:, 1], Q_g[:, 2]
        Q_xx, Q_xy, Q_yy = gradients(Q_x, x)[0][:, 1], \
                           gradients(Q_x, x)[0][:, 2], gradients(Q_y, x)[0][:, 2]

        P1 = (M * (1 - 2 * u_x * u_y) * Q_x) / (u_x ** 2 + u_y ** 2 + 1)
        P2 = (M * (1 - 2 * u_x * u_y) * Q_y) / (u_x ** 2 + u_y ** 2 + 1)

        P1_g = gradients(P1, x)[0]
        P1_t, P1_x, P1_y = P1_g[:, 0], P1_g[:, 1], P1_g[:, 2]
        P2_g = gradients(P2, x)[0]
        P2_t, P2_x, P2_y = P2_g[:, 0], P2_g[:, 1], P2_g[:, 2]

        div_surface = P1_x + P2_y - \
                      (u_x ** 2 * P1_x + u_x * u_y * (P1_x + P2_y) + u_y ** 2 * P2_y) / (u_x ** 2 + u_y ** 2 + 1)

        loss_1 = u_t - N * div_surface * (u_x ** 2 + u_y ** 2 + 1) * delt_u
        loss_2 = mu - 1 / epsilon * W_g + kappa_Laplacian * (u_x ** 2 + u_y ** 2 + 1)
        loss_3 = kappa - (u_xx * (1 + u_y ** 2) + u_yy * (1 + u_x ** 2) - 2 * u_x * u_y * u_xy) / (
                1 + u_x ** 2 + u_y ** 2) ** (3 / 2)
        loss_4 = kappa_Laplacian - (kappa_xx * (1 + kappa_y ** 2) + kappa_yy * (
                1 + kappa_x ** 2) - 2 * kappa_x * kappa_y * kappa_xy) / (1 + kappa_x ** 2 + kappa_y ** 2) ** (3 / 2)
        loss_5 = W - 0.5 * u ** 2 * (1 - u) ** 2
        loss_6 = M - W - gamma * epsilon ** 2
        loss_7 = N - torch.sqrt(M)
        loss_8 = Q - N * mu
        loss_9 = P1 - (M * (1 - 2 * u_x * u_y) * Q_x) / (u_x ** 2 + u_y ** 2 + 1)
        loss_10 = P2 - (M * (1 - 2 * u_x * u_y) * Q_y) / (u_x ** 2 + u_y ** 2 + 1)
        loss_11 = div_surface - (P1_x + P2_y - \
                                 (u_x ** 2 * P1_x + u_x * u_y * (P1_x + P2_y) + u_y ** 2 * P2_y) / (
                                         u_x ** 2 + u_y ** 2 + 1))

        loss = (loss_1 ** 2).mean() + (loss_2 ** 2).mean() + \
               (loss_3 ** 2).mean() + (loss_4 ** 2).mean() + \
               (loss_5 ** 2).mean() + (loss_6 ** 2).mean() + \
               (loss_7 ** 2).mean() + (loss_8 ** 2).mean() + \
               (loss_9 ** 2).mean() + (loss_10 ** 2).mean() + \
               (loss_11 ** 2).mean()
        return loss

    def loss_bc(self, x_l, x_r, x_up, x_dw):
        y_l, y_r, y_up, y_dw = self.net(x_l), self.net(x_r), self.net(x_up), self.net(x_dw)

        u_l = y_l[:, 0]
        u_r = y_r[:, 0]
        u_up = y_up[:, 0]
        u_dw = y_dw[:, 0]

        return ((u_l - u_r) ** 2).mean() +(u_up ** 2).mean() + (u_dw ** 2).mean()

    def loss_ic(self, x_i, u_i):
        y_pred = self.net(x_i)
        u_i_pred = y_pred[:, 0]
        # print(f'u_loss: {((u_i_pred-u_i)**2).mean():6f}')
        # print(f'v_loss: {((v_i_pred-v_i)**2).mean():6f}')
        # print(f'phi_loss: {((phi_i_pred-phi_i)**2).mean():6f}')
        return ((u_i_pred - u_i) ** 2).mean()

if __name__ == '__main__':
    # torch.Size([500, 3])
    # torch.Size([500, 4, 1])
    input = torch.rand([500,3])
    print(input.shape)
    model = DnCNN(channels=3)
    out = model(input)
    print(out.shape)