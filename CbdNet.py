# coding=utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            single_conv(6, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

class cbdnet(nn.Module):
    def __init__(self):
        super(cbdnet, self).__init__()
        self.fcn = FCN()
        self.unet = UNet()
        self.conv1 = nn.Conv2d(1,3,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x = x.reshape([-1,3,10,10])
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        return out

    def loss_pde(self, x):
        x = x.reshape(-1, 3, 10, 10)
        y = self.forward(x)
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
        y_l, y_r, y_up, y_dw = self.forward(x_l), self.forward(x_r), self.forward(x_up), self.forward(x_dw)

        u_l = y_l[:, 0]
        u_r = y_r[:, 0]
        u_up = y_up[:, 0]
        u_dw = y_dw[:, 0]

        return ((u_l - u_r) ** 2).mean() +(u_up ** 2).mean() + (u_dw ** 2).mean()

    def loss_ic(self, x_i, u_i):
        u_i = u_i.reshape(-1,1,10,10)
        u_i = self.conv1(u_i)
        y_pred = self.forward(x_i)
        #u_i_pred = y_pred[:, 0]
        # print(f'u_loss: {((u_i_pred-u_i)**2).mean():6f}')
        # print(f'v_loss: {((v_i_pred-v_i)**2).mean():6f}')
        # print(f'phi_loss: {((phi_i_pred-phi_i)**2).mean():6f}')
        return ((y_pred - u_i) ** 2).mean()