# coding=utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from plotting import newfig, savefig
parser = argparse.ArgumentParser(description='Semantic aware super-resolution')

parser.add_argument('--dataDir', default='./data', help='dataset directory')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
parser.add_argument('--load', default= 'model_name', help='save result')

parser.add_argument('--model_name', default= 'RDN', help='model to select')
parser.add_argument('--finetuning', default=False, help='finetuning the training')
parser.add_argument('--need_patch', default=False, help='get patch form image')

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=96,  help='patch size')

parser.add_argument('--nThreads', type=int, default=3, help='number of threads for data loading')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')

parser.add_argument('--scale', type=int, default= 1, help='scale output size /input size')


args = parser.parse_args()


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat * scale * scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 3, 2, 1)
        F_ = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        output = self.conv3(us)

        return output


    def loss_pde(self, x):
        epsilon = 0.5
        gamma = 0.1
        y = self.forward(x)
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
        y_l, y_r, y_up, y_dw = self.forward(x_l), self.forward(x_r), self.forward(x_up), self.forward(x_dw)

        u_l = y_l[:, 0]
        u_r = y_r[:, 0]
        u_up = y_up[:, 0]
        u_dw = y_dw[:, 0]

        return ((u_l - u_r) ** 2).mean() +(u_up ** 2).mean() + (u_dw ** 2).mean()

    def loss_ic(self, x_i, u_i):
        y_pred = self.forward(x_i)
        u_i_pred = y_pred[:, 0]
        # print(f'u_loss: {((u_i_pred-u_i)**2).mean():6f}')
        # print(f'v_loss: {((v_i_pred-v_i)**2).mean():6f}')
        # print(f'phi_loss: {((phi_i_pred-phi_i)**2).mean():6f}')
        return ((u_i_pred - u_i) ** 2).mean()



if __name__ == '__main__':
    # torch.Size([500, 3])
    # torch.Size([500, 3, 1, 1])
    input = torch.randn([500,3])
    print(input.shape)
    model = RDN(args)
    out = model(input)
    print(out.shape)
