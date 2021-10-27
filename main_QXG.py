# coding=utf-8

"""
phi(s)= paplacian**3(kappa)
"""
import sys

from tqdm import tqdm

from models.CbdNet import *
from models.RDN import *
from models.VDN import *
from models.DnCNN import *
from models.FNN import *
from utils import AC_2D_init, newfig, to_numpy

sys.path.insert(0, '../../Utils')

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

loss_history = {
    "loss_pde": [],
    "loss_ic": [],
    "loss_bc": [],
    "train_loss": []
}

def main( model ):
    # TODO 超参数 可以慢慢设置
    epochs = 1000
    lr = 0.001
    num_x = 100
    num_y = 100
    num_t = 100
    num_b_train = 50 # boundary sampling points
    num_f_train = 500 # inner sampling points
    num_i_train = 50  # initial sampling points
    num_t_train = 4

    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    x_2d = np.concatenate((x_grid.flatten()[:, None], y_grid.flatten()[:, None]), axis=1)

    ## initialization
    xt_init = np.concatenate((np.zeros((num_x * num_y, 1)), x_2d), axis=1)
    u_init = AC_2D_init(xt_init)

    ## save init fig
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off')
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_init.reshape(num_x, num_y), interpolation='nearest', cmap='rainbow',
                  # extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(xt_init[:, 1], xt_init[:, 2], 'kx', label='Data (%d points)' % (xt_init.shape[0]), markersize=4,
            clip_on=False)
    line = np.linspace(xt_init.min(), xt_init.max(), 2)[:, None]
    fig.savefig('Figures-1/u_init.png')

    x_2d_ext = np.tile(x_2d, [num_t, 1])
    t_ext = np.zeros((num_t * num_x * num_y, 1))
    for i in range(0, num_t):
        t_ext[i * num_x * num_y:(i + 1) * num_x * num_y, :] = t[i]
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)

    ## sampling
    id_f = np.random.choice(num_x * num_y * num_t, num_f_train, replace=False)
    id_b = np.random.choice(num_x, num_b_train, replace=False)  ## Dirichlet
    id_i = np.random.choice(num_x * num_y, num_i_train, replace=False)
    id_t = np.random.choice(num_t, num_t_train, replace=False)

    ## boundary
    t_b = t[id_t, :]
    t_b_ext = np.zeros((num_t_train * num_b_train, 1))
    for i in range(0, num_t_train):
        t_b_ext[i * num_b_train:(i + 1) * num_b_train, :] = t_b[i]
    x_up = np.vstack((x_grid[-1, :], y_grid[-1, :])).T
    x_dw = np.vstack((x_grid[0, :], y_grid[0, :])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T
    # x_bound = np.vstack((x_up, x_dw, x_l, x_r))

    x_up_sample = x_up[id_b, :]
    x_dw_sample = x_dw[id_b, :]
    x_l_sample = x_l[id_b, :]
    x_r_sample = x_r[id_b, :]

    x_up_ext = np.tile(x_up_sample, (num_t_train, 1))
    x_dw_ext = np.tile(x_dw_sample, (num_t_train, 1))
    x_l_ext = np.tile(x_l_sample, (num_t_train, 1))
    x_r_ext = np.tile(x_r_sample, (num_t_train, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_i = xt_init[id_i, :]
    xt_f = xt_2d_ext[id_f, :]

    ## set data as tensor and send to device
    xt_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    # x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    xt_i_train = torch.tensor(xt_init, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_2d, requires_grad=True, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    xt_l_train = torch.tensor(xt_l, requires_grad=True, dtype=torch.float32).to(device)
    xt_r_train = torch.tensor(xt_r, requires_grad=True, dtype=torch.float32).to(device)
    xt_up_train = torch.tensor(xt_up, requires_grad=True, dtype=torch.float32).to(device)
    xt_dw_train = torch.tensor(xt_dw, requires_grad=True, dtype=torch.float32).to(device)

    # ## instantiate model
    # model = DnCNN(channels=3).to(device)
    # print(model)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    train_loss = np.zeros((epochs, 1))
    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train)
            loss = loss_pde + loss_bc + 100 * loss_ic
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}')
            loss_history["loss_pde"].append(loss_pde.item())
            loss_history["loss_bc"].append(loss_bc.item())
            loss_history["loss_ic"].append(loss_ic.item())
            train_loss[epoch, 0] = loss
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        loss_history["train_loss"].append(loss_value)
    print('start training...')
    tic = time.time()
    # for epoch in range(1, epochs + 1):
    for epoch in range(0, epochs):
        train(epoch)
        print(f'Train Epoch: {epoch + 1}, Train Loss: {train_loss[epoch, 0]:6f}', flush=True)
    toc = time.time()
    print(f'total training time: {toc - tic}')
    np.savetxt("Figures-1/train_loss_laplacian-CNN-condition-1.txt", train_loss)
    ##plot loss progress
    fig, ax = newfig(2.0, 1.1)
    # ax.axis('off')
    plt.title("loss")
    plt.plot(range(1, epochs + 1), loss_history["train_loss"], label="total loss", linewidth=3)
    plt.plot(range(1, epochs + 1), loss_history["loss_pde"], label="loss pde", linewidth=3)
    plt.plot(range(1, epochs + 1), loss_history["loss_ic"], label="loss ic", linewidth=3)
    plt.plot(range(1, epochs + 1), loss_history["loss_bc"], label="loss bc", linewidth=3)
    plt.ylabel("Loss", fontsize=22)
    plt.xlabel("Epochs", fontsize=22)
    plt.legend(fontsize=20)
    fig.savefig('Figures-1/Loss.png', dpi=300)
    fig.savefig('Figures-1/Loss.pdf')
    #plt.show()
    # test
    u_test = np.zeros((num_t, num_x, num_y))
    print('画图中')
    for i in tqdm(range(0, num_t)):
        xt = np.concatenate((t[i]*np.ones((num_x*num_y, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)

        # print(xt_tensor.shape)
        # print('===================')
        # exit()
        y_pred = model(xt_tensor)
        u_test[i,:,:] = to_numpy(y_pred[:, 0]).reshape(num_x, num_y)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off')
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(u_test[i,:,:], interpolation='nearest', cmap='rainbow',
                      # extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('Figures-1/u_'+str(i+1000)+'.png')

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]='1'
    ## parameters
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)
    # TODO 2 选择网络
    #model = Model().to(device)
    #model = DnCNN(channels=3).to(device)
    #model = RDN(args).to(device)
    #model = VDN(in_channels=3).to(device)
    model = cbdnet().to(device)
    main(model)

