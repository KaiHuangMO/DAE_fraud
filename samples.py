from sklearn.datasets import make_classification

import torch as th
from torch import nn, optim
from torch.autograd import Variable as V
import numpy as np
import random

from torch.optim.lr_scheduler import LambdaLR
from dae import DAE
input_dim = 600
output_dim = 600
BATHCSIZE = 100

def create_batch1(x, y, batch_size, shuffle):
    if shuffle:
        a = list(range(len(x)))
        np.random.shuffle(a)
        x = x[a]
        y = y[a]

    batch_x = [x[batch_size * i: (i + 1) * batch_size, :].tolist() for i in range(len(x) // batch_size)]
    batch_y = [y[batch_size * i: (i + 1) * batch_size, :].tolist() for i in range(len(x) // batch_size)]
    return np.array(batch_x), np.array(batch_y)

def create_batch2(x_max_noise, x_max, x_min_noise, x_min, batch_size, shuffle):
    if shuffle:
        a = list(range(len(x_max_noise)))
        np.random.shuffle(x_max_noise)

        x_max_noise = x_max_noise[a]
        x_max = x_max[a]
        x_min_noise = x_min_noise[a]
        x_min = x_min[a]
    # mearge
    batch_size = int(batch_size / 2)
    noise_train = []
    normal_train = []
    tround = int(len(x_max) // batch_size)
    for i in range(0, tround):
        t1 = []
        t1.extend(x_max_noise[batch_size * i: (i + 1) * batch_size, :].tolist())
        t1.extend(x_min_noise[batch_size * i: (i + 1) * batch_size, :].tolist())
        noise_train.append(t1)

        t2 = []
        t2.extend(x_max[batch_size * i: (i + 1) * batch_size, :].tolist())
        t2.extend(x_min[batch_size * i: (i + 1) * batch_size, :].tolist())
        normal_train.append(t2)


    return np.array(noise_train), np.array(normal_train)

def train_(model_dae, trainx, trainy, opt, loss_f, batch_size):
    model_dae.train()
    batch_x, batch_y = create_batch1(trainx, trainy, batch_size, True)
    run_loss, pred = 0., 0
    for x, y in zip(batch_x, batch_y):
        opt.zero_grad()
        x, y = V(th.Tensor(x)), V(th.Tensor(y))
        output = model_dae(x, False)
        loss = loss_f(output, y)
        loss.backward()
        opt.step()
        run_loss += loss.item()

    return run_loss / (len(trainx) // batch_size)


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        slide = int(x.shape[0] / 2)
        x_max = x[0: slide]
        x_min = x[slide:]
        y_max = y[0: slide]
        y_min = y[slide:]

        avg_max = th.mean(th.pow((x_max - y_max), 2))
        avg_min = th.mean(th.pow((x_min - y_min), 2))

        if avg_min < 0: avg_min = 0
        if avg_min > 5 * avg_max: avg_min = 5 * avg_max


        return avg_max - 0.1 * avg_min


def train_new(model_dae, x_max_noise, x_max, x_min_noise, x_min, opt, loss_f, batch_size):
    model_dae.train()
    batch_x, batch_y = create_batch2(x_max_noise, x_max, x_min_noise, x_min, batch_size, True)
    run_loss, pred = 0., 0
    for x, y in zip(batch_x, batch_y):
        opt.zero_grad()
        x, y = V(th.Tensor(x)), V(th.Tensor(y))
        output = model_dae(x, False)
        loss = loss_f(output, y)
        loss.backward()
        opt.step()
        run_loss += loss.item()

    return run_loss / (len(x_max_noise) // batch_size)

def valid_(valx, valy, opt, loss_f, batch_size):
    model_dae.eval()

    batch_x, batch_y = create_batch1(valx, valy, batch_size, False)
    run_loss = 0.
    for x, y in zip(batch_x, batch_y):
        x, y = th.Tensor(x), th.Tensor(y)
        output = model_dae(x, False)

        loss = loss_f(output, y)
        run_loss += loss.item()

    return run_loss / (len(valx) // batch_size)

def Swap_noise(array):
    height = len(array)
    width = len(array[0])
    rands = np.random.uniform(0, 1, (height, width) )
    copy_array  = np.copy(array)

    for h in range(height):
        for w in range(width):
            if rands[h, w] <= 0.10:
                swap_target_h = random.randint(0,height)
                copy_array[h, w] = array[swap_target_h-1, w]
    return copy_array


def recontruct_distance(x_train, x_recon):
    x_loss = []
    for i in range(0, len(x_train)):
        t1 = x_train[i]
        t2 = x_recon[i]
        ds = np.linalg.norm(t1 - t2)
        x_loss.append(ds)
    return x_loss

import math

def construct_database(x_max, x_min, dist_max, dist_min):
    x_rebuild_min = []
    for i in range(0, len(x_max)):
        dm = dist_max[i]
        dtmp = (dist_min - dm) ** 2
        itmp = np.argmax (dtmp)
        x_rebuild_min.append(x_min[itmp])
    return x_rebuild_min



def pretrain_DAE(x_max):
    model_dae = DAE(input_dim, output_dim)
    opt = optim.Adam(model_dae.parameters())
    loss_f = nn.MSELoss()
    scheduler = LambdaLR(opt, lr_lambda=lambda i: DECAY ** i)

    all_noise = Swap_noise(x_max)
    all_org = x_max
    #BATHCSIZE = 100
    EPOCH = 10
    for e in range(EPOCH):
        tr_loss = train_(model_dae, all_noise, all_org, opt, loss_f, BATHCSIZE)  # this is pretrain
        print(f"EPOCH: {e} tr_loss :{tr_loss}")
    lr = opt.state_dict()["param_groups"][0]["lr"]
    return model_dae

def newtrain_DEA(x_max, x_min):
    model_dae = DAE(input_dim, output_dim)
    opt = optim.Adam(model_dae.parameters())
    loss_f = My_loss()
    scheduler = LambdaLR(opt, lr_lambda=lambda i: DECAY ** i)

    tr_losses, val_losses = [], []
    tr_r2, te_r2 = [], []
    x_min = np.array(x_min)
    x_max_noise = Swap_noise(x_max)
    x_min_noise = Swap_noise(x_min)

    EPOCH1 = 10
    for e in range(EPOCH1):
        tr_loss = train_new(model_dae, x_max_noise, x_max, x_min_noise, x_min, opt, loss_f, BATHCSIZE)  # this is pretrain
        print(f"EPOCH: {e} tr_loss :{tr_loss}")
    lr = opt.state_dict()["param_groups"][0]["lr"]
    return model_dae



if __name__ == "__main__":
    DECAY = 0.95
    x, y = make_classification(n_samples=1000, n_features=600, weights=[0.95, 0.05])  # 创建数据集 95:5 比例
    x_train = x[0:600]
    y_train = y[0:600]
    x_test = x[0:600]
    y_test = y[0:600]
    x_max = []
    x_min = []
    for i in range(0, len(y_train)):
        if y_train[i] == 0:
            x_max.append(x_train[i])
        else:
            x_min.append(x_train[i])
    x_max = np.array(x_max)
    x_min = np.array(x_min)
    '''
    pretrain DAE
    '''
    model_dae = pretrain_DAE(x_max)

    '''
    compute recontract and pair samples
    '''
    from scipy.spatial.distance import cdist

    x_max_recon = model_dae(th.Tensor(x_max), False)
    dist_max = recontruct_distance(x_max, x_max_recon.detach().numpy())
    x_min_recon = model_dae(th.Tensor(x_min), False)
    dist_min = recontruct_distance(x_min, x_min_recon.detach().numpy())
    x_rebuild_min = construct_database(x_max, x_min, dist_max, dist_min)

    '''
    build new DAE
    '''
    print ('---start new train---')
    model_dae_new = newtrain_DEA(x_max, x_rebuild_min)
    x_max_recon = model_dae_new(th.Tensor(x_max), False)
    dist_max = recontruct_distance(x_max, x_max_recon.detach().numpy())
    z = 1

    # https://github.com/tommyod/KDEpy
    from KDEpy import FFTKDE
    import matplotlib.pyplot as plt

    dist_x, dist_y = FFTKDE(kernel="gaussian", bw="silverman").fit(dist_max).evaluate()

    dx = dist_x[1] - dist_x[0]  # 每个矩形的宽度
    fArea = np.sum(dist_y * dx)  # 矩形宽*高，再求和
    # 暴力求解
    h = 0
    for yi in range(0, len(dist_y)):
        fArea = np.sum(dist_y[:yi] * dx)
        if fArea > .95:
            h = dist_x[yi]
    print (h)

    plt.plot(dist_x, dist_y)
    plt.show()

    '''
    for inference
    '''
    x_test_recon =  model_dae_new(th.Tensor(x_test), False)
    dist_test = recontruct_distance(x_test, x_test_recon.detach().numpy())
    y_pred = []
    for dt in dist_test:
        if dt > h:
            y_pred.append(1)
        else:
            y_pred.append(0)
    from sklearn.metrics import confusion_matrix
    print ( confusion_matrix(y_test, y_pred) )





