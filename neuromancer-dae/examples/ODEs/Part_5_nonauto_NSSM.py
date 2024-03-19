"""
Learning neural state space model (SSM) with exogenous inputs from time series data
"""

import torch
import torch.nn as nn

from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks


def get_data(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    nx = sys.nx
    nu = sys.nu
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    mean_x = modelSystem.stats['X']['mean']
    std_x = modelSystem.stats['X']['std']
    mean_u = modelSystem.stats['U']['mean']
    std_u = modelSystem.stats['U']['std']
    def normalize(x, mean, std):
        return (x - mean) / std

    trainX = normalize(train_sim['X'][:length], mean_x, std_x)
    trainX = trainX.reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = normalize(train_sim['U'][:length], mean_u, std_u)
    trainU = trainU.reshape(nbatch, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :],
                              'U': trainU}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = normalize(dev_sim['X'][:length], mean_x, std_x)
    devX = devX.reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = normalize(dev_sim['U'][:length], mean_u, std_u)
    devU = devU[:length].reshape(nbatch, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :],
                            'U': devU}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = normalize(test_sim['X'][:length], mean_x, std_x)
    testX = testX.reshape(1, nbatch*nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = normalize(test_sim['U'][:length], mean_u, std_u)
    testU = testU.reshape(1, nbatch*nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    test_data = {'X': testX, 'xn': testX[:, 0:1, :],
                 'U': testU}

    return train_loader, dev_loader, test_data


class SSM(nn.Module):
    """
    Baseline class for (neural) state space model (SSM)
    Implements discrete-time dynamical system:
        x_k+1 = fx(x_k) + fu(u_k)
    with variables:
        x_k - states
        u_k - control inputs
    """
    def __init__(self, fx, fu, nx, nu):
        super().__init__()
        self.fx, self.fu = fx, fu
        self.nx, self.nu = nx, nu
        self.in_features, self.out_features = nx+nu, nx

    def forward(self, x, u, d=None):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nu])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        # state space model
        x = self.fx(x) + self.fu(u)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    # select system:
    #   TwoTank, CSTR, SwingEquation,

    # %%  ground truth system
    system = psl.systems['CSTR']
    modelSystem = system()
    ts = modelSystem.ts
    nx = modelSystem.nx
    nu = modelSystem.nu
    raw = modelSystem.simulate(nsim=1000, ts=ts)
    plot.pltOL(Y=raw['Y'], U=raw['U'])
    plot.pltPhase(X=raw['Y'])

    # get datasets
    nsim = 2000
    nsteps = 10
    bs = 100
    train_loader, dev_loader, test_data = \
        get_data(modelSystem, nsim, nsteps, ts, bs)

    # instantiate neural nets
    fx = blocks.MLP(nx, nx, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ReLU,
                     hsizes=[40, 40])
    fu = blocks.MLP(nu, nx, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[40, 40])
    # construct NSSM model in Neuromancer
    ssm = SSM(fx, fu, nx, nu)
    # construct symbolic model
    model = Node(ssm, ['xn', 'U'], ['xn'], name='NODE')
    dynamics_model = System([model], name='system')

    # %% Constraints + losses:
    x = variable("X")
    xhat = variable('xn')[:, :-1, :]

    # trajectory tracking loss
    reference_loss = 5.*(xhat == x)^2
    reference_loss.name = "ref_loss"

    # one step tracking loss
    onestep_loss = 1.*(xhat[:, 1, :] == x[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    # %%
    objectives = [reference_loss, onestep_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)
    # plot computational graph
    problem.show()

    # %%
    optimizer = torch.optim.Adam(problem.parameters(),
                                 lr=0.003)

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=100,
        warmup=100,
        epochs=1000,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )
    # %%
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    # %%

    # Test set results
    test_outputs = dynamics_model(test_data)

    pred_traj = test_outputs['xn'][:, :-1, :].detach().numpy().reshape(-1, nx).transpose(1, 0)
    true_traj = test_data['X'].detach().numpy().reshape(-1, nx).transpose(1, 0)
    input_traj = test_data['U'].detach().numpy().reshape(-1, nu).transpose(1, 0)

    # plot rollout
    figsize = 25
    fig, ax = plt.subplots(nx + nu, figsize=(figsize, figsize))

    x_labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, x_labels)):
        axe = ax[row]
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, 'c', linewidth=4.0, label='True')
        axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)

    u_labels = [f'$u_{k}$' for k in range(len(input_traj))]
    for row, (u, label) in enumerate(zip(input_traj, u_labels)):
        axe = ax[row+nx]
        axe.plot(u, linewidth=4.0, label='inputs')
        axe.legend(fontsize=figsize)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.tick_params(labelbottom=True, labelsize=figsize)

    ax[-1].set_xlabel('$time$', fontsize=figsize)
    plt.tight_layout()