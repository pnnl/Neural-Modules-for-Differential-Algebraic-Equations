# %%
"""|
System ID for networked differential-algebraic dyanmical system. Toy problem for testing new class designs.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import neuromancer.slim as slim
from neuromancer.modules import blocks, activations
from neuromancer.dynamics import integrators, ode, physics
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable, Objective
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger

from collections import OrderedDict
from abc import ABC, abstractmethod

# Set device:
device = 'cpu'

dt = 1.0

# %%
def add_snr(data,db):

    snr = 10**(db/10)

    for i in range(len(data[0,:])):
        signal_power = np.mean( data[:,i]**2 )
        std_n = (signal_power / snr)**0.5
        if snr > 1e8:
            continue
        data[:,i] += 1.0*np.random.normal(0,std_n,len(data[:,0]))
    return data

def train_noise(snr,nEpochs):
    torch.manual_seed(0)
    # Load data from text file:
    print(os.getcwd())
    raw = np.float32(np.loadtxt('./tank_oscillation_sqrt_2.dat').transpose())
    raw = add_snr(raw,snr)

    time = np.float32(np.linspace(0.0,400-dt,len(raw[:,0])).reshape(-1, 1))

    train_data = {'Y': raw[1:], 'X': raw[1:], 'Time': time[1:]}
    dev_data = train_data
    test_data = train_data

    nsim = raw.shape[0]
    nx = raw.shape[1]
    nstep = 20

    for d in [train_data, dev_data]:
        d['X'] = d['X'].reshape(nsim//nstep, nstep, nx)
        d['Y'] = d['Y'].reshape(nsim//nstep, nstep, nx)
        d['xn'] = d['X'][:, 0:1, :] # Add an initial condition to start the system loop
        d['Time'] = d['Time'].reshape(nsim//nstep, -1)

    train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
    train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]

    states = {}
    states['h_1'] = 0
    states['h_2'] = 1
    states['h_3'] = 2
    states['h_4'] = 3
    states['m_1'] = 4
    states['m_2'] = 5
    states['m_3'] = 6
    states['m_4'] = 7
    states['m_5'] = 8

    # Tank area - height profiles: These should map height to area. R^1 -> R^1.
    profile_1 = lambda x: 2.0 
    profile_2 = lambda x: 1.0
    profile_3 = lambda x: 1.0
    profile_4 = lambda x: 10.0

    algebra_solver = blocks.MLP(insize=2, outsize=1, hsizes=[30,30],
                                linear_map=slim.maps['linear'],
                                nonlin=nn.ReLU)

    pump_dynamics = blocks.MLP(insize=2, outsize=1, hsizes=[30,30],
                                linear_map=slim.maps['linear'],
                                nonlin=nn.ReLU)
        
    d1 = blocks.Drain()
    d2 = blocks.Drain()

    # Non-algebraic agents:
    tank_1 = physics.MIMOTank(state_keys=["h_1"], in_keys = ["h_1"], profile=profile_1)
    tank_2 = physics.MIMOTank(state_keys=["h_2"], in_keys = ["h_2"], profile=profile_2)
    tank_3 = physics.MIMOTank(state_keys=["h_3"], in_keys = ["h_3"], profile=profile_3)
    tank_4 = physics.MIMOTank(state_keys=["h_4"], in_keys = ["h_4"], profile=profile_4)

    # Define algebraic agents according to above. 
    # These are mappings from in_keys -> state_keys. Dimensionality changes through these mappings.
    nln_pump = physics.SISOConservationNode(state_keys = ["m_1"], in_keys = ["h_1","h_2"], solver = pump_dynamics)
    outlet_1 = physics.SISOConservationNode(state_keys = ["m_4"], in_keys = ["h_2"], solver = d1)
    outlet_2 = physics.SISOConservationNode(state_keys = ["m_5"], in_keys = ["h_1"], solver = d2)
    # manifold input is m_1. Need to split m_1 into m_2 and m_3 s.t. m_2+m_3 = m_1. How to split stream?
    manifold = physics.SIMOConservationNode(state_keys = ["m_2","m_3"], in_keys = ["m_1","h_2","h_3"], solver = algebra_solver)

    # Accumulate agents in lists:
    # index:    0        1      2       3        4         5          6        7 
    agents = [tank_1, tank_2, tank_3, tank_4, manifold, nln_pump, outlet_1, outlet_2]

    couplings = []
    # Couple w/ pipes:
    couplings.append(physics.Pipe(in_keys = ["m_1"], pins = [[0,5],[5,4]])) # Tank 1 <-> Pump <-> Manifold
    couplings.append(physics.Pipe(in_keys = ["m_2"], pins = [[4,1]])) # Tank 2 <-> Manifold
    couplings.append(physics.Pipe(in_keys = ["m_3"], pins = [[4,2]])) # Tank 3 <-> Manifold
    couplings.append(physics.Pipe(in_keys = ["m_4"], pins = [[1,3]])) # Tank 2 <-> Tank 4
    couplings.append(physics.Pipe(in_keys = ["m_5"], pins = [[3,0]])) # Tank 1 <-> Tank 4

    model_ode = ode.GeneralNetworkedODE(
        states=states,
        agents=agents,
        couplings=couplings,
        insize=nx,
        outsize=nx,
    )

    model_algebra = ode.GeneralNetworkedAE(
        states=states,
        agents=agents,
        insize=nx,
        outsize=nx,
    )

    fx_int = integrators.EulerDAE(model_ode,algebra=model_algebra,h=dt)
    dynamics_model = System([Node(fx_int,['xn'],['xn'])])

    x = variable("X")
    xhat = variable("xn")[:, :-1, :]

    reference_loss = 1.0*((xhat == x)^2)
    reference_loss.name = "ref_loss"

    xFD = (x[:, 1:, :] - x[:, :-1, :])
    xhatFD = (xhat[:, 1:, :] - xhat[:, :-1, :])

    fd_loss = 0.0*((xFD == xhatFD)^2)
    fd_loss.name = 'FD_loss'

    objectives = [reference_loss,fd_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)

    optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                        stdout=["dev_loss","train_loss"])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=nEpochs,
        patience=10,
        warmup=500,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        logger=logger,
    )
    best_model = trainer.train()

    return best_model, fx_int, raw, time

def rollout(nstep, integrator, x0):
    sol = torch.zeros((nstep,9))
    for j in range(sol.shape[0] - 1):
        if j==0:
            sol[[0],:] = x0.float()
            sol[[j+1],:] = integrator(sol[[j],:])
        else:
            sol[[j+1],:] = integrator(sol[[j],:])
    return sol    

# %%

SNRDB = [90,40,30,20]

for db in SNRDB:
    print(db)
    model_dict, fx_int, data, time = train_noise(db,10000)

    path = "network_"+str(db)+".pth"

    torch.save(model_dict, path)


    nsteps = 400
    ic = torch.unsqueeze(torch.tensor(data[0,:]),0).float()

    sol = rollout(nsteps, fx_int, ic)

    plt.plot(time[0:-1],sol.detach().numpy()[:,0],label="Tank #1")
    plt.plot(time[0:-1],sol.detach().numpy()[:,1],label="Tank #2")
    plt.plot(time[0:-1],sol.detach().numpy()[:,2],label="Tank #3")
    plt.plot(time[0:-1],sol.detach().numpy()[:,3],label="Tank #4")
    plt.plot(time,data[:,0:4],label="Data",linestyle="--")
    plt.xlim([0,400])
    plt.ylim([0,3])
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.show()

# %%