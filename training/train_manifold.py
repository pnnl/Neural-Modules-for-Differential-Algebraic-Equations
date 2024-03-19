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

# Problem-specific constants:
area_data = np.loadtxt('area.dat')
nx = 4
nu = 1
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
    data = np.float32(np.loadtxt('tanks_splits.dat'))
    data = data[:len(data[:,0])//2,:]
    data = data[1:,:]
    # Add noise:
    data = add_snr(data,snr)

    # Create time vector
    time = np.float32(np.linspace(0.0,len(data[:,0])-1,len(data[:,0])).reshape(-1, 1))*dt
    # Create exogenous input vector:
    U = time*0.0 + 0.5
    # Create input data array:
    raw = np.concatenate((data,U),axis=1)
    # Dictionary:
    train_data = {'Y': torch.tensor(raw), 'X': torch.tensor(raw), 'Time': torch.tensor(time)}
    dev_data = train_data
    test_data = train_data

    nsim = raw.shape[0]
    nx = raw.shape[1]
    nstep = 5
    
    for d in [train_data, dev_data]:
        d['X'] = d['X'].reshape(nsim//nstep, nstep, nx)
        d['Y'] = d['Y'].reshape(nsim//nstep, nstep, nx)
        d['xn'] = d['X'][:, 0:1, :] # Add an initial condition to start the system loop
        d['Time'] = d['Time'].reshape(nsim//nstep, -1)

    train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
    train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]

    """
    Below are the actual states and their indices:

    sim['X'] contains:
    0 - height of tank 1
    1 - height of tank 2
    2 - m_dot into tank 1
    3 - m_dot into tank 2
    4 - m_dot into system

    """
    states = {}
    states['h_1'] = 0
    states['h_2'] = 1
    states['m_1'] = 2
    states['m_2'] = 3
    states['m'] = 4

    # Tank area - height profiles: These should map height to area. R^1 -> R^1.
    tank_profile = blocks.MLP(insize=1, outsize=1, hsizes=[5],
                                linear_map=slim.maps['linear'],
                                nonlin=nn.Sigmoid)

    # Surrogate for algebra solver: This should map 'algebraic state indices' to len(state names). 
    algebra_solver = blocks.MLP(insize=2, outsize=1, hsizes=[3],
                                linear_map=slim.maps['linear'],
                                nonlin=nn.Sigmoid)
                                
    # Individual components:
    tank_1 = physics.MIMOTank(state_keys=["h_1"], in_keys=["h_1"], profile= lambda x: 3.0)
    tank_2 = physics.MIMOTank(state_keys=["h_2"], in_keys=["h_2"], profile=tank_profile)
    pump = physics.SourceSink(state_keys=["m"], in_keys=["m"])

    # Define algebraic agent:
    #manifold = physics.SIMOConservationNode(in_keys = ["m","h_1","h_2","m_1","m_2"], state_keys=["m_1","m_2"], solver=algebra_solver)
    manifold = physics.SIMOConservationNode(in_keys = ["m","h_1","h_2"], state_keys=["m_1","m_2"], solver=algebra_solver)

    # Accumulate agents in list:
    # index:   0       1        2       3 
    agents = [pump, tank_1, tank_2, manifold]

    couplings = []
    # Couple w/ pipes:
    couplings.append(physics.Pipe(in_keys = ["m"], pins = [[0,3]])) # Pump -> Manifold
    couplings.append(physics.Pipe(in_keys = ["m_1"], pins = [[3,1]])) # Manifold -> tank_1
    couplings.append(physics.Pipe(in_keys = ["m_2"], pins = [[3,2]])) # Manifold -> tank_2

    model_ode = ode.GeneralNetworkedODE(
        states=states,
        agents=agents,
        couplings=couplings,
        insize=nx+nu,
        outsize=nx+nu,
    )

    model_algebra = ode.GeneralNetworkedAE(
        states=states,
        agents=agents,
        insize=nx+nu,
        outsize=nx+nu,
    )

    fx_int = integrators.EulerDAE(model_ode,algebra=model_algebra,h=1.0)
    dynamics_model = System([Node(fx_int,['xn'],['xn'])])

    x = variable("X")
    xhat = variable("xn")[:, :-1, :]
    reference_loss = ((xhat[:,:,[2,3]] == x[:,:,[2,3]])^2)
    reference_loss = ((xhat == x)^2)
    reference_loss.name = "ref_loss"

    height_loss = (1.0e0*(xhat[:,:,0] == xhat[:,:,1])^2)
    height_loss = (0.0*(xhat[:,:,0] == xhat[:,:,1])^2)
    height_loss.name = "height_loss"

    objectives = [reference_loss, height_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)

    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                        stdout=["dev_loss","train_loss"])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=nEpochs,
        patience=30,
        warmup=5,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        logger=logger,
    )

    trained_model_dict = trainer.train()
    return trained_model_dict, fx_int, tank_profile, data, time, U

def rollout(nstep, integrator, x0, U):
    sol = torch.zeros((nstep,5))
    sol[:,-1] = U[:,0]
    for j in range(sol.shape[0] - 1):
        if j==0:
            sol[[0],:] = x0.float()
            sol[[j+1],:4] = integrator(ic)[:,:4]
        else:
            sol[[j+1],:4] = integrator(sol[[j],:])[:,:4]
    return sol    

# %%

#SNRDB = [50,40,30,20]
SNRDB = [90]

for db in SNRDB:
    print(db)
    model_dict, fx_int, tank_profile, data, time, U = train_noise(db,20000)

    path = "manifold_"+str(db)+".pth"

    torch.save(model_dict, path)


    nsteps =500
    x0 = np.concatenate((data[1,:],U[0]))
    ic = torch.unsqueeze(torch.tensor(x0),0).float()

    sol = rollout(nsteps, fx_int, ic, torch.tensor(U))

    plt.style.use('default')
    plt.rcParams["font.family"] = "serif"
    #plt.rcParams["font.serif"] = ["Times"]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 10})

    params = {'legend.fontsize': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10}
    plt.rcParams.update(params)


    plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    plt.plot(time[0:-1],data[1:,0],label="Data",linestyle="--")
    plt.xlim([0,500])
    plt.ylim([0,40])
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.savefig('tanks_'+str(db)+'.png')
    plt.show()


    plt.plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    plt.plot(time,sol.detach().numpy()[:,3],label="Inflow #1")
    plt.plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    plt.plot(time[0:-1],data[1:,2],label="Data Inflow #1",linestyle="--")
    plt.plot(time[0:-1],data[1:,3],label="Data Inflow #2",linestyle="--")

    plt.xlim([0,500])
    plt.ylim([0,0.6])
    plt.xlabel("Time")
    plt.ylabel("Volumetric Flow")
    plt.legend()
    plt.savefig('flows_'+str(db)+'.png')
    plt.show()


    h = torch.tensor(np.linspace(0,40,401)).unsqueeze(-1).float()

    area = tank_profile(h)

    plt.plot(h.detach().numpy(),area.detach().numpy(),label="Model")
    plt.plot(h.detach().numpy(),area_data,label="Actual")
    plt.ylim([0,6])
    plt.xlim([0,35])
    plt.xlabel("Height")
    plt.ylabel("Area(h)")
    plt.legend()
    plt.savefig('areas_'+str(db)+'.png')
    plt.show()


    ###############
    # Eval on new conditions:
    nsteps = 501
    data = np.float32(np.loadtxt('tanks_splits.dat'))
    data = data[501:,]
    time = np.float32(np.linspace(0.0,len(data[:,0])-1,len(data[:,0])).reshape(-1, 1))*dt
    U = time*0.0 + 0.5 + 0.25*np.sin(time/100.0)

    x0 = np.concatenate((data[1,:],U[0]))
    ic = torch.unsqueeze(torch.tensor(x0),0).float()

    sol = rollout(nsteps, fx_int, ic, torch.tensor(U))

    plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    plt.plot(time[0:-1],data[1:,0],label="Truth",linestyle="--")
    plt.xlim([0,500])
    plt.ylim([0,50])
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.savefig('extrap_tanks_'+str(db)+'.png')
    plt.show()


    plt.plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    plt.plot(time,sol.detach().numpy()[:,3],label="Inflow #1")
    plt.plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    plt.plot(time[0:-1],data[1:,2],label="Data Inflow #1",linestyle="--")
    plt.plot(time[0:-1],data[1:,3],label="Data Inflow #2",linestyle="--")

    plt.xlim([0,500])
    plt.ylim([0,0.85])
    plt.xlabel("Time")
    plt.ylabel("Volumetric Flow")
    plt.legend()
    plt.savefig('extrap_flows_'+str(db)+'.png')
    plt.show()

# %%