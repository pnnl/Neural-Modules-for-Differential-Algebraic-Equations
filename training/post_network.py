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
from sklearn.metrics import mean_squared_error

from collections import OrderedDict
from abc import ABC, abstractmethod
import seaborn as sns
sns.set_context("paper", font_scale=2.8,rc={"lines.linewidth": 2})
# Set device:
device = 'cpu'

# Problem-specific constants:
data_1 = np.float32(np.loadtxt('tank_oscillation_sqrt_1.dat').transpose())
data_2 = np.float32(np.loadtxt('tank_oscillation_sqrt_2.dat').transpose())
nx = 9
dt = 1.0
# %%
def construct_model():
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
        
    d = blocks.Drain()

    # Non-algebraic agents:
    tank_1 = physics.MIMOTank(state_keys=["h_1"], in_keys = ["h_1"], profile=profile_1)
    tank_2 = physics.MIMOTank(state_keys=["h_2"], in_keys = ["h_2"], profile=profile_2)
    tank_3 = physics.MIMOTank(state_keys=["h_3"], in_keys = ["h_3"], profile=profile_3)
    tank_4 = physics.MIMOTank(state_keys=["h_4"], in_keys = ["h_4"], profile=profile_4)

    # Define algebraic agents according to above. 
    # These are mappings from in_keys -> state_keys. Dimensionality changes through these mappings.
    nln_pump = physics.SISOConservationNode(state_keys = ["m_1"], in_keys = ["h_1","h_2"], solver = pump_dynamics)
    outlet_1 = physics.SISOConservationNode(state_keys = ["m_4"], in_keys = ["h_2"], solver = d)
    outlet_2 = physics.SISOConservationNode(state_keys = ["m_5"], in_keys = ["h_1"], solver = d)
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

    return problem

# %% Test:

def Calculate_MSE(data, pred):
    
    MSE= mean_squared_error(data,pred)
    return MSE 

problem = construct_model()

SNRDB = [90,40,30,20]
SNRDB = [20]
np.random.seed(1452)
def add_snr(data,db):

    snr = 10**(db/10)

    for i in range(len(data[0,:])):
        signal_power = np.mean( data[:,i]**2 )
        std_n = (signal_power / snr)**0.5
        if snr > 1e8:
            continue
        data[:,i] += 1.0*np.random.normal(0,std_n,len(data[:,0]))
    return data

for db in SNRDB:
    torch.manual_seed(1)
    print(db)
    path = "network_"+str(db)+".pth"

    # Load dict:
    problem.load_state_dict(torch.load(path))
    problem.eval()

    # Add the noise back into signal:
    d2 = add_snr(data_2,db)

    # Get IC:
    input = {'X': torch.empty(1), 'xn': torch.tensor(d2[0,:]).unsqueeze(0).unsqueeze(0)}

    # Rollout:
    problem.nodes[0].nsteps = 400
    problem.nodes[0].nodes[0].callable.block.agents[0].profile = lambda x: 2.0
    print('alpha = ' + str(problem.nodes[0].nodes[0].callable.block.agents[6].solver.coeff))
    print('alpha = ' + str(problem.nodes[0].nodes[0].callable.block.agents[7].solver.coeff))
    out = problem.nodes[0](input)
    sol = out['xn'][0]

    # plt.style.use('default')
    # plt.rcParams["font.family"] = "serif"
    # #plt.rcParams["font.serif"] = ["Times"]
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams.update({'font.size': 10})

    # params = {'legend.fontsize': 10,
    #         'axes.labelsize': 10,
    #         'axes.titlesize': 10,
    #         'xtick.labelsize': 10,
    #         'ytick.labelsize': 10}
    # plt.rcParams.update(params)
    fig, ax  = plt.subplots(2,1,figsize=[10,12],constrained_layout=True)

    time = np.float32(np.linspace(0.0,len(d2[:,0])-1,len(d2[:,0])).reshape(-1, 1))*1
    
    
    ax[0].plot(time,d2[:,0],label="Data-Tank #1",linestyle="--",marker='o',markevery=20)
    ax[0].plot(time,d2[:,1],label="Data-Tank #2",linestyle="--",marker='>',markevery=20)
    ax[0].plot(time,d2[:,2],label="Data-Tank #3",linestyle="--",marker='^',markevery=20)
    ax[0].plot(time,d2[:,3],label="Data-Tank #4",linestyle="--",marker='s',markevery=20)
    ax[0].plot(time,sol.detach().numpy()[:,0],label="Pred-Tank #1")
    ax[0].plot(time,sol.detach().numpy()[:,1],label="Pred-Tank #2")
    ax[0].plot(time,sol.detach().numpy()[:,2],label="Pred-Tank #3")
    ax[0].plot(time,sol.detach().numpy()[:,3],label="Pred-Tank #4")

    # plt.plot(time,d2[:,0:4],label="Data",linestyle="--")
    ax[0].set_xlim([0,400])
    ax[0].set_ylim([0,3])
    # ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Height")
    ax[0].legend(ncol=2,fontsize=18)
    # ax[0].legend(ncol=2)
    # plt.savefig('network_'+str(db)+'.png')
    # plt.show()

    for i in range(4):
        MSE_ = Calculate_MSE(d2[:,i], sol.detach().numpy()[:,i])
        print(f" Network_ Tank #{i}, MSE: {MSE_}")

    MSE_agg = Calculate_MSE(np.sum(d2[:,0:4],axis=1), np.sum(sol.detach().numpy()[:,0:4],axis=1))
    print(f" Network_ Agg Tank, MSE: {MSE_agg}")

    # Get IC:
    input = {'X': torch.empty(1), 'xn': torch.tensor(data_1[0,:]).unsqueeze(0).unsqueeze(0)}
    problem.nodes[0].nodes[0].callable.block.agents[0].profile = lambda x: 1.0

    out = problem.nodes[0](input)
    sol = out['xn'][0]

    
    
    ax[1].plot(time,data_1[:,0],label="Data-Tank #1",linestyle="--",marker='o',markevery=20)
    ax[1].plot(time,data_1[:,1],label="Data-Tank #2",linestyle="--", marker='>',markevery=20)
    ax[1].plot(time,data_1[:,2],label="Data-Tank #3",linestyle="--", marker='^',markevery=20)
    ax[1].plot(time,data_1[:,3],label="Data-Tank #4",linestyle="--",marker='s',markevery=20)
    ax[1].plot(time,sol.detach().numpy()[:,0],label="Pred-Tank #1")
    ax[1].plot(time,sol.detach().numpy()[:,1],label="Pred-Tank #2")
    ax[1].plot(time,sol.detach().numpy()[:,2],label="Pred-Tank #3")
    ax[1].plot(time,sol.detach().numpy()[:,3],label="Pred-Tank #4")
    ax[1].set_xlim([0,400])
    ax[1].set_ylim([0,3])
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Height")

    for i in range(4):
        MSE_ = Calculate_MSE(data_1[:,i], sol.detach().numpy()[:,i])
        print(f" Extrapolation Tank #{i}, MSE: {MSE_}")
    
    MSE_agg_extrap = Calculate_MSE(np.sum(data_1[:,0:4],axis=1), np.sum(sol.detach().numpy()[:,0:4],axis=1))
    print(f" Network_ Unseen Initial condition Agg Tank , MSE: {MSE_agg_extrap}")
    # handles, labels = ax[1].get_legend_handles_labels()
    # lgd = ax[1].legend(handles, labels, loc='upper center',ncol=2, bbox_to_anchor=(0.5,-0.1))
    # text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
    ax[1].legend(ncol=2,fontsize=18)

    ax[0].annotate("a)", xy=(-0.15, 1.05), xycoords="axes fraction")
    ax[1].annotate("b)", xy=(-0.15, 1.05), xycoords="axes fraction")
    
    # fig.show()
    fig.savefig('network_noise_extrap'+str(db)+'.png', bbox_inches='tight')
    #####
    # For each model, we need to report the MSE for training, extrap, and parameters for \alpha_1 and \alpha_2



# %%