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

from sklearn.metrics import mean_squared_error

import seaborn as sns
sns.set_context("paper", font_scale=2.8,rc={"lines.linewidth": 2})
# Set device:
device = 'cpu'

# Problem-specific constants:
raw = np.float32(np.loadtxt('tanks_splits.dat'))
data_1 = raw[:len(raw[:,0])//2,:]
mDot = np.sum((data_1[:,2],data_1[:,3]),axis=0,keepdims=True).transpose()
data_1 = np.concatenate((data_1,mDot),axis=1)

data_2 = raw[-len(raw[:,0])//2:,:]
mDot = np.sum((data_2[:,2],data_2[:,3]),axis=0,keepdims=True).transpose()
data_2 = np.concatenate((data_2,mDot),axis=1)

area_data = np.loadtxt('area.dat')
nx = 4
nu = 1
dt = 1.0
# %%
def Calculate_MSE(data, pred):
    
    MSE= mean_squared_error(data,pred)
    return MSE 


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

def construct_model():
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
    reference_loss.name = "ref_loss"

    height_loss = (1.0e0*(xhat[:,:,0] == xhat[:,:,1])^2)
    height_loss.name = "height_loss"

    objectives = [reference_loss, height_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)

    return problem, model_ode,model_algebra

# %% Test:

problem,model_ode,model_algebra = construct_model()

# SNRDB = [90,50,40,30,20]
SNRDB = [90]

for db in SNRDB:
    print(db)
    path = "manifold_"+str(db)+".pth"

    # Load dict:
    problem.load_state_dict(torch.load(path))
    problem.eval()

    # Get IC:
    input = {'X': torch.empty(1), 'xn': torch.tensor(data_1[0,:]).unsqueeze(0).unsqueeze(0)}

    # Rollout:
    problem.nodes[0].nsteps = 500
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

    # time = np.float32(np.linspace(0.0,len(data_1[:,0])-1,len(data_1[:,0])).reshape(-1, 1))*1
    time = np.float32(np.linspace(0.0,problem.nodes[0].nsteps,problem.nodes[0].nsteps+1).reshape(-1, 1))*1   
    fig, ax  = plt.subplots(1,2,figsize=[14,8],constrained_layout=True)

    ax[0].plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    ax[0].plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    ax[0].plot(time[0:-1],data_1[1:,0],label="Data",linestyle="--")
    ax[0].set_xlim([0,500])
    ax[0].set_ylim([0,40])
    ax[0].set_xlabel("Time") 
    ax[0].set_ylabel("Height")
    ax[0].legend()
    # ax[0].show()
    print(data_1[1:,0].shape)
    MSE_0 = Calculate_MSE(data_1[1:,0], sol.detach().numpy()[:-1,0])
    MSE_1 = Calculate_MSE(data_1[1:,0], sol.detach().numpy()[:-1,1])
    MSE_Height_agg = Calculate_MSE(data_1[1:,0], np.sum(sol.detach().numpy()[:-1,0:1],axis=1))
    
    print(f" Manifold height #{0}, MSE: {MSE_0}")
    print(f" Manifold height #{1}, MSE: {MSE_1}")
    print(f" Manifold height # Agg., MSE: {MSE_Height_agg}")

    ax[1].plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    ax[1].plot(time,sol.detach().numpy()[:,3],label="Inflow #2")
    ax[1].plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    ax[1].plot(time[0:-1],data_1[1:,2],label="Data Inflow #1",linestyle="--")
    ax[1].plot(time[0:-1],data_1[1:,3],label="Data Inflow #2",linestyle="--")
    ax[1].set_xlim([0,500])
    ax[1].set_ylim([0,0.6])
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Volumetric Flow")
    ax[1].legend()
    # ax[1].show()

    MSE_vol0 = Calculate_MSE(data_1[1:,2], sol.detach().numpy()[:-1,2])
    MSE_vol1 = Calculate_MSE(data_1[1:,3], sol.detach().numpy()[:-1,3])

    MSE_vol_agg = Calculate_MSE( np.sum(data_1[1:,2:3], axis=0), np.sum(sol.detach().numpy()[:-1,2:3],axis=0))
    print(f" Manifold vol  #{1}, MSE: {MSE_vol0}")
    print(f" Manifold vol  #{2}, MSE: {MSE_vol1}")
    print(f" Manifold vol  # Agg, MSE: {MSE_vol_agg}")


    ax[0].annotate("a)", xy=(-0.01, 1.05), xycoords="axes fraction")
    ax[1].annotate("b)", xy=(-0.01, 1.05), xycoords="axes fraction")



    # plt.savefig('tanks_'+str(db)+'.svg', format='svg', dpi=1200)
    fig.savefig('tanks_flows_'+str(db)+'.png', format='png', dpi=1200)
    
    plt.close()
    # plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    # plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    # plt.plot(time[0:-1],data_1[1:,0],label="Data",linestyle="--")
    # plt.xlim([0,500])
    # plt.ylim([0,40])
    # plt.xlabel("Time") 
    # plt.ylabel("Height")
    # plt.legend()
    # plt.show()
    # plt.savefig('tanks_'+str(db)+'.svg', format='svg', dpi=1200)

    # plt.plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    # plt.plot(time,sol.detach().numpy()[:,3],label="Inflow #2")
    # plt.plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    # plt.plot(time[0:-1],data_1[1:,2],label="Data Inflow #1",linestyle="--")
    # plt.plot(time[0:-1],data_1[1:,3],label="Data Inflow #2",linestyle="--")

    # plt.xlim([0,500])
    # plt.ylim([0,0.6])
    # plt.xlabel("Time")
    # plt.ylabel("Volumetric Flow")
    # plt.legend()
    # plt.show()
    # plt.savefig('flows_'+str(db)+'.svg', format='svg', dpi=1200)


    h = torch.tensor(np.linspace(0,40,401)).unsqueeze(-1).float()

    tank_profile = problem.nodes[0].nodes[0].callable.block.agents[2].profile
    area = tank_profile(h)
    
    fig, ax  = plt.subplots(1,2,figsize=[14,8],constrained_layout=True)


    ax[0].plot(h.detach().numpy(),area.detach().numpy(),label="Model-Prediction")
    ax[0].plot(h.detach().numpy(),area_data,label="Actual")
    ax[0].set_ylim([0,6])
    ax[0].set_xlim([0,35])
    ax[0].set_xlabel("Height")
    ax[0].set_ylabel("Area(h)")
    ax[0].legend()
    # plt.show()
  
    MSE_area = Calculate_MSE(area_data, area.detach().numpy())
    
    print(f" Manifold area  #{1}, MSE: {MSE_area}")
    # %%
    ### New condition: ####
    # Get IC:
    nsteps = 501
    data = np.float32(np.loadtxt('tanks_splits.dat'))
    data = data[501:,]
    time = np.float32(np.linspace(0.0,len(data[:,0])-1,len(data[:,0])).reshape(-1, 1))*dt
    U = time*0.0 + 0.5 + 0.25*np.sin(time/100.0)

    x0 = np.concatenate((data[1,:],U[0]))
    ic = torch.unsqueeze(torch.tensor(x0),0).float()
    fx_int = integrators.EulerDAE(model_ode,algebra=model_algebra,h=1.0)

    sol = rollout(nsteps, fx_int, ic, torch.tensor(U))


    ## Extrapolate the tank profile...

    # plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    # plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    # plt.plot(time[0:-1],data[1:,0],label="Truth",linestyle="--")
    # plt.xlim([0,500])
    # plt.ylim([0,50])
    # plt.xlabel("Time")
    # plt.ylabel("Height")
    # plt.legend()
    # plt.savefig('extrap_tanks_'+str(db)+'.png')
    # plt.show()


    ax[1].plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    ax[1].plot(time,sol.detach().numpy()[:,3],label="Inflow #2")
    ax[1].plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    ax[1].plot(time[0:-1],data[1:,2],label="Data Inflow #1",linestyle="--")
    ax[1].plot(time[0:-1],data[1:,3],label="Data Inflow #2",linestyle="--")
    ax[1].set_xlim([0,500])
    ax[1].set_ylim([0,0.85])
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Volumetric Flow")
    ax[1].legend()

    MSE_vol0_extrap = Calculate_MSE(data[1:,2], sol.detach().numpy()[:-1,2])
    MSE_vol1_extrap = Calculate_MSE(data[1:,3], sol.detach().numpy()[:-1,3])

    MSE_vol_agg_extrap = Calculate_MSE( np.sum(data[1:,2:3], axis=0), np.sum(sol.detach().numpy()[:-1,2:3],axis=0))
   
    print(f" Manifold vol  extrap #{1}, MSE: {MSE_vol0_extrap}")
    print(f" Manifold vol extrap  #{2}, MSE: {MSE_vol1_extrap}")

    print(f" Manifold vol extrap  agg, MSE: {MSE_vol_agg_extrap}")
    


    fig.show()
    ax[0].annotate("a)", xy=(-0.01, 1.05), xycoords="axes fraction")
    ax[1].annotate("b)", xy=(-0.01, 1.05), xycoords="axes fraction")
    fig.savefig('extrap_HeightAreas_flows_'+str(db)+'.png')
    # ax[1].show()

    # Rollout:
    # problem.nodes[0].nsteps = 500
    # out = problem.nodes[0](input)

    # sol = out['xn'][0]

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

    # plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    # plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    # plt.plot(time[0:-1],data_2[1:,0],label="Data",linestyle="--")
    # plt.xlim([0,500])
    # plt.ylim([0,40])
    # plt.xlabel("Time")
    # plt.ylabel("Height")
    # plt.legend()
    # plt.show()
    # plt.savefig('extrap_tanks_'+str(db)+'.png', format='png', dpi=1200)


    # plt.plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    # plt.plot(time,sol.detach().numpy()[:,3],label="Inflow #2")
    # plt.plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    # plt.plot(time[0:-1],data_2[1:,2],label="Data Inflow #1",linestyle="--")
    # plt.plot(time[0:-1],data_2[1:,3],label="Data Inflow #2",linestyle="--")

    # plt.xlim([0,500])
    # plt.ylim([0,0.6])
    # plt.xlabel("Time")
    # plt.ylabel("Volumetric Flow")
    # plt.legend()
    # plt.show()
    # plt.savefig('extrap_flows_'+str(db)+'.png', format='png', dpi=1200)

# %%