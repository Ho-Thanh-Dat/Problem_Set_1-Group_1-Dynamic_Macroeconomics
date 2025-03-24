"""

run_vfi_dgm.py
--------------
This code solves the stochastic growth model using value function iteration.

"""

#%% Import from Python and set project directory
#import os
#os.chdir("/Users/datho/Downloads/Stochastic Growth with Labor-3/Python")
#main = os.getcwd()
#figout = main+"\\output\\figures"

#%% Import from folder
from model import planner
from solve import plan_allocations
from simulate import grow_economy
from my_graph import track_growth

#%% Stochastic Growth Model.
benevolent_dictator = planner()

# Set the parameters, state space, and utility function.
#benevolent_dictator.setup(main=main,figout=figout,beta = 0.96,sigma=2.00) 
benevolent_dictator.setup(beta = 0.96,sigma=2.00) 

# Solve the model.
sol = plan_allocations(benevolent_dictator) # Obtain the policy functions for capital.

# Simulate the model.
sim = grow_economy(benevolent_dictator, sol) # forward in time.

# Graphs.
track_growth(benevolent_dictator.par, sol, sim) # Plot policy functions and simulations.
