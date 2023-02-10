# indivitual modules

import cem_maddpg_buffer
import cem_maddpg_models
import cem_maddpg_trainer
import cem_maddpg_runner
import cem_maddpg_env
import cem_maddpg_utils
import cem_maddpg
from cem_maddpg_config import get_config

# installed packages

import numpy as np
import pdb
import torch
import torch.nn as nn
import copy


class MaddpgCem:
    
    def __init__(self):
        
        # get args
        
        parser = get_config()
        
        args = parser.parse_args()
        
        self.maddpg_runner = copy.deepcopy(cem_maddpg_runner.Runner(args))
        
        
        # # evolutionary alogorithm runner
        
        network_model = copy.deepcopy(self.maddpg_runner.trainer.actor)
        
        self.cem_runner = cem_maddpg.CEM(network_model = network_model, 
                                        rl_algorithm_runner = self.maddpg_runner, parameters_lower_bound = args.parameters_lower_bound,  
                                        parameters_upper_bound = args.parameters_upper_bound, num_genomes = args.pop_size,
                                        initial_sigma = args.initial_sigma,  symmetry = args.symmetry, 
                                        max_iteration = args.max_iteration, initial_epsilon = args.initial_epsilon,
                                        epsilon_limit = args.epsilon_limit, tau = args.tau_cem,
                                        num_parents_prob = args.num_parents_prob)
    
    
    def MainLoop(self, number_migration):
        
        for iteration in range(number_migration):
            
            # run rl part
            
            print("start of rl part")
            
            self.maddpg_runner.total_step, self.maddpg_runner.train_step = 0, 0
            
            self.maddpg_runner.Run()

            #migration
            
            self.cem_runner.InsertSolutions(self.maddpg_runner.num_migration) 
            
            # # ea part 
            
            print("start of ea part")
            
            self.cem_runner.MainLoop()
                
if __name__ == "__main__":
                        
    MCEM = MaddpgCem()

    MCEM.MainLoop(10)