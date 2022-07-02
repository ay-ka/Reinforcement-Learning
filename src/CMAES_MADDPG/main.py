# indivitual modules

import cmaes_maddpg_buffer
import cmaes_maddpg_models
import cmaes_maddpg_trainer
import cmaes_maddpg_runner
import cmaes_maddpg_env
import cmaes_maddpg_utils
import cmaes_maddpg
from cmaes_maddpg_config import get_config

# installed packages

import numpy as np
import pdb
import torch
import torch.nn as nn
import copy


class MaddpgCmaes:
    
    def __init__(self):
        
        # get args
        
        parser = get_config()
        
        args = parser.parse_args()
        
        # # reinforcement learning runner
        
        self.maddpg_runner = copy.deepcopy(cmaes_maddpg_runner.Runner(args))
        
        
        # # evolutionary alogorithm runner
        
        network_model = copy.deepcopy(self.maddpg_runner.trainer.actor)
        
        self.cmaes_runner = cmaes_maddpg.CMAES(network_model = network_model, 
                                        rl_algorithm_runner = self.maddpg_runner, parameters_lower_bound = args.parameters_lower_bound,  
                                        parameters_upper_bound = args.parameters_upper_bound, num_genomes = args.pop_size, initial_sigma = args.initial_sigma, 
                                        symmetry = args.symmetry, max_iteration = args.max_iteration,  num_parents_prob = args.num_parents_prob,
                                        step_size = args.step_size)
    
    
    def MainLoop(self, number_migration):
        
        for iteration in range(number_migration):
            
            # run rl part
            
            print("start of rl part")
            
            self.maddpg_runner.total_step, self.maddpg_runner.train_step = 0, 0
            
            self.maddpg_runner.Run()

            #migration
            
            self.cmaes_runner.InsertSolutions(self.maddpg_runner.num_migration) 
            
            # # ea part 
            
            print("start of ea part")
            
            self.cmaes_runner.MainLoop()
            
                

                
if __name__ == "__main__":
                        
    MCMAES = MaddpgCmaes()

    MCMAES.MainLoop(2)