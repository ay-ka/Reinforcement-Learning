# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg

from __future__ import print_function
from cv2 import add

import gym
import gym.wrappers
import rware
from collections import namedtuple
from collections import OrderedDict

import matplotlib.pyplot as plt
from einops import rearrange
from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical
from src.QMIX_NEAT.envs_repo.constructor import EnvConstructor
from src.QMIX_NEAT.core.buffer import Buffer
import multiprocessing
import src.QMIX_NEAT.neat as neat
from src.QMIX_NEAT.neat.six_util import iteritems, itervalues
import numpy as np
import os
import pickle
import random
import pdb
import time
import torch

# import visualize

NUM_CORES = 1




class RwareGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def configure_new(self, config):
        super().configure_new(config)

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)

    def mutate(self, config):
        super().mutate(config)

    def distance(self, other, config):
        dist = super().distance(other, config)  
        return dist

    def __str__(self):
        return "salam_genome"



class PooledErrorCompute(object):
    def __init__(self, buffer, add_limit = 5, episode_limit = 500, args = None):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200
        
        self.episode_limit = episode_limit
        
        self.replay_buffer = buffer
        
        self.add_limit = add_limit
        env_constructor = EnvConstructor("RWARE", args.agents)
        self.env = env_constructor.make_env()
        
        self.best_score = -float('inf')
        
        self.episode_terminates = {}

        self.episode_score = []
        self.episode_length = []
        
        self.episode_terminates = {}
        

    def simulate(self, net):
        epsilon = .2
        transition = namedtuple("Transition", field_names=["obs",  "done", "reward",  "action_onehot"])
        transition = transition(0, 0, 0, 0)
        fitness = 0
        total_frame = 0; store_frame = None
        eps_terminate = False
        observation = self.env.reset()
        transitions = []
        obs_dim = observation.shape[-1]
        num_action = self.env.action_dim
        num_agent = observation.shape[1]
        last_action_onehot = np.zeros([num_agent, num_action])
        specify_agents = np.eye(num_agent)
        while True:
            obs = np.concatenate((observation[0], specify_agents, last_action_onehot), axis = 1)
            q_values = []
            for agent_obs in obs:
                q_value = net.activate(agent_obs)
                q_values.append(torch.tensor([q_value]))
            q_values = torch.cat(q_values, dim = 0)
            
            
            random_numbers = torch.rand([q_values.shape[0], 1])
            flag = random_numbers < epsilon
            probabilities = torch.tensor( [1/num_action] * num_action ).repeat(q_values.shape[0], 1)
            dist = OneHotCategorical(probs = probabilities )
            radnom_action_onehot = dist.sample()
            #max action
            action = torch.argmax(q_values, dim = 1)
            max_action_onehot = torch.zeros(q_values.shape[0], num_action)
            for index, selected in enumerate(action):
                max_action_onehot[index][selected] = 1
            actions_onehot = (1 * flag) * radnom_action_onehot + (1 - 1 * (flag)) * max_action_onehot
            actions = torch.argmax(actions_onehot, dim = 1)
        
            # convert to numpy
            action = np.array(actions)
            action_onehot = np.array(actions_onehot)
            # modify shapes
            action = np.expand_dims(np.expand_dims(action, axis = 1), axis =0)
            action_onehot = np.expand_dims(action_onehot, axis = 0)
            
            

            next_observation, reward, done, done_env = self.env.step(np.squeeze(action, axis = 0))
            
            state = rearrange(observation, "d0 d1 d2 -> d0 (d1 d2)")
            state = np.expand_dims(state, axis = 0)
            next_state = rearrange(next_observation, "d0 d1 d2 -> d0 (d1 d2)")
            next_state = np.expand_dims(next_state, axis = 0)
            
            transition = transition._replace(obs =observation, action_onehot = action_onehot, reward = reward, done = done)
            transitions.append(transition)  
            fitness += np.sum(reward)
            observation = next_observation
            total_frame += 1
            
            last_action_onehot = action_onehot[0]

            if done_env:
                self.env.env.close()
                for index in range(total_frame, self.episode_limit + 1):
                    transition = transition._replace(obs = np.zeros([1, num_agent, obs_dim]), action_onehot = np.zeros([1, num_agent, num_action]),
                                                    reward = np.zeros([1, num_agent, 1]), done = np.ones([1, num_agent, 1]))
                    transitions.append(transition)
                eps_terminate = True
                store_frame = total_frame
                break
            
            if total_frame > self.episode_limit: #--> > instead of >=
                self.env.env.close()
                break
            
        if eps_terminate:
            self.eps_terminate[self.replay_buffer.store_index] = store_frame
           
        self.best_score = max(self.best_score, fitness)
         
        if fitness >= self.best_score - self.add_limit:
            self.replay_buffer.add(transitions)

        self.episode_score.append(fitness)
        self.episode_length.append(total_frame)

        self.test_episodes.append((reward, transitions))
        
        return fitness
        
        

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        if self.pool is None:
            for genome, net in nets:
                fitness = self.simulate(net)
                genome.fitness = fitness
        else:
            jobs = []

        
        
class NEAT_EVO:
    
    def __init__(self, buffer, num_run = 5, rl_network = None, args = None):
        
        self.num_run = num_run
        
        self.ec = PooledErrorCompute(buffer, args = args)           
        
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        self.config = neat.Config(RwareGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path, self.ec.env, args)

        self.pop = neat.Population(self.config)
        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 25 generations or 900 seconds.
        self.pop.add_reporter(neat.Checkpointer(25, 900))
        
        self.rl_genome = rl_network
        
        self.add_rl_genome(rl_network, len(self.pop.population))
        
        
        
        
    def Run(self):
        
        
        gen_best = self.pop.run(self.ec.evaluate_genomes, self.num_run)

        return self.ec.replay_buffer
    
    
    def add_rl_genome(self, rl_network, genome_idx = None):
        
        self.ec.evaluate_genomes(list(iteritems(self.pop.population)), self.config)
        
        #sort pop
        if genome_idx is None:
            fitness = []
            for genome_idx, genome in self.pop.population.items():
                fitness.append((genome.key, genome.fitness))
            try:
                min_genome_idx, min_genome = min(fitness, key = lambda x : x[1])
            except:
                pdb.set_trace()
                pass
            genome_idx = min_genome_idx
        else:
            pass
        
        self.rl_genome = self.pop.reproduction.create_rl_genome(self.config.genome_type,
                                               self.config.genome_config,
                                               genome_idx)
        
        #add bias 
        nodes = self.rl_genome.nodes 
        bias_dict = OrderedDict({"biases" : []})
        for key, value in rl_network.state_dict().items():
            if len(rl_network.state_dict()[key].shape) == 1:
                bias_dict["biases"].extend(value.tolist())
        done = list(map(self.map_bais, zip(list(bias_dict.values())[0], nodes)))
        
        #add weights and enabled
        connections = self.rl_genome.connections
        weight_dict = OrderedDict({"weights" : []})
        for key, value in rl_network.state_dict().items():
            if len(rl_network.state_dict()[key].shape) == 2:
                transposed = torch.transpose(value, 0, 1)
                reshaped_list = transposed.reshape(1, -1).tolist()[0]
                weight_dict["weights"].extend(reshaped_list)
        done = list(map(self.map_connections, zip(list(weight_dict.values())[0], connections)))
        
        self.pop.population[genome_idx] = self.rl_genome
        
        if self.pop.generation == 0:
            gen = self.pop.generation
        else:
            gen = self.pop.generation - 1
        self.pop.species.speciate(self.config, self.pop.population, generation = gen)
        
    def map_bais(self, x):
        self.rl_genome.nodes[x[1]].bias = x[0]
        
    def map_connections(self, x):
        self.rl_genome.connections[x[1]].weight = x[0]
        self.rl_genome.connections[x[1]].enabled = True
            
                


if __name__ == '__main__':
    pdb.set_trace()
    run()