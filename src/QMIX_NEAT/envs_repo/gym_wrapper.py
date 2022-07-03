
import numpy as np
import gym
from src.QMIX_NEAT.core import utils
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
import pressureplate
import pdb
import rware


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, num_agents, mode):
        """
        A base template for all environment wrappers.
        """
        self.env_name = env_name
        self.num_agents = num_agents
        
        
        if env_name == "RWARE":  
            if mode == "very_easy":
                self.env = gym.make("rware-tiny-" + str(num_agents) + "ag-v1")
            if mode == "easy":
                self.env = gym.make("rware-small-" + str(num_agents) + "ag-v1")
            if mode == "medium":
                self.env = gym.make("rware-medium-" + str(num_agents) + "ag-v1")
            if mode == "hard":
                self.env = gym.make("rware-large-" + str(num_agents) + "ag-v1")       
        elif env_name == "MPE":
            self.env = simple_spread_v2.env(N = num_agents, max_cycles=25, local_ratio = 0)

        elif env_name == "PressurePlate":
            self.env = gym.make('pressureplate-linear-' + str(self.num_agents) + 'p-v0')
        
        
        self.is_discrete = self.is_discrete(self.env) # need modification

        #State and Action Parameters
        if isinstance(self.env, pettingzoowrapper):
            self.obs_dim = self.env.observation_space("agent_0").shape[0]
            if self.is_discrete:
                self.action_dim = self.env.action_space("agent_0").n
            else: #need modification
                self.action_dim = self.env.action_space("agent_0").n
                self.action_low = float(self.env.action_space("agent_0").low[0])
                self.action_high = float(self.env.action_space("agent_0").high[0])
            self.test_size = 10
        else:
            self.obs_dim = self.env.observation_space[0].shape[0]
            if self.is_discrete:
                self.action_dim = self.env.action_space[0].n
            else: #need modification
                self.action_dim = self.env.action_space[0].shape[0]
                self.action_low = float(self.env.action_space[0].low[0])
                self.action_high = float(self.env.action_space[0].high[0])
            self.test_size = 10

    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        if isinstance(self.env, pettingzoowrapper):
        
            self.env.reset()
            obs = []
            self.env.agent_selection = "agent_0"
            for agent in self.env.agents:
                self.env.agent_selection = agent
                observation, _, _, _ = self.env.last()
                obs.append(observation)
            self.env.agent_selection = "agent_0"
        else:
            obs = self.env.reset()

        #convert list of array to numpy.array 
        obs = np.array(obs)

        # convert to batch like shape
        obs = utils.Unsqueeze(obs, dim = 0)

        return obs

    def step(self, action): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                action (ndarray): action to take in the env (num_agent x 1)

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                done (bool): Simulation done?
                info (None): Template from OpenAi gym (doesnt have anything)
        """ 
        if isinstance(self.env, pettingzoowrapper):
            
            next_obs, rewards, dones = [], [], []
            for agent_index, agent in enumerate(self.env.agents):
                self.env.agent_selection = agent
                self.env.step(action[agent_index][0]) # (1 x num_agents X 1)
            for agent_index, agent in enumerate(self.env.agents): 
                self.env.agent_selection = agent
                next_observation, reward, done, _ = self.env.last()
                next_obs.append(next_observation)
                rewards.append([reward])
                dones.append(done)           
            reward = np.sum(rewards)
            rewards = [[reward]] * self.num_agents # num agent
                
        else:
            next_obs, rewards, dones, info = self.env.step(action)
            rewards = utils.ToNumpy_(rewards).reshape(-1, 1).tolist()
            
        done_env = np.all(dones)
        reward = utils.Unsqueeze(rewards, dim = 0)
        done = utils.Unsqueeze(utils.Unsqueeze(dones, dim = 1), dim = 0)
        next_obs = utils.Unsqueeze(next_obs, dim = 0)
        done_env = done_env.reshape(1,1,1)
        return next_obs, reward, done, done_env

    def render(self):
        self.env.render()

    def is_discrete(self, env):
        
        if isinstance(env, pettingzoowrapper):
            try:
                k = env.action_space("agent_0").n
                return True
            except:
                return False   
        else:
            try:
                k = env.action_space[0].n
                return True
            except:
                return False


