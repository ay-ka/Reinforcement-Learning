
import numpy as np
import gym
from core import utils
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
import pressureplate
import robosuite
import rware
from robosuite.controllers import load_controller_config
controller_config = load_controller_config(default_controller="OSC_POSE")
from robosuite.environments.manipulation.lift import Lift
import pdb


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, args, mode="very_easy"):
        """
        A base template for all environment wrappers.
        """
        
        self.env_name = env_name
        self.num_agents = args.agents
        
        
        if env_name == "RWARE":  
            if mode == "very_easy":
                self.env = gym.make("rware-tiny-" + str(args.agents) + "ag-v1")
            if mode == "easy":
                self.env = gym.make("rware-small-" + str(args.agents) + "ag-v1")
            if mode == "medium":
                self.env = gym.make("rware-medium-" + str(args.agents) + "ag-v1")
            if mode == "hard":
                self.env = gym.make("rware-large-" + str(args.agents) + "ag-v1")       
        elif env_name == "MPE":
            self.env = simple_spread_v2.env(N = self.num_agents, max_cycles=25, local_ratio = 0, continuous_actions = True)

        elif env_name == "PressurePlate":
            self.env = gym.make('pressureplate-linear-' + str(self.num_agents) + 'p-v0')

        elif env_name == "RobotManipulator":
            self.env = robosuite.make(
                args.task_name,
                robots=args.robots,             # load a Sawyer robot and a Panda robot
                gripper_types=args.grippers,                # use default grippers per robot arm
                controller_configs=controller_config,   # each arm is controlled using OSC
                env_configuration=args.robots_position, # (two-arm envs only) arms face each other
                has_renderer=False,                     # no on-screen rendering
                has_offscreen_renderer=False,           # no off-screen rendering
                control_freq=args.control_freq,                        # 20 hz control for applied actions
                horizon=args.episode_limit,                            # each episode terminates after 200 steps
                use_object_obs=True,                    # provide object observations to agent
                use_camera_obs=False,                   # don't provide image observations to agent
                reward_shaping=args.reward_shaping,                    # use a dense reward signal for learning
            )
        
        
        self.is_discrete = self.is_discrete(self.env) # need modification


        #State and Action Parameters
        if isinstance(self.env, pettingzoowrapper):
            self.obs_dim = self.env.observation_space("agent_0").shape[0]
            if self.is_discrete:
                self.action_dim = self.env.action_space("agent_0").n
            else: #need modification
                self.action_dim = self.env.action_space("agent_0").shape[0]
                self.action_low = float(self.env.action_space("agent_0").low[0])
                self.action_high = float(self.env.action_space("agent_0").high[0])
            self.test_size = 10
        elif isinstance(self.env, Lift):
            self.obs_dim = np.concatenate((self.env.observation_spec()['robot0_proprio-state'].reshape(1, -1),
                                        self.env.observation_spec()['object-state'].reshape(1, -1)), axis = 1).shape[1]
            if self.is_discrete:
                self.action_dim = self.env.action_space[0].n
            else: #need modification
                self.action_dim = self.env.action_dim
                self.action_low = float(self.env.action_spec[0][0])
                self.action_high = float(self.env.action_spec[1][0])
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
        obs_list = []
        if isinstance(self.env, pettingzoowrapper):
        
            self.env.reset()
            obs = []
            self.env.agent_selection = "agent_0"
            for agent in self.env.agents:
                self.env.agent_selection = agent
                observation, _, _, _ = self.env.last()
                obs.append(observation)
            self.env.agent_selection = "agent_0"
        elif isinstance(self.env, Lift):
            obs_dict = self.env.reset()
            for robot_index in range(len(self.env.robots)):
                obs = np.concatenate((obs_dict['robot' + str(robot_index) + '_proprio-state'].reshape(1, -1),
                            obs_dict['object-state'].reshape(1, -1)), axis = 1)
                obs_list.append(obs)
            obs = np.concatenate(obs_list)
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
                self.env.step(action[agent_index]) # (1 x num_agents X 1)
            for agent_index, agent in enumerate(self.env.agents): 
                self.env.agent_selection = agent
                next_observation, reward, done, _ = self.env.last()
                next_obs.append(next_observation)
                rewards.append([reward])
                dones.append(done)           
            reward = np.sum(rewards)
            rewards = [[reward]] * len(self.env.agents) # num agent
        elif isinstance(self.env, Lift):
            next_obs, rewards, dones, info = self.env.step(np.squeeze(action.reshape(1, -1)))   
            next_obs_list = []
            for robot_index in range(len(self.env.robots)):
                next_obs = np.concatenate((next_obs['robot' + str(robot_index) + '_proprio-state'].reshape(1, -1),
                            next_obs['object-state'].reshape(1, -1)), axis = 1)
                next_obs_list.append(next_obs)
                next_obs = np.concatenate(next_obs_list)
            dones = [dones] * len(self.env.robots)
            rewards = [[rewards]] * len(self.env.robots)
        else:
            next_obs, rewards, dones, info = self.env.step(action)
            rewards = utils.ToNumpy_(rewards).reshape(-1, 1).tolist()


        done_env = np.all(dones)
        reward = utils.Unsqueeze(rewards, dim = 0)
        done = utils.Unsqueeze(utils.Unsqueeze(dones, dim = 1), dim = 0)
        next_obs = utils.Unsqueeze(next_obs, dim = 0)
        done_env = done_env.reshape(1,1,1)
        #pdb.set_trace()
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
        elif isinstance(env, Lift):
            return False
        else:
            try:
                k = env.action_space[0].n
                return True
            except:
                return False


