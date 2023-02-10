
from re import I
from core import utils as utils
import numpy as np
import torch
from collections import namedtuple
from einops import rearrange
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
import pdb
np.random.seed(10)




# Rollout evaluate an agent in a complete game
#@torch.no_grad()
def rollout_worker(id = 0, type = "test", store_data = True, model_bucket = None, env_constructor = None, learner = None, args = None):



    env = env_constructor.make_env(agents = learner.args.agents)


    ###LOOP###
    identifier = id

    #store data place
    transition = namedtuple("Transition", field_names=["obs", "state", "action", "next_obs", "next_state", "done",
                                                       "reward", "logit", "action_onehot", "intrinsic_reward"])
    transition = transition(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Get the requisite network
    if type == "test":
        net = learner.actor.cpu()
    else:
        net = model_bucket[identifier]

    learner.GetHiddenStates(num_episode_batch = 1)
    fitness = 0.0
    total_frame = 0 
    store_frame = None
    eps_terminate = False
    obs = env.reset()
    obs_dim = obs.shape[-1]; action_dim = learner.num_actions; num_agent = learner.num_agent
    rollout_trajectory = []
    transitions = []
    while True:  # unless done
   

####################################################################
        #get action
        action, action_onehot, logit = net.clean_action(utils.Unsqueeze(obs, dim = 0), learner, collecting_data = True, args = args)

        #convert action to 2d in input not in total (in total 3d)
        next_obs, reward, done, done_env = env.step(utils.Squeeze(action, dim = 0))  
            
#########################################################################
        
        state = rearrange(obs, "d0 d1 d2 -> d0 (d1 d2)")
        state = utils.Unsqueeze(state, dim = 0)
        next_state = rearrange(next_obs, "d0 d1 d2 -> d0 (d1 d2)")
        next_state = utils.Unsqueeze(next_state, dim = 0)
    

        # If storing transitions
        if store_data: 
            
            transition = transition._replace(obs = obs, next_obs = next_obs, state = state,
                                        next_state = next_state, action = action, action_onehot = action_onehot, 
                                        reward = reward, done = done, logit = logit, intrinsic_reward = np.zeros([1, learner.args.agents, obs.shape[-1]]))
            
            transitions.append(transition)
            
        
        fitness += np.sum(reward)    
        obs = next_obs
        total_frame += 1

        # DONE FLAG IS Received
        if done_env:
            env.env.close()
            for index in range(total_frame, learner.args.episode_limit + 1):
                transition = transition._replace(obs = np.zeros([1, num_agent, obs_dim]), action_onehot = np.zeros([1, num_agent, action_dim]),
                                               reward = np.zeros([1, num_agent, 1]), done = np.ones([1, num_agent, 1]), 
                                               next_obs = np.zeros([1, num_agent, obs_dim]), action = np.zeros([1, num_agent, 1])
                                               ,logit = np.zeros([1, num_agent, action_dim]), intrinsic_reward = np.zeros([1, learner.args.agents, obs.shape[-1]]),
                                               state = np.zeros([1, 1, num_agent * obs_dim]), next_state = np.zeros([1, 1, num_agent * obs_dim]))
                transitions.append(transition)
            eps_terminate = True
            store_frame = total_frame
            break
        
        if total_frame >= learner.args.episode_limit:
            env.env.close()
            break
        
    return identifier, fitness, total_frame, transitions, False, store_frame
