
from re import I
from core import utils as utils
import numpy as np
import torch
from collections import namedtuple
from einops import rearrange
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
import pdb




# Rollout evaluate an agent in a complete game
#@torch.no_grad()
def rollout_worker(id = 0, type = "test", store_data = True, model_bucket = None, env_constructor = None, learner = None, args = None):



    env = env_constructor.make_env(args)
    np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    identifier = id

    #store data place
    transition = namedtuple("Transition", field_names=["obs", "state", "action", "next_obs", "next_state", "done",
                                                       "reward", "logit", "action_onehot", "intrinsic_reward"])
    transition = transition(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Get the requisite network
    if type == "test":
        net = learner.actor
    else:
        net = model_bucket[identifier]

    learner.GetHiddenStates(num_episode_batch = 1)
    fitness = 0.0
    total_frame = 0
    obs = env.reset()
    rollout_trajectory = []
    transitions = []
    while True:  # unless done


####################################################################
        #get action
        action = net.clean_action(utils.Unsqueeze(obs, dim = 0), learner, collecting_data = True, args = args)

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
                                        next_state = next_state, action = action, action_onehot = 0, 
                                        reward = reward, done = done, logit = 0, intrinsic_reward = np.zeros([1, learner.args.agents, obs.shape[-1]]))
            
            transitions.append(transition)
            
        
        fitness += np.sum(reward)    
        obs = next_obs
        total_frame += 1

        # DONE FLAG IS Received
        if done_env:
            env.env.close()
            break
        
        if total_frame >= learner.args.episode_limit:
            
            break
        
    return identifier, fitness, total_frame, transitions
