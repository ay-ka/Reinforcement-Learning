
from re import I
from src.QMIX_NEAT.core import utils as utils
import numpy as np
import torch
from collections import namedtuple
from collections import namedtuple
from einops import rearrange
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
import pdb




# Rollout evaluate an agent in a complete game
#@torch.no_grad()
def rollout_worker(id = 0, type = "test_add", store_data = None, model_bucket = None, env_constructor = None, learner = None, epsilons = None):



    env = env_constructor.make_env()
    #np.random.seed(id) ###make sure the random seeds across learners are different

    ###LOOP###
    identifier = id

    #store data place
    transition = namedtuple("Transition", field_names=["obs",  "done", "reward",  "action_onehot"])
    transition = transition(0, 0, 0, 0)

    # Get the requisite network
    if type == "test_add":
        net = learner.qmix_critic.cpu()
    else:
        net = model_bucket[identifier]


    fitness = 0.0
    total_frame = 0; store_frame = None
    eps_terminate = False
    obs = env.reset()
    obs_dim = obs.shape[-1]; action_dim = learner.num_action; num_agent = learner.num_agent
    last_action_onehot = np.zeros([1, learner.num_agent, learner.num_action])
    rollout_trajectory = []
    transitions = []
    #pdb.set_trace()
    while True:  # unless done


####################################################################
        #get action
        action, action_onehot, epsilon = net.clean_action(utils.Unsqueeze(obs, dim = 0), utils.Unsqueeze(last_action_onehot, dim = 0),
                                                          learner, epsilons.epsilon, collecting_data = True)

        #convert action to 2d in input not in total (in total 3d)
        next_obs, reward, done, done_env = env.step(utils.Squeeze(action, dim = 0))  
            
#########################################################################

        #env.render()
        
        state = rearrange(obs, "d0 d1 d2 -> d0 (d1 d2)")
        state = utils.Unsqueeze(state, dim = 0)
        next_state = rearrange(next_obs, "d0 d1 d2 -> d0 (d1 d2)")
        next_state = utils.Unsqueeze(next_state, dim = 0)
    

        # If storing transitions
        if store_data: 
            
            transition = transition._replace(obs = obs, action_onehot = action_onehot, reward = reward, done = done)
            
            transitions.append(transition)
            
        if type == 'pg':
            epsilons._replace(epsilon = epsilon)
            if epsilons.main_epsilon > epsilons.min_epsilon:
                epsilons = epsilons._replace(main_epsilon = epsilons.main_epsilon - epsilons.epsilon_decrease_rate)
            epsilons = epsilons._replace(epsilon = epsilons.main_epsilon)
            
        
        fitness += np.sum(reward)    
        obs = next_obs
        total_frame += 1
        
        last_action_onehot = action_onehot

        # DONE FLAG IS Received
        if done_env:
            pdb.set_trace()
            env.env.close()
            for index in range(total_frame, learner.args.episode_limit + 1):
                transition = transition._replace(obs = np.zeros([1, num_agent, obs_dim]), action_onehot = np.zeros([1, num_agent, action_dim]),
                                              reward = np.zeros([1, num_agent, 1]), done = np.ones([1, num_agent, 1]))
                transitions.append(transition)
            eps_terminate = True
            store_frame = total_frame
            break
        
        if total_frame > learner.args.episode_limit: #--> > instead of >=
            env.env.close()
            break
        
    return identifier, fitness, total_frame, transitions, epsilons, eps_terminate, store_frame
