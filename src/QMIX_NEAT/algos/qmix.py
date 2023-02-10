import os, random, copy
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from zmq import device
from src.QMIX_NEAT.core import utils
from src.QMIX_NEAT.core.utils import soft_update, hard_update
import torch, pdb
from einops import rearrange
from loguru import logger



class QMIX:
    

    def __init__(self, args, model_constructor, action_dim):

        
        super(QMIX, self).__init__()
        
        self.args = args
        self.num_agent = args.agents
        self.num_action = action_dim
        self.use_double_q_network = args.use_double_q_network
        self.gamma = args.gamma
        self.learning_rate = args.lr
        self.grad_clip = args.grad_clip
        self.target_update_interval = args.target_update_interval
        self.hidden_states = None
        self.target_hidden_states = None
        self.tau = args.tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_updates = 0
        
        self.qmix_critic, self.qmix_critic_target, self.qmixnet, self.qmixnet_target = model_constructor.make_model("QMIX")
        
        self.qmix_critic.to(device = self.device), self.qmix_critic_target.to(device = self.device)
        self.qmixnet.to(device = self.device), self.qmixnet_target.to(device = self.device)
        
        hard_update(self.qmix_critic_target, self.qmix_critic)
        hard_update(self.qmixnet_target, self.qmixnet)
        
        
        # update target networks
        self.update_parameters = list(self.qmix_critic.parameters()) + list(self.qmixnet.parameters())
        self.optimizer = torch.optim.Adam(self.update_parameters, lr  = self.learning_rate, amsgrad = True)
        
    def updateParameters(self, data_batch, buffer_indices, eps_terminate):
        
        self.TrainQMIX(data_batch, buffer_indices, eps_terminate)
        self.num_updates += 1
        
        
    def TrainQMIX(self, data_batch, buffer_indices, eps_terminate):

        data_batch = self.createData(data_batch)
        
        (obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, last_action_onehot_batch, 
                    reward_batch, done_batch) = data_batch
        
        # obs_batch.to(device = self.device), next_obs_batch.to(device = self.device), state_batch.to(device = self.device), next_state_batch.to(device = self.device)
        # action_batch.to(device = self.device), action_onehot_batch.to(device = self.device), last_action_onehot_batch.to(device = self.device)
        # reward_batch.to(device = self.device), done_batch.to(device = self.device), intrinsic_reward_batch.to(device = self.device)
        
        critic_loss = self.CriticLOSS(obs_batch, next_obs_batch, state_batch, next_state_batch, action_onehot_batch, last_action_onehot_batch, action_batch, reward_batch, done_batch, 
                             self.gamma, buffer_indices, eps_terminate)
        
        self.optimizer.zero_grad() 
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.update_parameters, self.grad_clip)
        self.optimizer.step()
        if self.num_updates % self.target_update_interval == 0:
            utils.soft_update(self.qmix_critic_target, self.qmix_critic, self.tau) 
            utils.soft_update(self.qmixnet_target, self.qmixnet, self.tau) 
        if self.num_updates % 100 == 0:
            logger.log("EXTRA", f"-critic loss is {critic_loss} and grad_norm is {grad_norm}")    
            with open("ant_log_2", "a") as f:    
                f.write("at train step  : " + str(self.num_updates) + "  qmix  : \n" + "critic losss is : " + str(critic_loss) + "\n" + " grad_norm is : " + 
                        str(grad_norm) + "\n")
                
    def CriticLOSS(self, obs_batch, next_obs_batch, state_batch, next_state_batch, action_onehot_batch, 
                   last_action_onehot_batch, action_batch, reward_batch, done_batch, gamma,
                   buffer_indices, eps_terminate):
        
        num_episode_batch, episode_limit, num_agents = obs_batch.shape[0], obs_batch.shape[1], obs_batch.shape[2]
        
        inputs, next_inputs = self.GetInputs(obs_batch = obs_batch, last_action_onehot_batch = last_action_onehot_batch,
                                              next_obs_batch = next_obs_batch, action_onehot_batch = action_onehot_batch,
                                              train = True)
        
        q_values, q_target_values = [], []
        for transition_index in range(episode_limit):
            input_minibatch = copy.deepcopy(inputs[:, transition_index])
            next_input_minibatch = copy.deepcopy(next_inputs[:, transition_index])
            q_value, q_target_value = self.GetQvalues(input_minibatch, next_input_minibatch, train=True)
            q_eval = q_value.view(num_episode_batch, self.num_agent, -1).to(device=self.device)
            q_target = q_target_value.view(num_episode_batch, self.num_agent, -1).to(device=self.device)
            q_values.append(q_eval)
            q_target_values.append(q_target)
            
        # reshape list of [episode_limit * (num_episode, num_agent, dim)] to (num_episode, episode_limit, num_agent, dim)
        q_values  = torch.stack(q_values, dim = 1).to(device=self.device)
        q_target_values  = torch.stack(q_target_values, dim = 1).to(device=self.device)
        
        # choose one q_value among number_action'th q_value for agents wrt action_batch sampled from buffer 
        action_batch = utils.ToTensor_(action_batch, dtype = torch.int64).to(device=self.device)
        q_values = torch.gather(q_values, dim = 3, index = action_batch).to(device=self.device)
        
        # double q learner
        if self.use_double_q_network:
            q_target_values = self.DoubleQnetwork(next_inputs =  next_inputs, q_target_values = q_target_values)
        else:
            q_target_values = utils.Unsqueeze(torch.max(q_target_values, dim = 3)[0], tensor = True, dim = 3).to(device=self.device)
        
        # calculate q_totals ansd q_target_totals
        state_batch = utils.ToTensor_(state_batch).to(device=self.device)
        self.qmixnet.to(device=self.device)
        q_totals = self.qmixnet(q_values, state_batch)
        q_totals =utils.Unsqueeze(q_totals.repeat(1, 1, num_agents), tensor = True, dim = 3).to(device=self.device)

        next_state_batch = utils.ToTensor_(next_state_batch).to(device=self.device)
        self.qmixnet_target.to(device=self.device)
        q_target_totals = self.qmixnet_target(q_target_values, next_state_batch)
        q_target_totals = utils.Unsqueeze(q_target_totals.repeat(1, 1, num_agents), tensor = True, dim = 3).to(device="cpu")
        
        
        # calculate targets
        reward_batch = self.StandardlizeReward(reward_batch, intrinsic = False) 
        #intrinsic_reward_batch = self.StandardlizeReward(intrinsic_reward_batch, intrinsic = True)
        targets = self.CalculateTargets(reward_batch, done_batch, q_target_totals, gamma)
        
        # calculate loss
        mask = torch.ones([num_episode_batch, episode_limit, num_agents, 1]).to(device = self.device)
        for buffer_id, start_frame in eps_terminate.items():
            if buffer_id in buffer_indices:
                pdb.set_trace()
                batch_index = buffer_indices.index(buffer_id)
                mask[batch_index, start_frame:] = torch.zeros(episode_limit - start_frame - 1, num_agents, 1)
        td_errors = (q_totals - targets.detach()).to(device=self.device)
        masked_td_errors = (mask * td_errors).to(device= self.device)
        loss = ((masked_td_errors**2).sum() / mask.sum()).to(device=self.device)
        #loss = torch.mean((q_totals - targets.detach()) ** 2).to(device=self.device)
        
        return loss
    
    
    
    def GetQvalues(self, input_minibatch, next_input_minibatch = None, train = False):
        
        
        batch_size = input_minibatch.shape[0]
        input_minibatch = rearrange(input_minibatch, "d0 d1 d2 -> (d0 d1) d2")
        self.qmix_critic.to(device=self.device)
        q_values, hidden_states = self.qmix_critic(input_minibatch, None)
        if train:
            next_input_minibatch = rearrange(next_input_minibatch, "d0 d1 d2 -> (d0 d1) d2")
            self.qmix_critic_target.to(device=self.device)
            q_target_values, target_hidden_states = self.qmix_critic_target(next_input_minibatch, None)
        else:
            q_target_values = None
        
        return q_values, q_target_values
    
    def GetInputs(self, obs_batch, last_action_onehot_batch, next_obs_batch = None,
                      action_onehot_batch = None, train = False):
        
        num_agents, num_episode_batch, episode_limit = obs_batch.shape[2], obs_batch.shape[0], obs_batch.shape[1]
        specify_agent = torch.eye(num_agents).expand(num_episode_batch, episode_limit, num_agents, -1).to(device=self.device)
        obs_batch = utils.ToTensor_(obs_batch).to(device=self.device)
        last_action_onehot_batch = utils.ToTensor_(last_action_onehot_batch).to(device=self.device)
        inputs = torch.cat([obs_batch, last_action_onehot_batch, specify_agent], dim = 3).to(device=self.device)
        
        if train:
            next_obs_batch = utils.ToTensor_(next_obs_batch).to(device=self.device)
            action_onehot_batch = utils.ToTensor_(action_onehot_batch).to(device=self.device)
            inputs_next = torch.cat([next_obs_batch, action_onehot_batch, specify_agent], dim = 3).to(device=self.device)
        else:
            inputs_next = None
        
        return inputs, inputs_next
        
    def DoubleQnetwork(self, next_inputs, q_target_values):

        num_episode_batch = next_inputs.shape[0]
        episode_limit = next_inputs.shape[1]
        self.GetHiddenStates(num_episode_batch)
        q_values = []
        
        for transition_index in range(episode_limit):
            mini_next_input_batch = next_inputs[:, transition_index]
            q_val, _ = self.GetQvalues(mini_next_input_batch)
            q_val = q_val.view(num_episode_batch, self.num_agent, -1).to(device=self.device)
            q_values.append(q_val)
            
        q_values = torch.stack(q_values, dim = 1).to(device=self.device)
        q_double_network_actions = utils.Unsqueeze(torch.max(q_values, dim = 3)[1], tensor = True, dim = 3).to(device=self.device)
        selected_q_target_values = torch.gather(q_target_values, dim = 3, index = q_double_network_actions).to(device=self.device)
        
        return selected_q_target_values
    
    def StandardlizeReward(self, reward_batch, intrinsic):
        
        if intrinsic:
            reward_standardlize = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
        else:
            reward_batch = utils.Squeeze(reward_batch, dim = 3)
            reward_batch = np.sum(reward_batch, axis = 2)
            reward_batch = utils.Unsqueeze(reward_batch, dim = 2)
            reward_batch = utils.Unsqueeze(reward_batch, dim = 2)
            reward_batch = np.broadcast_to(reward_batch, shape = (reward_batch.shape[0], reward_batch.shape[1], 
                                                                self.num_agent, reward_batch.shape[3]))
            reward_standardlize = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
        
        return reward_standardlize
    
    
    def CalculateTargets(self, reward_batch, done_batch, q_target_totals, gamma : float = 0.99):
        
        q_target_totals.to(device="cpu")
        q_target_totals = utils.ToNumpy_(q_target_totals.detach())
        targets = (reward_batch) + self.gamma * (q_target_totals) * (1 - done_batch)
        targets = utils.ToTensor_(targets).to(device=self.device)
        return targets


    def createData(self, data_batch):

        obs_batch, action_onehot_batch, reward_batch, done_batch = data_batch

        reward_batch = reward_batch[:, 0:obs_batch.shape[1] - 1, :, :]

        done_batch = done_batch[:, 0:obs_batch.shape[1] - 1, :, :]

        action_onehot_batch = action_onehot_batch[:, 0:obs_batch.shape[1] - 1, :, :]

        states, next_states, next_obses, actions, last_action_onehots = [], [], [], [], []

        state_batch, next_state_batch, next_obs_batch, action_batch, last_action_onehot_batch = [], [], [], [], []

        for eps_index in range(reward_batch.shape[0]):

            states, next_states, next_obses, actions, last_action_onehots = [], [], [], [], []

            for transition_index in range(reward_batch.shape[1]):

                next_obses.append(utils.Unsqueeze(obs_batch[eps_index][transition_index + 1], dim=0))

                state = rearrange(utils.Unsqueeze(obs_batch[eps_index][transition_index], dim = 0), "d0 d1 d2 -> d0 (d1 d2)")
                state = utils.Unsqueeze(state, dim = 0)
                states.append(state)

                next_state = rearrange(next_obses[transition_index], "d0 d1 d2 -> d0 (d1 d2)")
                next_state = utils.Unsqueeze(next_state, dim = 0)
                next_states.append(next_state)

                actions.append(np.expand_dims(np.expand_dims(np.argmax(action_onehot_batch[eps_index][transition_index], axis = 1), axis=1), axis=0))

                if transition_index == 0:

                    last_action_onehots.append(np.zeros([1, self.num_agent, self.num_action]))

                else:

                    last_action_onehots.append(np.expand_dims(action_onehot_batch[eps_index][transition_index - 1], axis =0))

            concatenateor = utils.NumpyConcatenate(states, next_states, actions, last_action_onehots, next_obses) 

            state_batch.append([concatenateor.__next__()]) 
            next_state_batch.append([concatenateor.__next__()]) 
            action_batch.append([concatenateor.__next__()])
            last_action_onehot_batch.append([concatenateor.__next__()])
            next_obs_batch.append([concatenateor.__next__()])


        concatenateor = utils.NumpyConcatenate(state_batch, next_state_batch, action_batch, last_action_onehot_batch, next_obs_batch) 

        output_batch = []
        
        while True:
            
            try:
            
                output_batch.append( concatenateor.__next__() )
                
            except:
                
                break

        state_batch, next_state_batch, action_batch, last_action_onehot_batch, next_obs_batch = output_batch

        obs_batch = obs_batch[:, 0:obs_batch.shape[1] - 1, :, :]

        return (obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, last_action_onehot_batch, reward_batch, done_batch)

