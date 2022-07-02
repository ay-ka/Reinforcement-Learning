import os, random, copy
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from core import utils
from core.utils import soft_update, hard_update
import wandb
import torch, pdb
from einops import rearrange
random.seed(10), np.random.seed(10), torch.manual_seed(10)
from loguru import logger

class MADDPG:
    
    def __init__(self, args, model_constructor, action_dim):
        
        super(MADDPG, self).__init__()
        
        self.args = args
        self.num_agent = args.agents
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.tau = args.tau
        self.num_actions = action_dim
        self.gamma = args.gamma
        self.requlirization = args.reqularization
        self.grad_clip = args.grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_interval = args.target_update_interval
        self.num_updates = 0
        
        self.actor, self.critic, self.actor_target, self.critic_target = model_constructor.make_model("MADDPG")
        
        self.actor.to(device = self.device), self.critic.to(device = self.device)
        self.actor_target.to(device = self.device), self.critic_target.to(device = self.device)

        # define optimizers       
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr  = args.actor_lr, amsgrad = True)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr  = args.critic_lr, amsgrad = True)
        
        #update target networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
    def update_parameters(self, data_batch):
        
        (obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, logit_batch, 
        reward_batch, done_batch, intrinsic_reward_batch) = data_batch
        
        
        self.TrainActor(obs_batch, state_batch, done_batch)
        self.TrainCritic(obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, reward_batch, done_batch)
        
        self.num_updates += 1
        
        
    
    def TrainActor(self, obs_batch, state_batch, done_batch):
    
        #calculate loss
        policy_loss = self.ActorLOSS(obs_batch, state_batch, done_batch)
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.optimizer_actor.step()
        #wandb.log({"actor_loss": policy_loss})
        
        # update target network weight
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.actor_target, self.actor, self.tau)
        if self.num_updates % 100 == 0:
            with open("ant_loggg", "a") as f: 
                logger.log("EXTRA", f"-critic loss is {policy_loss} and grad_norm is {grad_norm}") 
                f.write("at train step  :" + str(self.num_updates) + " actor parameters : \n" + "policy losss is : " + str(policy_loss) + "\n" + " grad_norm is : " + 
                        str(grad_norm) + "\n")
                

        
    def ActorLOSS(self, obs_batch, state_batch, done_batch):
        
        batch_size, episode_limit = obs_batch.shape[0], obs_batch.shape[1]
        
        #calculate actions wrt current policy
        self.GetHiddenStates(batch_size)
        action_batch, agent_action_onehot, logit_batch = self.GetActions(obs_batch, collecting_data = False)
        
        # modify actions
        agent_action_onehot = agent_action_onehot.view(batch_size, -1, 1, self.num_agent * self.num_actions).expand(-1, -1, self.num_agent, -1)
        
        new_actions = []
        for targeted_agent_index in range(self.num_agent):
            temp_action = torch.split(agent_action_onehot[:, :, targeted_agent_index, :], self.num_actions, dim=2)
            actions_agents = []
            for agent_index in range(self.num_agent):
                if targeted_agent_index == agent_index:  
                    actions_agents.append(temp_action[agent_index])
                else:
                    actions_agents.append( temp_action[agent_index].detach() )
            actions_agents = utils.TensorConcatenate_(actions_agents, dim = -1).to(device = self.device)
            new_actions.append(utils.Unsqueeze(actions_agents, dim =2, tensor = True) )
        new_actions = utils.TensorConcatenate_(new_actions, dim = 2).to(device = self.device)
        
          
        # make <actions_onehot_batch> first and second dim same as <state_batch> dims to handle concatination>
        state_batch = utils.ToTensor_(state_batch).to(device = self.device)
        state_batch = state_batch.expand(-1, -1, self.num_agent, -1).to(device = self.device)
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1).to(device = self.device)
        critic_inputs = torch.cat((state_batch, new_actions, specify_agent),  dim = 3).to(device = self.device)
            
        # deactivate gradient process in critic parameters
        #for parameter in model.parameters():
            #parameter.requires_grad_(False)
        self.critic.to(device = self.device)
        values = self.critic(critic_inputs)
        values = values.view(-1, 1).to(device = self.device)
        mask = utils.ToTensor_((1 - done_batch ).reshape(-1, 1) ).to(device = self.device)
        loss = -1 * torch.mean(values * 1) + (self.requlirization * (logit_batch.view(-1, 1) ** 2 ).mean() ).to(device = self.device)
            
        return loss
    
    
    def GetActions(self, obs_batch, collecting_data = True, use_target = False):
                
        batch_size, episode_limit = obs_batch.shape[0], obs_batch.shape[1]
        actor_inputs, _ = self.GetInputs(obs_batch)
        actions, actions_onehot, logits  = [], [], []
        
        for time_step in range(episode_limit):
            action, action_onehot, logit = self.AgentAction(use_target, time_step, actor_inputs)
            actions.append(action)
            actions_onehot.append(action_onehot)
            logits.append(logit)
                
        concatenator = utils.TensorStack(actions, actions_onehot, logits, dim = 1)
        
        if collecting_data == True: 
            actions_batch = utils.ToNumpy_(concatenator.__next__())
            actions_onehot_batch = utils.ToNumpy_(concatenator.__next__().detach())
            logit_batch = utils.ToNumpy_(concatenator.__next__().detach())  
            
        else:
            actions_batch = concatenator.__next__()
            actions_onehot_batch = concatenator.__next__()
            logit_batch = concatenator.__next__()
        
        return actions_batch, actions_onehot_batch, logit_batch
    
    
    def GetInputs(self, obs_batch, last_action_onehot = None, next_obs_batch = None, action_onehot_batch = None, train = False):
        
        batch_size, episode_limit = obs_batch.shape[0], obs_batch.shape[1]
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, self.num_agent, -1).to(device = self.device)
        obs_batch = utils.ToTensor_(obs_batch).to(device = self.device)
        inputs = torch.cat([obs_batch, specify_agent], dim = 3).to(device = self.device)
        
        if train:
            next_obs_batch = utils.ToTensor_(next_obs_batch).to(device = self.device)
            inputs_next = torch.cat([next_obs_batch, specify_agent], dim = 3).to(device = self.device)
        else:
            inputs_next = None
            
        return inputs, inputs_next
    
    def AgentAction(self, use_target, time_step, actor_inputs, gumbel_softmax = True):
        
        actor_input = copy.deepcopy(actor_inputs[:, time_step])
        batch_size = actor_inputs.shape[0]
        
        if use_target:       
            actor_input = utils.ToTensor_(actor_input).to(device = self.device)
            actor_input = rearrange(actor_input, "d0 d1 d2 -> (d0 d1) d2")
            self.actor_target.to(device = self.device)
            if self.args.rnn_hidden_dim == 0:
                logit, target_hidden_states = self.actor_target(actor_input, None)
            else:
                target_hidden_states = rearrange(self.target_hidden_states, "d0 d1 d2 -> (d0 d1) d2")
                logit, target_hidden_states = self.actor_target(actor_input, target_hidden_states)
                self.target_hidden_states = target_hidden_states.view(batch_size, self.num_agent, self.rnn_hidden_dim)
                   
        else:        
            actor_input = utils.ToTensor_(actor_input).to(device = self.device)
            actor_input = rearrange(actor_input, "d0 d1 d2 -> (d0 d1) d2")
            self.actor.to(device = self.device)
            if self.args.rnn_hidden_dim==0:
                logit, hidden_states = self.actor(actor_input, None) 
            else:
                hidden_states = rearrange(self.hidden_states, "d0 d1 d2 -> (d0 d1) d2")
                logit, hidden_states = self.actor(actor_input, hidden_states)    
                self.hidden_states = hidden_states.view(batch_size, self.num_agent, self.rnn_hidden_dim)
                
        if gumbel_softmax:
            action_onehot = F.gumbel_softmax(logits = logit, hard = True).to(device = self.device)
            action = torch.argmax(action_onehot, dim = 1).view(batch_size, self.num_agent, -1).to(device = self.device)
            logit = logit.view(batch_size, self.num_agent, -1).to(device = self.device)
            action_onehot = action_onehot.view(batch_size, self.num_agent, -1).to(device = self.device)
            if batch_size == 1:  
                action = action.detach()
                action_onehot = action_onehot.detach()
                logit = logit.detach()
        
        else: 
            pass
        
        return action, action_onehot, logit 
        
                    
    def TrainCritic(self, obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, reward_batch, done_batch):
                        
        critic_loss = self.CriticLoss(state_batch, next_state_batch, reward_batch, done_batch,
                                      action_batch, action_onehot_batch, next_obs_batch)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer_critic.step()
        #wandb.log({"critic_loss": critic_loss})
        
        
        # update target network weight
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        if self.num_updates % 100 == 0:
            with open("ant_loggg", "a") as f:  
                logger.log("EXTRA", f"-critic loss is {critic_loss} and grad_norm is {grad_norm}") 
                f.write("at train step  : " + str(self.num_updates) + "  critic parameters : \n" + "critic losss is : " + str(critic_loss) + "\n" + " grad_norm is : " + 
                        str(grad_norm) + "\n")
    
    
    def CriticLoss(self, state_batch, next_state_batch, reward_batch, done_batch, action_batch, action_onehot_batch, next_obs_batch):
        
        batch_size, episode_limit = action_batch.shape[0], action_batch.shape[1]
    
        #calculate target actions wrt target_actor_model
        self.GetHiddenStates(batch_size)
        target_actions_batch, target_actions_onehot_batch, _, = self.GetActions(next_obs_batch, use_target = True, collecting_data = True)
    
        with torch.no_grad():
            next_state_batch = utils.ToTensor_(next_state_batch).to(device = self.device)
            next_state_batch = next_state_batch.expand(-1, -1, self.num_agent, -1).to(device = self.device)
            target_actions_onehot_batch = utils.ToTensor_(target_actions_onehot_batch).to(device = self.device)
            target_actions_onehot_batch =  target_actions_onehot_batch.view(batch_size, -1, 1, self.num_agent * self.num_actions).expand(
                                                                                                                -1, -1, self.num_agent, -1).to(device = self.device)
            specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1).to(device = self.device)
            target_critic_inputs = torch.cat((next_state_batch, target_actions_onehot_batch, specify_agent),  dim = 3).to(device = self.device)

            #calculate next_state values
            self.critic_target.to(device = self.device)
            next_state_values = self.critic_target(target_critic_inputs)
            next_state_values = utils.ToNumpy_(next_state_values)
            
            # create shared rewards and standardalize
            reward_batch = self.StandardlizeReward(reward_batch)
        
            #calculate targets
            targets = self.CalculateTargets(reward_batch, next_state_values, done_batch)

            
        # calculate cirrent_state_values
        
        #prepare action code
        action_onehot_batch = utils.ToTensor_(action_onehot_batch).to(device = self.device)
        action_onehot_batch =  action_onehot_batch.view(batch_size, -1, 1, self.num_agent * self.num_actions).expand(
                                                                                          -1, -1, self.num_agent, -1).to(device = self.device)
        #prepare state code
        state_batch = utils.ToTensor_(state_batch).to(device = self.device)
        state_batch = state_batch.expand(-1, -1, self.num_agent, -1).to(device = self.device)
        
        #prepare agent code
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1).to(device = self.device)
        
        # get critic input
        current_critic_inputs = torch.cat((state_batch, action_onehot_batch.detach(), specify_agent), dim = 3).to(device = self.device)
        self.critic.to(device = self.device)
        values = self.critic(current_critic_inputs)
        values = values.reshape(-1, 1)
        targets = utils.ToTensor_(targets).to(device = self.device)
        
        #calculate loss
        mask = utils.ToTensor_( (1 - done_batch).reshape(-1, 1) ).to(device = self.device)
        loss  = torch.mean(((values - targets.detach())**2)).to(device = self.device)

        return loss
        
    def StandardlizeReward(self, reward_batch):
        
        reward_batch = utils.Squeeze(reward_batch, dim = 3)
        reward_batch = np.sum(reward_batch, axis = 2)
        reward_batch = utils.Unsqueeze(reward_batch, dim = 2)
        reward_batch = utils.Unsqueeze(reward_batch, dim = 2)
        reward_batch = np.broadcast_to(reward_batch, shape = (reward_batch.shape[0], reward_batch.shape[1], 
                                                            self.num_agent, reward_batch.shape[3]))
        reward_standardlize = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
        
        return reward_standardlize
    
    
    def CalculateTargets(self, reward_batch, next_state_values, done_batch):
        
        targets = reward_batch.reshape(-1, 1) + self.gamma * next_state_values.reshape(-1, 1) * (1 - done_batch.reshape(-1, 1))
        
        return targets 
    
    
    def GetHiddenStates(self, num_episode_batch):
                    
        self.hidden_states = torch.zeros( (num_episode_batch, self.num_agent, self.rnn_hidden_dim) ).to(device = self.device)
        self.target_hidden_states = torch.zeros( (num_episode_batch, self.num_agent, self.rnn_hidden_dim) ).to(device = self.device)
    
    



