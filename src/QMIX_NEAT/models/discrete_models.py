import torch, random, copy
import torch.nn as nn
from torch.distributions import  Normal, RelaxedOneHotCategorical, Categorical
import torch.nn.functional as F
from src.QMIX_NEAT.core import utils
from einops import rearrange
import pdb
from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical
   




class QMIX_Critic(nn.Module):
    
    
    def __init__(self, before_rnn_layers, after_rnn_layers):
        
        super(QMIX_Critic, self).__init__()
        
        ## we use nn.Modulelist to build structure of layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hiddens = nn.ModuleList()
        self.before_rnn_layers = before_rnn_layers
        self.after_rnn_layers = after_rnn_layers
        self.fc_layers = before_rnn_layers + after_rnn_layers    
            
        #create layers
        

        for input_dim, output_dim in zip(self.fc_layers, self.fc_layers[1:]):
            self.hiddens.append(nn.Linear(input_dim, output_dim).to(device=self.device))

            
        
        self.hidden_layer_act = nn.ReLU().to(device=self.device)
        
        



            
    def forward(self, activations, hidden_state = None):
        
        if hidden_state == None:
              
            # forward fully_connected layers
            max_num_layers = len(self.fc_layers) - 1
            for layer_index, targeted_layer in zip(range(max_num_layers), self.hiddens):
                if layer_index < len(self.hiddens) - 1:
                    activations = targeted_layer(activations)
                    #activations = self.BatchNorms[layer_index](activations)
                    activations = self.hidden_layer_act(activations)
                else:
                    activations = targeted_layer(activations)
                    q_values = activations
            new_hidden_state = None   
        else:
            # before rnn forward
            index_of_before_rnn = len(self.before_rnn_layers) - 1
            for layer_index, targeted_layer in zip(range(index_of_before_rnn), self.hiddens):
                activations = targeted_layer(activations)
                #activations = self.BatchNorms[layer_index](activations)
                activations = self.hidden_layer_act(activations)
            # rnn forward       
            index_of_rnn_layer = len(self.before_rnn_layers) - 1
            new_hidden_state = self.hiddens[index_of_rnn_layer](activations, hidden_state)
            activations  = new_hidden_state
            # after rnn forward
            start_index = len(self.before_rnn_layers) 
            end_index = len(self.hiddens)
            for layer_index, targeted_layer in zip(range(start_index, end_index), self.hiddens[start_index : end_index]):
                if layer_index < len(self.hiddens) - 1:
                    activations = targeted_layer(activations)
                    #activations = self.BatchNorms[layer_index - 1](activations) # why -1? rnn layer dosen't have batchnorm
                                                                                # so num batchnorm layer is main layer -1
                    activations = self.hidden_layer_act(activations)
                else:
                    activations = targeted_layer(activations) 
                    q_values = activations
                    
        return q_values, new_hidden_state   
    
    
    def clean_action(self, obs_batch, last_action_onehot_batch, learner, epsilon, collecting_data = True, use_target = False):
            
        num_episode_batch, episode_limit, num_action = obs_batch.shape[0], obs_batch.shape[1], last_action_onehot_batch.shape[-1]
        
        specify_agent = torch.eye(learner.num_agent).expand(num_episode_batch, episode_limit, learner.num_agent, -1)
        obs_batch = utils.ToTensor_(obs_batch)
        last_action_onehot_batch = utils.ToTensor_(last_action_onehot_batch)
        inputs = torch.cat([obs_batch, last_action_onehot_batch, specify_agent], dim = 3)

        
        input_minibatch = rearrange(inputs[:, 0], "d0 d1 d2 -> (d0 d1) d2")
        q_values, hidden_states = self.forward(input_minibatch, None)
        
        
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
        action = utils.ToNumpy_(actions)
        action_onehot = utils.ToNumpy_(actions_onehot)
        # modify shapes
        action = utils.Unsqueeze(utils.Unsqueeze(action, dim = 1), dim =0)
        action_onehot = utils.Unsqueeze(action_onehot, dim = 0)
        
        return action, action_onehot, epsilon 
    
    
    
class QMIXNet(nn.Module):
    
    
    def __init__(self, hypernet_hidden_dim : int, qmix_net_hidden_dim : int, state_shape : int, num_agents : int):
        
        super(QMIXNet, self).__init__()
        
        torch.manual_seed(0)
        
        self.qmix_net_hidden_dim = qmix_net_hidden_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim
        self.state_shape = state_shape
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create hypernetwork
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_shape, self.hypernet_hidden_dim).to(device=self.device),
                                      nn.ReLU().to(device=self.device),
                                      nn.Linear(self.hypernet_hidden_dim, self.num_agents  * self.qmix_net_hidden_dim).to(device=self.device))
            

        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, self.hypernet_hidden_dim).to(device=self.device),
                                          nn.ReLU().to(device=self.device),
                                          nn.Linear(self.hypernet_hidden_dim, self.qmix_net_hidden_dim).to(device=self.device))
        

        self.hyper_b1 = nn.Linear(state_shape, self.qmix_net_hidden_dim).to(device=self.device)
        

        self.hyper_b2 =nn.Sequential(nn.Linear(state_shape, self.qmix_net_hidden_dim).to(device=self.device),
                                     nn.ReLU().to(device=self.device),
                                     nn.Linear(self.qmix_net_hidden_dim, 1).to(device=self.device)
                                     )
        
        
    def forward(self, q_value_batch, state_batch):
        
        """
         
        feedforward for QMIXNet

        Args:
        
            q_value_batch: selected q_values for all agents for all states of input batch --> 
                                                                              (num_episode, episode_limit, num_agent, 1)
                                                                              
            state_batch: batch of global states seen by agent for all states of input batch -->
                                                                              (num_episode, episode_limit, 1, state_shape)  

        Returns:
        
            q_total: all centralize q_values for all states of input batch --> 
                                                                              (num_episode, episode_limit, 1) 

        """
        
        batch_size = q_value_batch.shape[0]
        q_values = q_value_batch.view(-1, 1, self.num_agents)  # (episode_batch * episode_limit, 1, n_agents) 
        states = state_batch.reshape(-1, self.state_shape)  # (episode_batch * episode_limit, state_shape)
        w1 = torch.abs(self.hyper_w1(states))  # (100, 160)
        b1 = self.hyper_b1(states)  # (100, 32)
        w1 = w1.view(-1, self.num_agents, self.qmix_net_hidden_dim)  # (100, 5, 32)
        b1 = b1.view(-1, 1, self.qmix_net_hidden_dim)  # (100, 1, 32)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (100, 1, 32)
        w2 = torch.abs(self.hyper_w2(states))  # (100, 32)
        b2 = self.hyper_b2(states)  # (100, 1)
        w2 = w2.view(-1, self.qmix_net_hidden_dim, 1)  # (100, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (100, 1， 1)
        q_total = torch.bmm(hidden, w2) + b2  # (100, 1, 1)
        q_total = q_total.view(batch_size, -1, 1)  # (5, 20, 1)
        return q_total
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
       
   



