import torch, random, copy
import torch.nn as nn
from torch.distributions import  Normal, RelaxedOneHotCategorical, Categorical
import torch.nn.functional as F
from core import utils
from einops import rearrange
import pdb

class MADDPG_Actor(nn.Module):
    
    
    def __init__(self, before_rnn_layers, after_rnn_layers, rnn_hidden_dim = 0, dropout_p = 0):
        
        """
         
        Actor network 

        Args:
        
            before_rnn_layers: number of nurons in each hidden layer for <fc> network before connecting to <rnn> (list)
            
            after_rnn_layers: number of nurons in each hidden layer for fc network after <rnn> network  (list)
            
            rnn_hidden_dim: number of hidden nurons in <rnn> network
            
            dropout_p: dropout probability

        Returns:
        
            no return -->  no return --> building layers structure and initializing other methods used in network 
                            (batchnorm, dropout, weight_initializing)

        """
        
        super(MADDPG_Actor, self).__init__()
        
        ## we use nn.Modulelist to build structure of layers
        
        self.hiddens = nn.ModuleList()
        
        self.before_rnn_layers = before_rnn_layers
        
        self.after_rnn_layers = after_rnn_layers
        
        self.fc_layers = before_rnn_layers + after_rnn_layers
            
            
        #create layers
        
        if rnn_hidden_dim == 0:
            
            # create fully connected layers 
        
            for input_dim, output_dim in zip(self.fc_layers, self.fc_layers[1:]):
            
                self.hiddens.append(nn.Linear(input_dim, output_dim))

        else:
            
            #create before rnn layers
            
            for input_dim, output_dim in zip(before_rnn_layers, before_rnn_layers[1:]):
            
                self.hiddens.append(nn.Linear(input_dim, output_dim))
                
            #add rnn layer
        
            self.hiddens.append(nn.GRUCell(before_rnn_layers[-1], rnn_hidden_dim))
            
            #create after rnn layer
            
            after_rnn_layers = [rnn_hidden_dim] + self.after_rnn_layers

            for input_dim, output_dim in zip(after_rnn_layers,  after_rnn_layers[1:]):

                self.hiddens.append(nn.Linear(input_dim, output_dim))
        
        
        # activation functions 
        
        self.hidden_layer_act  = nn.ReLU()

            
    def forward(self, activations, hidden_state):
        
        """
         
        feedforward for Critic network 

        Args:
        
            activations: inputs to network (torch tensor)
            
            hidden_state: hidden inputs to rnn network

        Returns:
        
            q_values: output of main network ( value of actions in states)
            
            new_hidden_state: outputs of rnn network (update hidden_states)

        """
        # try:
        
        # if hidden_state == None:
            
        # forward fully_connected layers

        # max_num_layers = len(self.fc_layers) - 1

        # for layer_index, targeted_layer in zip(range(max_num_layers), self.hiddens):

        #     if layer_index < len(self.hiddens) - 1:

        #         activations = targeted_layer(activations)

        #         activations = self.hidden_layer_act(activations)


        #     else:

        #         logits = targeted_layer(activations)
                
        # new_hidden_state = None
                   
        # except:
            
        #before rnn forward

        index_of_before_rnn = len(self.before_rnn_layers) - 1

        for layer_index, targeted_layer in zip(range(index_of_before_rnn), self.hiddens):

            activations = targeted_layer(activations)

            #activations = self.BatchNorms[layer_index](activations)

            activations = self.hidden_layer_act(activations)

            #activations = self.dropout(activations)
            
        # rnn forward
            
        index_of_rnn_layer = len(self.before_rnn_layers) - 1

        activations = self.hiddens[index_of_rnn_layer](activations, hidden_state)

        new_hidden_state  = activations
        
        # after rnn forward
    
        start_index = len(self.before_rnn_layers) 

        end_index = len(self.hiddens)

        for layer_index, targeted_layer in zip(range(start_index, end_index), self.hiddens[start_index : end_index]):

            if layer_index < len(self.hiddens) - 1:

                activations = targeted_layer(activations)

                #activations = self.BatchNorms[layer_index - 1](activations) # why -1? rnn layer dosen't have batchnorm
                                                                            # so num batchnorm layer is main layer -1

                activations = self.hidden_layer_act(activations)

                #activations = self.dropout(activations)

            else:

                logits = targeted_layer(activations)
                
                #logits = self.BatchNorms[layer_index - 1](activations)

                
        return logits, new_hidden_state    
    
    
    def clean_action(self, obs_batch, learner, collecting_data = True, use_target = False, args = None):
        
        batch_size, episode_limit, num_agent = obs_batch.shape[0], obs_batch.shape[1], obs_batch.shape[2]
        specify_agent = torch.eye(num_agent).expand(batch_size, episode_limit, num_agent, -1)
        obs_batch = utils.ToTensor_(obs_batch)
        actor_input = torch.cat([obs_batch, specify_agent], dim = 3)
        
        actions, actions_onehot, logits  = [], [], []
        actor_input = copy.deepcopy(actor_input[:, 0])
        actor_input = utils.ToTensor_(actor_input) 
        actor_input = rearrange(actor_input, "d0 d1 d2 -> (d0 d1) d2")
        if args.rnn_hidden_dim == 0:
            actions, hidden_states = self.forward(actor_input, None)
        else:
            hidden_states = rearrange(learner.hidden_states, "d0 d1 d2 -> (d0 d1) d2")
            actions, hidden_states = self.forward(actor_input, hidden_states)
            learner.hidden_states = hidden_states.view(batch_size, learner.num_agent, learner.rnn_hidden_dim)
          

        
        if batch_size == 1:
            action = action.detach()

            
        actions.append(action) 
        concatenator = utils.TensorStack(actions, dim=0) 
        if collecting_data == True:
            actions_batch = utils.ToNumpy_(concatenator.__next__())
        else: 
            actions_batch = concatenator.__next__()

        
        return actions_batch
        


class MADDPG_Critic(nn.Module):
    
    def __init__(self, layers, dropout_p):
        
        """
         
        Critic network 

        Args:
        
            layers: list --> contaning number of nurons at each layer of network
            
            dropout_pr: dropout probabilitiy

        Returns:
        
            no return --> building layers structure and initializing other methods used in network 
                            (batchnorm, dropout, weight_initializing)

        """
        
        super(MADDPG_Critic, self).__init__()
        
        # we use nn.Modulelist to build structure of layers
        
        self.hiddens = nn.ModuleList()
        
        # create fully connected layers of network
        
        for input_dim, output_dim in zip(layers, layers[1:]):
            
            self.hiddens.append(nn.Linear(input_dim, output_dim))
            
        self.Relu = nn.ReLU()

            
    def forward(self, activations):
        
        """
         
        feedforward for Critic network 

        Args:
        
            activations: inputs to network (torch tensor)

        Returns:
        
            values: outputs of network (value for each action)

        """
        
        num_layers = len(self.hiddens)
        
        for layer_index, targeted_layer in zip(range(num_layers), self.hiddens):
            
            if layer_index < num_layers - 1:
                
                activations = targeted_layer(activations)
                
                activations = self.Relu(activations)
                
            else:              
                
                activations = targeted_layer(activations)
                
        values = activations
                
        return values







