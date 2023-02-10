from torch import nn
import torch
from einops import rearrange
from torch.distributions import OneHotCategorical
import pdb


class CuriosityNetwork(nn.Module):
    
    def __init__(self, layers, target = True):
        
        """

        curiosity networks 

        Args:

            layers: number of hidden nodes on hidden layers of network
            
            target: is model created to represent a target network or predictor network --> refer to curiosity driven paper  

        Returns:

            no return -->  no return --> building layers structure and initializing other methods used in network 
                            (batchnorm, weight_initializing)

        """
        
        super(CuriosityNetwork, self).__init__()
        
        self.hiddens = nn.ModuleList()
        
        # create layers
        
        for input_dim, output_dim in zip(layers[0:], layers[1:]):
        
            self.hiddens.append(nn.Linear(input_dim, output_dim))
            
            
        
        #batch normalization and weight initialazation
        
        self.BatchNorms = []
        
        for hidden_layer in self.hiddens:
            
            #batch_norm part
            
            input_to_batch_norm = hidden_layer.weight.shape[0]
            
            self.BatchNorms.append(nn.BatchNorm1d(input_to_batch_norm))
            
            #weight initialazation
            
            if target:
            
                nn.init.orthogonal_(hidden_layer.weight.data)
                
                nn.init.constant_(hidden_layer.bias.data, 0)
                
            else:
                
                nn.init.orthogonal_(hidden_layer.weight.data)
                
                nn.init.constant_(hidden_layer.bias.data, 1)
                
                
    def forward(self, activation):
            
            
        for layer_index in range(0, len(self.hiddens) - 1):
            
            layer = self.hiddens[layer_index]

            activation = torch.relu(layer(activation))

        last_layer = self.hiddens[-1]

        features = torch.tanh(last_layer(activation))

        return features
    
class Actor(nn.Module):
    
    
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
        
        super(Actor, self).__init__()
        
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
            
            
        # create batch normalization layers
        
        # self.BatchNorms = []
            
        # for layer_index in range(1, len(self.before_rnn_layers)):
            
        #     BatchNormLayer = nn.BatchNorm1d(self.before_rnn_layers[layer_index])
            
        #     self.BatchNorms.append(BatchNormLayer)
            
        # for layer_index in range(0, len(self.after_rnn_layers)):
            
        #     BatchNormLayer = nn.BatchNorm1d(self.after_rnn_layers[layer_index])
            
        #     self.BatchNorms.append(BatchNormLayer)
            
        
        #create dropout layer
            
        # self.dropout = nn.Dropout(p = dropout_p)
        
        
        # activation functions 
        
        self.hidden_layer_act  = nn.ReLU()
        
        
        
        # weight initializing of layers
        
        # if rnn_hidden_dim != 0:
        
        #     #before rnn layers

        #     before_rnn_layers_index = len(self.before_rnn_layers) - 1

        #     for layer_index in range(before_rnn_layers_index):

        #         nn.init.orthogonal_(self.hiddens[layer_index].weight.data)

        #     #rnn layer

        #     rnn_layer_index = len(self.before_rnn_layers) - 1

        #     nn.init.orthogonal_(self.hiddens[rnn_layer_index].weight_ih.data)

        #     nn.init.orthogonal_(self.hiddens[rnn_layer_index].weight_hh.data)

        #     #after rnn layer

        #     start_index = len(self.before_rnn_layers)

        #     end_index = start_index + len(self.after_rnn_layers)

        #     for layer_index in range(start_index, end_index):

        #         nn.init.orthogonal_(self.hiddens[layer_index].weight.data)
                
        # else:
            
        #     max_num_layers = len(self.fc_layers) - 1

        #     for layer_index in range(max_num_layers):

        #         nn.init.orthogonal_(self.hiddens[layer_index].weight.data)


            
    def forward(self, activations, hidden_state = None):
        
        """
         
        feedforward for Critic network 

        Args:
        
            activations: inputs to network (torch tensor)
            
            hidden_state: hidden inputs to rnn network

        Returns:
        
            q_values: output of main network ( value of actions in states)
            
            new_hidden_state: outputs of rnn network (update hidden_states)

        """
        
        if hidden_state == None:
              
            # forward fully_connected layers

            max_num_layers = len(self.fc_layers) - 1

            for layer_index, targeted_layer in zip(range(max_num_layers), self.hiddens):

                if layer_index < len(self.hiddens) - 1:

                    activations = targeted_layer(activations)

                    activations = self.hidden_layer_act(activations)


                else:

                    logits = targeted_layer(activations)
                    
            new_hidden_state = None
                   
        else:
            
            # before rnn forward

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

    


class Critic(nn.Module):
    
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
        
        super(Critic, self).__init__()
        
        # we use nn.Modulelist to build structure of layers
        
        self.hiddens = nn.ModuleList()
        
        # create fully connected layers of network
        
        for input_dim, output_dim in zip(layers, layers[1:]):
            
            self.hiddens.append(nn.Linear(input_dim, output_dim))
            
        # create batch normalization layers
        
        # self.BatchNorms = []
            
        # for layer_index in range(1, len(self.hiddens)):
            
        #     BatchNormLayer = nn.BatchNorm1d(layers[layer_index])
            
        #     self.BatchNorms.append(BatchNormLayer)
        
        #create dropout layer
            
        # self.dropout = nn.Dropout(p = dropout_p)
        
        #weight iniytializing of layers (ortaghonal weight initialization)
        
        # for layer in self.hiddens:
            
        #     nn.init.orthogonal_(layer.weight.data)
            
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
                
                # try:
                
                #     activations = self.BatchNorms[layer_index](activations)
                    
                # except:
                    
                #     pass
                
                activations = self.Relu(activations)
                
                # activations = self.dropout(activations)
                
            else:              
                
                activations = targeted_layer(activations)
                
        values = activations
                
        return values
                

