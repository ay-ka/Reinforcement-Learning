import torch
import numpy as np
from einops import rearrange
from cem_maddpg_models import Actor, Critic, CuriosityNetwork
import torch.nn.functional as F
import pdb
import copy
import torchviz 


class Trainer:
    
    def __init__(self, num_agent: int, obs_dim : int, before_rnn_layers : list, 
                 after_rnn_layers : list, rnn_hidden_dim : int, n_actions : int, 
                 target_update_interval : int, critic_input_dim : int, critic_nodes_hidden_layers : list,
                 tau : float, grad_clip : float, curiosity_hidden_layers : list,
                 actor_lr : float, critic_lr : float):
        
        """
         
        trainer for training actor and critic networks

        Args:
        
            tau: percentages of grabbing weights from main network in target network ( 1- tau is percantage of it's own
                 weights
                 
            num_agent: numer of agents cocurrently intracting with environment
        
            obs_dim: observation dim taken from environment at every state (all agent have same obs_dim) (int)
            
            critic_input_dim: dimension of input to critic network (obs_dim + num+agents + num_actions)
            
            critic_nodes_hidden_layers: list of number of nuerons in hidden layers of critic network
            
            n_actions: number of available action for all agnets (agents assumed to be homogenious)
            
            target_update_interval: time interval (number of calling train function) for updating target actor and critic
                                                                                weights wrt main actor and critic networks
                                                                                
            before_rnn_layers: number of nurons in each hidden layer for <fc> network before connecting to <rnn> (list)
            
            after_rnn_layers: number of nurons in each hidden layer for fc network after <rnn> network  (list)
            
            rnn_hidden_dim : number of nurons hidden layer of rnn network                                                                     
                                                                                
            curiosity_hidden_layers: number of nurons in each hidden layers of curiosity driven network 
                                    (target and predictor networks)                                                                     

            grad_clip: clip gradient calculated in every step
            
            actor_lr : actor network learning rate
            
            critci_lr : critic network learning rate

        Returns:
        
            no return --> creatingg actor and critic networks and their target networks as well

        """
        
        super(Trainer, self).__init__()
        
        self.target_update_interval = target_update_interval
        
        self.num_agent = num_agent
        
        self.rnn_hidden_dim = rnn_hidden_dim
        
        self.num_actions = n_actions
        
        
        
        actor_input_dim = obs_dim + num_agent
        
        #make actor layers
        
        actor_layers = [actor_input_dim]
            
        before_rnn_layers = actor_layers + before_rnn_layers
        
        after_rnn_layers.append(n_actions)
        
        
        
        #make critic layers
        
        critic_layers = [critic_input_dim]
        
        critic_layers = critic_layers + critic_nodes_hidden_layers
        
        critic_layers.append(1)
        
    
        # creating networks    
        
        self.actor = Actor(before_rnn_layers, after_rnn_layers, rnn_hidden_dim = self.rnn_hidden_dim, dropout_p = 0)
        
        self.target_actor = Actor(before_rnn_layers, after_rnn_layers, rnn_hidden_dim = self.rnn_hidden_dim, dropout_p = 0)
        
        self.critic = Critic(critic_layers, dropout_p = 0)
        
        self.target_critic = Critic(critic_layers, dropout_p = 0)
        
        
        
        
        # create an actor to play role of placeholder for evaluating ea transfered network
        
        self.actor_ea = Actor(before_rnn_layers, after_rnn_layers, rnn_hidden_dim, dropout_p = 0)
        
        
        # make curisity driven hidden layer nodes list compatible with observation dim
        
        curiosity_hidden_layers = [obs_dim * num_agent] + curiosity_hidden_layers
        
        
        #make curiosity target and predictor network
        
        self.target_network = CuriosityNetwork(layers = curiosity_hidden_layers, target = True)
        
        self.predictor_network = CuriosityNetwork(layers = curiosity_hidden_layers, target = False)
        
        
        #update target networks
        
        self.tau = tau

        self.grad_clip = grad_clip
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        
        # define optimizers
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr  = actor_lr, amsgrad = True)
        
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr  = critic_lr, amsgrad = True)
        
        
        
        
    
    def TrainActor(self, input_batch : dict, train_step : int, epsilon : float, requlirization : float):
        
        """
         
        trainer for training actor network

        Args:
        
            requlirization : requlizer intensity added when calculating loss
        
            input_batch: batch sampled from buffer (dictionary of named tuples)
            
            train_step: number of training step has been done up to now

            epsilon: small number used in epsilon-greedy method

        Returns:
        
            no return --> update weights of actor network

        """
        
        batch_return = self.PreperationBatch(input_batch)

        
        (obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, logit_batch, 
                reward_batch, done_batch, intrinsic_reward_batch) = batch_return
        
        
        #calculate loss
        
        policy_loss = self.ActorLOSS(obs_batch, state_batch, action_onehot_batch, epsilon, done_batch, requlirization)
        
        
        
        self.optimizer_actor.zero_grad()
        
        policy_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        
        self.optimizer_actor.step()
        
        # update target network weight
        
        if train_step % self.target_update_interval == 0:
        
            self.UpdateTargetWeights(actor = True)
            
        if train_step % 100 == 0:
            
            with open("cem_log", "a") as f:
                
                f.write("at train step  :" + str(train_step) + " actor parameters : \n" + "policy losss is : " + str(policy_loss) + "\n" + " grad_norm is : " + 
                        str(grad_norm) + "\n")
                

        
    def ActorLOSS(self, obs_batch, state_batch, action_onehot_batch, epsilon, done_batch, requlirization : float):
        
        """
         
        calculating actor network loss

        Args:
        
            requlirization : requlizer intensity added when calculating loss
        
            done_batch : batch of dones (means is env terminated or not) sampled from buffer (numpy.array)
        
            obs_batch: batch of observations sampled from buffer (numpy.array)
            
            state_batch: batch of states sampled from buffer (numpy.array)
            
            action_onehot_batch: batch of actions sampled from buffer in onehot form (numpy.array)

            epsilon: small number used in epsilon-greedy method

        Returns:
        
            loss --> loss for training actor network

        """
        
        batch_size = action_onehot_batch.shape[0]
        
        episode_limit = action_onehot_batch.shape[1]
        
        #calculate actions wrt current policy
        
        self.GetHiddenStates(batch_size)
        
        action_batch, agent_action_onehot, logit_batch = self.GetActions(obs_batch, collecting_data = False, epsilon = epsilon)
        
        # modify actions
        
        agent_action_onehot = agent_action_onehot.view(batch_size, -1, 1, self.num_agent * self.num_actions).expand(
                                                                                                -1, -1, self.num_agent, -1)
        new_actions = []
        
        for targeted_agent_index in range(self.num_agent):
            
            temp_action = torch.split(agent_action_onehot[:, :, targeted_agent_index, :], self.num_actions, dim=2)
            
            actions_agents = []
            
            for agent_index in range(self.num_agent):
                
                if targeted_agent_index == agent_index:
                    
                    actions_agents.append(temp_action[agent_index])
                    
                else:
                    
                    actions_agents.append( temp_action[agent_index].detach() )
                    
            actions_agents = self.TensorConcatenate_(actions_agents, dim = -1)
            
            new_actions.append( self.Unsqueeze(actions_agents, dim =2, tensor = True) )
            
        new_actions = self.TensorConcatenate_(new_actions, dim = 2)
        
          
        # make <actions_onehot_batch> first and second dim same as <state_batch> dims to handle concatination>
        
        state_batch = self.ToTensor_(state_batch)
        
        state_batch = state_batch.expand(-1, -1, self.num_agent, -1)
        
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1)

        critic_inputs = torch.cat((state_batch, new_actions, specify_agent),  dim = 3)
            
        # deactivate gradient process in critic parameters
        
        #for parameter in model.parameters():
    
            #parameter.requires_grad_(False)
        
        values = self.critic(critic_inputs)
        
        values = values.view(-1, 1)
        
        mask = self.ToTensor_( ( 1 - done_batch ).reshape(-1, 1) )
        
        loss = -1 * torch.mean(values * 1) + ( requlirization * (logit_batch.view(-1, 1) ** 2 ).mean() )
            
        return loss
    
    
    def GetActions(self, obs_batch, epsilon, ea = False, collecting_data = True, use_target = False):
        
        """
         
        get actions wrt to a policy (output of actor network)

        Args:
        
            ea: is evaluated network transfered from ea side
        
            obs_batch: batch of observations sampled from buffer (numpy.array)
            
            use_target: True or False --> True means using target actor network for getting actions

            collecting_data: the function is collected for filling buffer (True means for buffer False means for training)

            epsilon: small number used in epsilon-greedy method

        Returns:
        
            actions_batch: batch of actions taken from environment wrt to sampled obserations from buffer
            
            actions_onehot_batch: batch of actions taken from environment wrt to sampled obserations from buffer (onehot)
            
            logits_batch:  batch of logits (output of actor network) taken from environment wrt to sampled obserations \
                            from buffer 

            updated_epsilon: updated targeted epsilon

        """
        
        batch_size = obs_batch.shape[0]
        
        episode_limit = obs_batch.shape[1]
        
        actor_inputs, _ = self.GetInputs(obs_batch)
        
        actions, actions_onehot, logits  = [], [], []
        
        for time_step in range(episode_limit):

            action, action_onehot, logit = self.AgentAction(use_target, time_step, actor_inputs,
                                                            ea, epsilon)
        
            actions.append(action)
            
            actions_onehot.append(action_onehot)
            
            logits.append(logit)
                
        concatenator = self.TensorStack(actions, actions_onehot, logits, dim = 1)
        
        if collecting_data == True:
            
            actions_batch = self.ToNumpy_(concatenator.__next__())

            actions_onehot_batch = self.ToNumpy_(concatenator.__next__().detach())

            logit_batch = self.ToNumpy_(concatenator.__next__().detach())  
            
        else:
            
            actions_batch = concatenator.__next__()
        
            actions_onehot_batch = concatenator.__next__()

            logit_batch = concatenator.__next__()
        
        return actions_batch, actions_onehot_batch, logit_batch
    
    
    def GetInputs(self, obs_batch, last_action_onehot = None, next_obs_batch = None, action_onehot_batch = None, train = False):
        
        """
         
        create inputs of Critic (Q-Learning) network
        
        Args:
        
            obs_batch: observation batch sampled from buffer (num_episode x episode_limit x num_agent x observation_dim)
        
            
            next_obs_batch: next observation batch sampled from buffer 
                                                             (num_episode x episode_limit x num_agent x observation_dim)
            
            action_onehot_batch: current action_onehot batch sampled from buffer 
                                                             (num_episode x episode_limit x num_agent x num_actions)
            
            train: function is called for training or not (input_next needed or not)
            
        Returns:
        
            inputs: current timestep (t) inputs to critic network 
            
            inputs_next: next timesteps (t+1) inputs to critic network

        """
        
        batch_size = obs_batch.shape[0]
        
        episode_limit = obs_batch.shape[1]
        
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, self.num_agent, -1)
        
        obs_batch = self.ToTensor_(obs_batch)
        
        inputs = torch.cat([obs_batch, specify_agent], dim = 3)
        
        if train:
        
            next_obs_batch = self.ToTensor_(next_obs_batch)
            
            inputs_next = torch.cat([next_obs_batch, specify_agent], dim = 3)
              
        else:
            
            inputs_next = None
        
        return inputs, inputs_next
    
    def AgentAction(self, use_target, time_step, actor_inputs, ea, epsilon, gumbel_softmax = True):
        
        """
         
        get action of targeted agent
        
        Args:
        
            gumbel_softmax : use gumbel softmax or not
        
            epsilon: small number used in epsilon-greedy method
        
            ea: is evaluated network transfered from ea side
        
            actor_inputs: actor network inputs
            
            time_step: which time_step in our batch we are in
            
            use_target: True or False --> True means using target actor network for getting actions
            
        Returns:
        
            action : batch of chosen action at fix point
            
            action_onehot : batch of chosen action at fix point (onehot format)
            
            logit : batch of output of actor network at fix time step 
            
            updated_epsilon : updated epsilon used for epsilon greedy algorithm

        """
        
        actor_input = copy.deepcopy(actor_inputs[:, time_step])
        
        batch_size = actor_inputs.shape[0]
        
        if use_target:
                        
            actor_input = self.ToTensor_(actor_input)
            
            actor_input = rearrange(actor_input, "d0 d1 d2 -> (d0 d1) d2")
            
            if self.rnn_hidden_dim == 0:
                
                logit, target_hidden_states = self.target_actor(actor_input, None)
                
            else:
                
                target_hidden_states = rearrange(self.target_hidden_states, "d0 d1 d2 -> (d0 d1) d2")
                
                logit, target_hidden_states = self.target_actor(actor_input, target_hidden_states)
                
                self.target_hidden_states = target_hidden_states.view(batch_size, self.num_agent, self.rnn_hidden_dim)

                        
        else:
                    
            actor_input = self.ToTensor_(actor_input)
            
            actor_input = rearrange(actor_input, "d0 d1 d2 -> (d0 d1) d2")
                    
            if ea:
                
                if self.rnn_hidden_dim==0:
                    
                    logit, hidden_states = self.actor_ea(actor_input, None) 
                else:
                    
                    hidden_states = rearrange(self.hidden_states, "d0 d1 d2 -> (d0 d1) d2")
                    
                    logit, hidden_states = self.actor(actor_input, hidden_states) 
                    
                    self.hidden_states = hidden_states.view(batch_size, self.num_agent, self.rnn_hidden_dim)
                        
            else:
                
                if self.rnn_hidden_dim==0:
                    
                    logit, hidden_states = self.actor(actor_input, None) 
                else:
                    
                    hidden_states = rearrange(self.hidden_states, "d0 d1 d2 -> (d0 d1) d2")
                    
                    logit, hidden_states = self.actor(actor_input, hidden_states) 
                    
                    self.hidden_states = hidden_states.view(batch_size, self.num_agent, self.rnn_hidden_dim)

        # if use_target:
            
        #     action = torch.argmax(logit, axis = 1)
            
        #     action_onehot = torch.zeros(logit.shape[0], logit.shape[1])
            
        #     action_onehot[torch.arange(logit.shape[0]), action] = 1
            
        #     action = action.view(batch_size, self.num_agent, -1)
            
        #     logit = logit.view(batch_size, self.num_agent, -1)
            
        #     action_onehot = action_onehot.view(batch_size, self.num_agent, -1)

        # else:
                
        if gumbel_softmax:
            
            action_onehot = F.gumbel_softmax(logits = logit, hard = True)
            
            action = torch.argmax(action_onehot, dim = 1).view(batch_size, self.num_agent, -1)
            
            logit = logit.view(batch_size, self.num_agent, -1)
            
            action_onehot = action_onehot.view(batch_size, self.num_agent, -1)
            
            if batch_size == 1:
                
                action = action.detach()
                
                action_onehot = action_onehot.detach()
                
                logit = logit.detach()
        
        else:
                
            action, action_onehot, logit = self.Select_My_Way(logit = logit, epsilon = epsilon, batch_size = batch_size)
        
        return action, action_onehot, logit
                

    
    def Select_My_Way(self, logit, epsilon, batch_size):
        
        """ get action of targeted agent
        
        Args:
        
            logit : output of actor network (unnormilized log prob)
            
            epsilon : epsilon used for epsilon greedy algorithm
            
            batch_size : batch size sampled from buffer 
            
        Returns:
        
            action : batch of chosen action at fix point
            
            action_onehot : batch of chosen action at fix point (onehot format)
            
            logit : batch of output of actor network at fix time step 

        """
        
        actions, actions_onehot = [], []
                
        for batch_index in range(logit.shape[0]):
                    
            action, action_onehot, updated_epsilon = self.EpsilonGreedySelector(epsilon, self.num_actions, logit[batch_index])
            
            actions.append(action)
            
            actions_onehot.append(action_onehot)
            
        action_onehot = self.TensorStack(actions_onehot, dim = 0).__next__()
        
        action = self.TensorStack(actions, dim = 0).__next__()
        
        action_onehot = action_onehot.view(batch_size, self.num_agent, -1)
        
        action = action.view(batch_size, self.num_agent, -1)
        
        logit = logit.view(batch_size, self.num_agent, -1)
            
        return action, action_onehot, logit
        

    def EpsilonGreedySelector(self, epsilon : float, num_action : int, logits):
               
        """
         
        select actions based on q_values (output of critic network)
        
        Args:
        
            epsilon: float (small number for epsilon greedy process)
            
            num_action: number of available actions

            logits: output of actor network (tensor)
            
        Returns:
        
            no return --> get hidden_states for main and target critic network

        """
        
        if np.random.random() < epsilon:
            
            random_action_index = torch.randint(low = 0, high = num_action, size = (1,))

            mask = torch.eye(5)[random_action_index]

            masked_logits = (logits * mask)

            selected_action_logit = masked_logits[0][random_action_index]

            selected_action_logit = selected_action_logit - 1

            result = masked_logits - selected_action_logit.detach()

            action_onehot = (result * torch.eye(num_action)[random_action_index])

            action = torch.argmax(action_onehot)
            
        else:
            
            logit = self.Squeeze(logits, tensor = True)
            
            selected_action_logit_placeholder = torch.zeros([num_action])
            
            action_index = torch.argmax(logit)
            
            selected_action_logit_placeholder[action_index] = logit[action_index]
            
            constant = logit[action_index] - 1
            
            action_onehot_placeholder = selected_action_logit_placeholder - constant.detach()
            
            if action_onehot_placeholder[action_index] != 1:
            
                action_onehot_placeholder = action_onehot_placeholder + torch.eye(num_action)[action_index]
            
            action_onehot = action_onehot_placeholder * torch.eye(num_action)[action_index]
            
            action_onehot = action_onehot.view(1, -1)
            
            # action_onehot = F.gumbel_softmax(logits = logits, hard = True, tau = 1)
                
            action = torch.argmax(action_onehot)
        
        return action, action_onehot, epsilon
            
        
        
                    
    def TrainCritic(self, input_batch : dict, gamma : float, train_step : int, epsilon : float):
        
        """
         
        trainer for training actor network

        Args:
        
            input_batch: batch sampled from buffer (dictionary of named tuples)
            
            train_step: number of training step has been done up to now
            
            gamma: discounted factor 

            epsilon: float (small number for epsilon greedy process)

        Returns:
        
            no return --> update weights of critic network

        """
        
        batch_return = self.PreperationBatch(input_batch)
        
        (obs_batch, next_obs_batch, state_batch, next_state_batch, action_batch, action_onehot_batch, logit_batch, 
                reward_batch, done_batch, intrinsic_reward_batch) = batch_return
        
        critic_loss = self.CriticLoss(gamma, state_batch, next_state_batch, reward_batch, done_batch,
                                    action_batch, action_onehot_batch, next_obs_batch, intrinsic_reward_batch,
                                    epsilon)
        
        
        
        self.optimizer_critic.zero_grad()
        
        critic_loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        
        self.optimizer_critic.step()
        
        
        # update target network weight
        
        if train_step % self.target_update_interval == 0:
        
            self.UpdateTargetWeights(critic = True)
            
        if train_step % 100 == 0:
            
            with open("cem_log", "a") as f:
                
                f.write("at train step  : " + str(train_step) + "  critic parameters : \n" + "critic losss is : " + str(critic_loss) + "\n" + " grad_norm is : " + 
                        str(grad_norm) + "\n")
    
    
    def CriticLoss(self, gamma : float, state_batch, next_state_batch, reward_batch, done_batch, action_batch,
                   action_onehot_batch, next_obs_batch, intrinsic_reward_batch, epsilon : float):
        
        """
         
        calculating critic network loss

        Args:
        
            state_batch: state batch sampled from buffer (numpy.array)
            
            next_state_batch: next_state batches sampled from buffer (numpy.array)
            
            reward_batch: batch of rewards sampled from buffer (numpy.array)
            
            action_batch: batch of actions sampled from buffer (numpy.array)
            
            action_onehot_batch: batch of action in onehot format sampled from buffer (numpy.array)
            
            next_obs_batch: batch of next_observations sampled from buffer (numpy.array)
            
            intrinsic_reward_batch: batch of reward created by curiosity driven methods ((numpy.array))
            
            done_batch: batch of dones from buffer (specify episode finished in particular state or not) (numpy.array)
            
            gamma: discounted factor (float)

            epsilon: float (small number for epsilon greedy process)

        Returns:
        
            loss: loss for training critic network

        """
        
        batch_size = action_batch.shape[0]
        
        episode_limit = action_batch.shape[1]
        
        #calculate target actions wrt target_actor_model
        
        self.GetHiddenStates(batch_size)
        
        target_actions_batch, target_actions_onehot_batch, _, = self.GetActions(next_obs_batch, use_target = True,
                                                                                  collecting_data = True, epsilon = epsilon)
        
        
        
        # make <target_actions_onehot_batch> first and second dim same as <next_state_batch> dims to handle concatination>
    
        with torch.no_grad():
        
            next_state_batch = self.ToTensor_(next_state_batch)
            
            next_state_batch = next_state_batch.expand(-1, -1, self.num_agent, -1)

            target_actions_onehot_batch = self.ToTensor_(target_actions_onehot_batch)
            
            target_actions_onehot_batch =  target_actions_onehot_batch.view(batch_size, -1, 1, self.num_agent * self.num_actions
                                                                            ).expand(-1, -1, self.num_agent, -1)
            
            specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1)

            target_critic_inputs = torch.cat((next_state_batch, target_actions_onehot_batch, specify_agent),  dim = 3)

            #calculate next_state values
        
            next_state_values = self.target_critic(target_critic_inputs)
            
            next_state_values = self.ToNumpy_(next_state_values)
            
            # create shared rewards and standardalize
            
            reward_batch = self.StandardlizeReward(reward_batch, intrinsic = False)
            
            intrinsic_reward_batch = self.StandardlizeReward(intrinsic_reward_batch, intrinsic = True)
        
            #calculate targets
        
            targets = self.CalculateTargets(gamma, reward_batch, next_state_values, done_batch, intrinsic_reward_batch)

            
        # calculate cirrent_state_values
        
        #prepare action code

        action_onehot_batch = self.ToTensor_(action_onehot_batch)
            
        action_onehot_batch =  action_onehot_batch.view(batch_size, -1, 1, self.num_agent * self.num_actions
                                                                            ).expand(-1, -1, self.num_agent, -1)
        
        #prepare state code

        state_batch = self.ToTensor_(state_batch)
        
        state_batch = state_batch.expand(-1, -1, self.num_agent, -1)
        
        #prepare agent code
        
        specify_agent = torch.eye(self.num_agent).expand(batch_size, episode_limit, -1, -1)
        
        # get critic input

        current_critic_inputs = torch.cat((state_batch, action_onehot_batch.detach(), specify_agent), dim = 3)
  
        values = self.critic(current_critic_inputs)
        
        values = values.reshape(-1, 1)
        
        targets = self.ToTensor_(targets)
        
        #calculate loss
    
        mask = self.ToTensor_( (1 - done_batch).reshape(-1, 1) )
        
        loss  = torch.mean(torch.square(1 * (values - targets.detach() ) ) )

            
        return loss
    
    
    def TrainPredictorNetwork(self, next_obs):
            
        """
         
       train predictor network

        Args:
        
            next_obs: curiosity driven networks networks inputs (next_observations of current step)

        Returns:
        
            no return --> train predictor network

        """
        
        inputs = self.ToTensor_(next_obs) 
        
        inputs = rearrange(inputs, "d0 d1 d2 -> (d0 d1 d2)")
        
        criterion = torch.nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr = 0.00001)
        
        main_features = self.target_network(inputs)
        
        predicted_features = self.predictor_network(inputs)
        
        loss = criterion(main_features.detach(), predicted_features)

        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
    def StandardlizeReward(self, reward_batch, intrinsic):
        
        """
         
        calculating targets for using in learning process

        Args:
        
            intrinsic: batch of reward is belong to intrinsic rewards or extrinsic
            
            reward_batch: batch of rewards sampled from buffer (numpy.array)

        Returns:
        
            reward_standardlize : batch of reward taken from buffer in standardlize format

        """
        
        if intrinsic:
            
            reward_standardlize = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
            
        else:
        
            reward_batch = self.Squeeze(reward_batch, dim = 3)

            reward_batch = np.sum(reward_batch, axis = 2)

            reward_batch = self.Unsqueeze(reward_batch, dim = 2)

            reward_batch = self.Unsqueeze(reward_batch, dim = 2)

            reward_batch = np.broadcast_to(reward_batch, shape = (reward_batch.shape[0], reward_batch.shape[1], 
                                                                self.num_agent, reward_batch.shape[3]))
            
            reward_standardlize = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-5)
        
        return reward_standardlize
    
    
    def CalculateTargets(self, gamma : float, reward_batch, next_state_values, done_batch, intrinsic_reward_batch):
        
                
        """
         
        calculating targets for using in learning process

        Args:
        
            gamma: discounted factor
            
            reward_batch: batch of rewards sampled from buffer (numpy.array)
            
            done_batch: batch of dones sampled from buffer (specify episode finished in particular state or not) (numpy.array)
            
            next_state_batch: next_state batches sampled from buffer (numpy.array)
            
            intrinsic_reward_batch: batch of reward created by curiosity driven methods ((numpy.array))

        Returns:
        
            targets: batch of targets to use in calculating loss as estimates of real value 

        """
        
        targets = (reward_batch.reshape(-1, 1) + intrinsic_reward_batch.reshape(-1, 1)) + gamma * \
                                                    next_state_values.reshape(-1, 1) * (1 - done_batch.reshape(-1, 1))
        
        return targets 
    
    
    def UpdateTargetWeights(self, actor = False, critic = False):
        
        """
         
        updated target actor or critic networks with weights of main actor or critic networks

        Args:
        
            actor: True or False --> True means update target actor network weights
            
            critic: True or False --> True means update target critic network weights

        Returns:
        
            no return --> update meantioned taget network

        """
        
        if actor:
            
            for target_param, actor_param in zip(self.target_actor.parameters(), self.actor.parameters()):
                
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * actor_param.data)
            
        if critic:
        
            for target_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
                
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * critic_param.data)
    

    
    def PreperationBatch(self, input_batch : dict) -> list:
        
        """
         
        create seprated batch from transitions (obs_batch, state_batch, .....)

        Args:
        
            input_batch: batch sampled from buffer (dictionary of named tuples)

        Returns:
        
            output_batch: batch sampled from buffer with targeted modifications
                                                   (seprated batches --> obs_batch or next_obs_batch and ....)

        """
        
        obs_batch, next_obs_batch, state_batch, next_state_batch, intrinsic_reward_batch =  [], [], [], [], []
        
        action_batch, action_onehot_batch, logit_batch, done_batch, reward_batch = [], [], [], [], []
        
        
        #create batches as inputs to insert to train method

        for index, episode in input_batch.items():
            
            obs, next_obs, state, next_state, action, intrinsic_reward = [], [], [], [], [], []
            
            action_onehot, logit, done, reward = [], [], [], []
            
            for transition in episode:

                obs.append(transition.obs)

                next_obs.append(transition.next_obs)

                state.append(transition.state)

                next_state.append(transition.next_state)

                action.append(transition.action)

                reward.append(transition.reward)
                
                intrinsic_reward.append(transition.intrinsic_reward)

                done.append(transition.done)

                logit.append(transition.logit)

                action_onehot.append(transition.action_onehot)
                
            concatenateor = self.NumpyConcatenate(obs, next_obs, state, next_state,
                                              action, action_onehot, logit,
                                              reward, done, intrinsic_reward)   
            
            obs_batch.append([concatenateor.__next__()])
            
            next_obs_batch.append([concatenateor.__next__()])
            
            state_batch.append([concatenateor.__next__()])
            
            next_state_batch.append([concatenateor.__next__()])

            action_batch.append([concatenateor.__next__()])
            
            action_onehot_batch.append([concatenateor.__next__()])
            
            logit_batch.append([concatenateor.__next__()])
            
            reward_batch.append([concatenateor.__next__()])
            
            done_batch.append([concatenateor.__next__()])
            
            intrinsic_reward_batch.append([concatenateor.__next__()])
            

            
            
        concatenateor = self.NumpyConcatenate(obs_batch, next_obs_batch, state_batch, next_state_batch,
                                              action_batch, action_onehot_batch, logit_batch,
                                              reward_batch, done_batch, intrinsic_reward_batch)
        
        output_batch = []
        
        while True:
            
            try:
            
                output_batch.append( concatenateor.__next__() )
                
            except:
                
                break

        return output_batch
    
    def GetHiddenStates(self, num_episode_batch):
            
        """
         
        get rnn_network used in critic network hidden_state
        
        Args:
        
            num_episode_batch: batch_size (number of episode sampled from buffer)
            
        Returns:
        
            no return --> get hidden_states for main and target critic network

        """
        
        self.hidden_states = torch.zeros( (num_episode_batch, self.num_agent, self.rnn_hidden_dim) )
        
        self.target_hidden_states = torch.zeros( (num_episode_batch, self.num_agent, self.rnn_hidden_dim) )
    
    
    def TensorConcatenate(self, *inpt, dim = 0):
        
        """
         
        concatenate tensors to create batches

        Args:
        
            inpt: list of list of tensors
            
            dim: along which axis concatenate tensors

        Returns:
        
            generator --> each time returns a tensor concatenated from list of input tensors

        """
        
        for element in inpt:
            
            yield torch.cat(element, dim = dim)
            
    def TensorConcatenate_(self, inpt, dim = 0):
        
        """
         
        concatenate tensors 

        Args:
        
            inpt: list of tensors
            
            dim: along which axis concatenate tensors

        Returns:
        
            result: a torch.tensor datatructure concatenated from list of torch.tensor's

        """
        
        result = torch.cat(inpt, dim = dim)
        
        return result
            
            
    def TensorStack(self, *inpt, dim = 0):
        
        """
         
        stack tensors along one axis 

        Args:
        
            inpt: list of tensors list

        Returns:
        
            generator --> each time returns a tensor stacked from list of input tensors

        """
        
        for element in inpt:
            
            yield torch.stack(element, dim = dim)
            
            
    def NumpyConcatenate(self, *inpt, axis = 0):
        
        """
         
        concatenate numpy.array to create batches

        Args:
        
            inpt: list of list of numpy.arrays
            
            axis: along which axis concatenate arrays

        Returns:
        
            generator --> each time returns a numpy.array concatenated from list of input numpy.arrays

        """
        
        for element in inpt:
            
            yield np.concatenate(element, axis  = axis)
            
            
    def NumpyConcatenate_(self, inpt,  axis = 0):
        
        """
         
        concatenate numpy.array

        Args:
        
            inpt: list of numpy.arrays
            
            axis: along which axis concatenate arrays

        Returns:
        
            result: a numpy.array datatructure concatenated from list of numpy.array's

        """
        
        result = np.concatenate(inpt, axis = axis)
        
        return result
    
    
    def ToTensor(self, *inpt):
        
        """
         
        convert a list of <to tensor convertable> datastrcuture to tensors

        Args:
        
            inpt: list of <to tensor convertable> datastrcuture like numpy.array or list

        Returns:
        
            generator --> each time returns tensor of targeted element

        """
        
        for element in inpt:
            
            yield torch.tensor(element, dtype = torch.float32)
            
    def ToTensor_(self, inpt):
        
        """
         
        convert a <to tensor convertable> datastrcuture to tensors

        Args:
        
            inpt: a <to tensor convertable> datastracture 

        Returns:
        
            result --> to torch tensor converted input

        """
        
        result = torch.tensor(inpt, dtype = torch.float32)
        
        return result
    
    
    def ToNumpy(self, *inpt):
        
        """
         
        convert a list of <to numpy convertable> datastrcuture to numpy

        Args:
        
            inpt: list of <to numpy convertable> datastrcuture like torch.tensor or list

        Returns:
        
            generator --> each time returns numpy.array of targeted element

        """
        
        for element in inpt:
            
            yield np.array(element)
            
    def ToNumpy_(self, inpt):
        
        
        """
         
        convert a <to numpy.array convertable> datastrcuture to numpy.array

        Args:
        
            inpt: a <to numpy.array convertable> datastracture 

        Returns:
        
            result --> to numpy.array converted input

        """
        
        result = np.array(inpt)
        
        return result
    
    
    def Squeeze(self, inpt, tensor  = False, dim = 0):
        
        """
         
        remove an axis from input datastructure

        Args:
        
            inpt: datastructure (numpy.array or torch.tensor)
            
            tensor: True or False --> True means input datastructure is Tensor
            
            dim: along which direction remove axis

        Returns:
        
            result --> transformed datastructure to target type

        """
                                     
        if tensor:
            
            result = inpt.squeeze(dim = dim)
                                     
        else:
            
            result = np.squeeze(inpt, axis = dim)
                                     
        return result
    
    
    def Unsqueeze(self, inpt, tensor  = False, dim = 0):
        
                
        """
         
        add an axis from input datastructure

        Args:
        
            inpt: datastructure (numpy.array or torch.tensor)
            
            tensor: True or False --> True means input datastructure is Tensor
            
            dim: along which direction add axis

        Returns:
        
            result --> transformed datastructure to target type

        """
        
        if tensor:
            
            result = inpt.unsqueeze(dim = dim)
                                     
        else:
            
            result = np.expand_dims(inpt, axis = dim)
                                     
        return result


