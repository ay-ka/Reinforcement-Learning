import torch
import copy
torch.manual_seed(10)

class ModelConstructor:

    def __init__(self, obs_dim, action_dim, num_agents, before_rnn_h_size, after_rnn_h_size, rnn_hidden_dim,
                 critic_hidden_sizes, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed
        self.num_agents = num_agents
        self.state_dim = (self.obs_dim * self.num_agents) + (self.num_agents * self.action_dim) + self.num_agents
        self.rnn_hidden_dim = rnn_hidden_dim
        actor_input_dim = self.obs_dim + self.num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #make actor layers
        actor_layers = [actor_input_dim]
        self.before_rnn_layers = actor_layers + before_rnn_h_size
        after_rnn_h_size.append(self.action_dim)
        self.after_rnn_layers = after_rnn_h_size
        
        
        
        #make critic layers
        self.critic_layers = [self.state_dim]
        self.critic_layers = self.critic_layers + critic_hidden_sizes
        self.critic_layers.append(1)


    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        if type == 'MADDPG':
            from models.discrete_models import MADDPG_Actor, MADDPG_Critic
            model_actor = MADDPG_Actor(self.before_rnn_layers, self.after_rnn_layers, rnn_hidden_dim = self.rnn_hidden_dim, dropout_p = 0).to(device=self.device)
            model_target_actor = MADDPG_Actor(self.before_rnn_layers, self.after_rnn_layers, rnn_hidden_dim = self.rnn_hidden_dim, dropout_p = 0).to(device=self.device)
            model_critic = self.critic = MADDPG_Critic(self.critic_layers, dropout_p = 0).to(device=self.device)
            model_target_critic = MADDPG_Critic(self.critic_layers, dropout_p = 0).to(device=self.device)

        else:
            AssertionError('Unknown model type')


        return model_actor, model_critic, model_target_actor, model_target_critic



