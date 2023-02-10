import torch
import copy
import pdb

class ModelConstructor:

    def __init__(self, obs_dim, action_dim, num_agents, before_rnn_h_size, after_rnn_h_size,
                 qmix_net_hidden_dim, hypernet_hidden_dim, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.num_agents = num_agents
        self.state_dim = (obs_dim * self.num_agents)
        self.obs_dim = obs_dim + num_agents + action_dim
        self.action_dim = action_dim
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed
        actor_input_dim = self.obs_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #make actor layers
        actor_layers = [actor_input_dim]
        self.before_rnn_layers = actor_layers + before_rnn_h_size
        after_rnn_h_size.append(self.action_dim)
        self.after_rnn_layers = after_rnn_h_size
        
        
        
        #make critic layers
        self.hypernet_hidden_dim = hypernet_hidden_dim
        self.qmix_net_hidden_dim = qmix_net_hidden_dim



    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        # if type == 'Gaussian_FF':
        #     from models.continous_models import Gaussian_FF
        #     model = Gaussian_FF(self.state_dim, self.action_dim, self.hidden_size)
        #     if seed:
        #         model.load_state_dict(torch.load(self.critic_seed))
        #         print('Critic seeded from', self.critic_seed)


        # elif type == 'Tri_Head_Q':
        #     from models.continous_models import Tri_Head_Q
        #     model = Tri_Head_Q(self.state_dim, self.action_dim, self.hidden_size)
        #     if seed:
        #         model.load_state_dict(torch.load(self.critic_seed))
        #         print('Critic seeded from', self.critic_seed)

        if type == 'QMIX':
            from models.discrete_models import QMIX_Critic, QMIXNet
            qmix_critic = QMIX_Critic(self.before_rnn_layers, self.after_rnn_layers).to(device=self.device)
            qmix_critic_target = QMIX_Critic(self.before_rnn_layers, self.after_rnn_layers).to(device=self.device)
            qmixnet = QMIXNet(self.hypernet_hidden_dim, self.qmix_net_hidden_dim, self.state_dim, self.num_agents).to(device=self.device)
            qmixnet_target = QMIXNet(self.hypernet_hidden_dim, self.qmix_net_hidden_dim, self.state_dim, self.num_agents).to(device=self.device)
        else:
            AssertionError('Unknown model type')


        return qmix_critic, qmix_critic_target, qmixnet, qmixnet_target



