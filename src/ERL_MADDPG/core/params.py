import os
from torch.utils.tensorboard import SummaryWriter

class Parameters:
    def __init__(self, parser):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        #Env args
        self.env_name = vars(parser.parse_args())['env']
        self.agents = vars(parser.parse_args())['agents']
        self.warmup = vars(parser.parse_args())['warmup']
        self.before_rnn_layers = vars(parser.parse_args())['before_rnn_layers']
        self.after_rnn_layers = vars(parser.parse_args())['after_rnn_layers']
        self.rnn_hidden_dim = vars(parser.parse_args())['rnn_hidden_dim']
        self.grad_clip = vars(parser.parse_args())['grad_clip']
        self.critic_nodes_hidden_layers = vars(parser.parse_args())['critic_nodes_hidden_layers']
        self.reqularization = vars(parser.parse_args())['reqularization']
        self.episode_limit = vars(parser.parse_args())['episode_limit']
        self.epsilon = vars(parser.parse_args())['epsilon']
        self.min_epsilon = vars(parser.parse_args())['epsilon_interval']
        self.epsilon_interval = vars(parser.parse_args())['epsilon_interval']
        self.epsilon_range = vars(parser.parse_args())['epsilon_range']
        self.train_interval = vars(parser.parse_args())['train_interval']
        self.num_landmark = vars(parser.parse_args())['num_landmark']
        self.evaluate_episode = vars(parser.parse_args())['evaluate_episode']
        self.mode = vars(parser.parse_args())['mode']
        self.PER = vars(parser.parse_args())['PER']
        
        
        self.total_steps = int(vars(parser.parse_args())['total_steps'] * 1000000)
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.savetag = vars(parser.parse_args())['savetag']
        self.seed = vars(parser.parse_args())['seed']
        self.batch_size = vars(parser.parse_args())['batch_size']
        self.rollout_size = vars(parser.parse_args())['rollsize']

        self.critic_lr = vars(parser.parse_args())['lr_critic']
        self.actor_lr = vars(parser.parse_args())['lr_actor']
        self.tau = vars(parser.parse_args())['tau']
        self.gamma = vars(parser.parse_args())['gamma']
        self.reward_scaling = vars(parser.parse_args())['reward_scale']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.learning_start = vars(parser.parse_args())['learning_start']

        self.pop_size = vars(parser.parse_args())['popsize']
        self.num_test = vars(parser.parse_args())['num_test']
        self.test_frequency = 1
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

        #Non-Args Params
        self.elite_fraction = 0.2
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform


        self.alpha = vars(parser.parse_args())['alpha']
        self.target_update_interval = vars(parser.parse_args())['target_update_interval']
        self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)
        self.aux_folder = 'Results/Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.savetag += str(self.env_name)
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_pop' + str(self.pop_size)
        self.savetag += '_alpha' + str(self.alpha)


        self.writer = SummaryWriter(log_dir='Results/tensorboard/' + self.savetag)




