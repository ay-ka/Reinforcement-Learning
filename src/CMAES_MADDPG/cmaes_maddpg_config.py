import argparse

def get_config():
    
        """

        get argument necessary for algorithms

        Args:

            no input

        Returns:

            parser : a parser object to convert terminal argument to python object argument to use in scripts

        """
        
        # prepare parameters
        
        parser = argparse.ArgumentParser()
        
        # discriminate environment args
        
        subparser = parser.add_subparsers(dest= "env")
        
        MPE = subparser.add_parser("MPE")
        
        RWARE = subparser.add_parser("RWARE")

        PressurePlate = subparser.add_parser("PressurePlate")
        
        # environment parameters
        
        # MPE;
        
        MPE.add_argument("--agents", default=3, help='it is best to have 3 agents', type = int)
        
        MPE.add_argument("--warmup", default=20,  help='warm up to fill buffer not be empty for sampling', type = int)
        
        MPE.add_argument("--before_rnn_layers",  nargs = "+", default = [64], help='nodes in fully connected network before rnn network', type = int)
        
        MPE.add_argument("--after_rnn_layers", nargs = "+", default= [64],  help='nodes in fully connected network after rnn network', type = int)
        
        MPE.add_argument("--rnn_hidden_dim", default=64,  help='nodes rnn network', type = int)
        
        MPE.add_argument("--grad_clip", default=10, help='clipping gradient norm to prevent diverging', type = int)
        
        MPE.add_argument("--lr_critic", default=0.001,  help='learning_rate of critic network', type = float)
        
        MPE.add_argument("--lr_actor", default=0.005,  help='learning_rate of actor network', type = float)
        
        MPE.add_argument("--critic_nodes_hidden_layers", nargs = "+", default = [128, 128], help='hidden nodes in layers of critic network', type = int)
        
        MPE.add_argument("--tau", default=0.01,  help='percentage of grasping main network weights by targets networks', type = float)
        
        MPE.add_argument("--requlirization", default=0.001, help='reqularization term added in calculating loss to preventing overfitting', type = float)
        
        MPE.add_argument("--epsilon", default=1,  help='initial epsilon used in epsilon-greedy policy', type = float)
        
        MPE.add_argument("--min_epsilon", default=0.05, help='minimum epsilon reacheable by epsilon-greedy policy', type = float)
        
        MPE.add_argument("--epsilon-interval", default=1,  help='step to pass to update epsilon', type = int)
        
        MPE.add_argument("--epsilon_range", default=50000, help='in how mant step epsilon reach it minimum amount', type = int)
        
        MPE.add_argument("--episode_limit", default=25,  help='one epsidoe how many step should be --> better be 25', type = int)
        
        MPE.add_argument("--train_interval", default=1, help='step to pass to train once', type = int)
        
        MPE.add_argument("--step_to_run", default=2000,  help='how many steps algorithm will run in total', type = int)   
                
        MPE.add_argument("--num_landmark", default=3, help='number of targets in rl environment --> best is 3', type = int)
        
        MPE.add_argument("--gamma", default=0.99,  help='gamma as percentage we care about future rewards', type = float)
        
        MPE.add_argument("--target_update_interval", default=50, help='steps to pass to update target networks weights', type = int)
                 
        MPE.add_argument("--curiosity_hidden_layers", nargs = "+", default= [512, 256, 128, 64, 8], help='number of hidden nodes in curiosity purpose created network', type = int)
        
        MPE.add_argument("--intrinsic_reward_scale", default = 0,  help='percentage of loss in curiosity driven method to get as reward', type = float)
        
        MPE.add_argument("--evaluate_episode", default=20, help='step to pass to evaluate trained network', type = int)
        
        MPE.add_argument("--batch_size", default=64,  help='number of batch to input networks in one tarining step', type = int)
        
        MPE.add_argument("--mode", default="very_easy",  help='RWARE environment difficulity level', type = str)
        
        MPE.add_argument("--PER", action = "store_true",  help='use prioritzed reply buffer or not --> do not touch this is not acomplished')
        
        MPE.add_argument("--parameters_lower_bound", default="-100", help='decision parameters value can not go beyond this value', type = int)
    
        MPE.add_argument("--parameters_upper_bound", default="100", help='decision parameters value can not go beyond this value', type = int)
            
        MPE.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
        
        MPE.add_argument("--max_iteration", default="5", help='max iteration for evolutionary method', type = int)
            
        MPE.add_argument("--initial_sigma", default="1e-3", help='initial standard deviation for calculation initial covariances', type = float)
        
        MPE.add_argument("--symmetry", action="store_false", help='devide population to 2 part which are symmetry to each other --> parameters in second part of population is negative of first part of population')
        
        MPE.add_argument("--num_parents_prob", default=".5", help='rate of parent number to population size in each generation to create off springs', type = float)
        
        MPE.add_argument("--step_size", default=".5", help='covariance calculation parameter --> (refer to paper)', type = float)
        
        # Rware
    
        RWARE.add_argument("--agents", default=3,  help='up to 5 agent can take', type = int)
        
        RWARE.add_argument("--warmup", default=20,  help='warm up to fill buffer not be empty for sampling', type = int)
        
        RWARE.add_argument("--before_rnn_layers",  nargs = "+", default = [64], help='nodes in fully connected network before rnn network', type = int)
        
        RWARE.add_argument("--after_rnn_layers", nargs = "+", default= [64],  help='nodes in fully connected network after rnn network', type = int)
        
        RWARE.add_argument("--rnn_hidden_dim", default=64,  help='nodes rnn network', type = int)
        
        RWARE.add_argument("--grad_clip", default=10, help='clipping gradient norm to prevent diverging', type = float)
        
        RWARE.add_argument("--lr_critic", default=0.001,  help='learning_rate of critic network', type = float)
        
        RWARE.add_argument("--lr_actor", default=0.005,  help='learning_rate of actor network', type = float)
        
        RWARE.add_argument("--critic_nodes_hidden_layers", nargs = "+", default = [128, 128], help='hidden nodes in layers of critic network', type = int)
        
        RWARE.add_argument("--tau", default=0.01,  help='percentage of grasping main network weights by targets networks', type = float)
        
        RWARE.add_argument("--requlirization", default=0.001, help='reqularization term added in calculating loss to preventing overfitting', type = float)
        
        RWARE.add_argument("--epsilon", default=1,  help='initial epsilon used in epsilon-greedy policy', type = float)
        
        RWARE.add_argument("--min_epsilon", default=0.05, help='minimum epsilon reacheable by epsilon-greedy policy', type = float)
        
        RWARE.add_argument("--epsilon_interval", default=1,  help='step to pass to update epsilon', type = int)
        
        RWARE.add_argument("--epsilon_range", default=50000, help='in how mant step epsilon reach it minimum amount', type = int)
        
        RWARE.add_argument("--episode_limit", default=500,  help='one epsidoe how many step should be --> better be 500', type = int)
        
        RWARE.add_argument("--num_landmark", default=3, help='number of targets in rl environment --> best is 3', type = int)
        
        RWARE.add_argument("--train_interval", default=1, help='step to pass to train once', type = int)
        
        RWARE.add_argument("--step_to_run", default=2000000,  help='how many steps algorithm will run in total', type = int)
        
        RWARE.add_argument("--gamma", default=0.99,  help='gamma as percentage we care about future rewards', type = float)
        
        RWARE.add_argument("--target_update_interval", default=50, help='steps to pass to update target networks weights', type = int)
        
        RWARE.add_argument("--curiosity_hidden_layers", nargs = "+", default= [64, 64, 64], help='number of hidden nodes in curiosity purpose created network', type = int)
        
        RWARE.add_argument("--intrinsic_reward_scale", default=0,  help='percentage of loss in curiosity driven method to get as reward', type = float)
        
        RWARE.add_argument("--evaluate_episode", default=20, help='step to pass to evaluate trained network', type = int)
        
        RWARE.add_argument("--batch_size", default=64,  help='number of batch to input networks in one tarining step', type = int)
        
        RWARE.add_argument("--mode", default="very_easy",  help='RWARE environment difficulity level', type = str)
        
        RWARE.add_argument("--PER", action = "store_true",  help='use prioritzed reply buffer or not --> do not touch this is not acomplished')

        RWARE.add_argument("--parameters_lower_bound", default="-100", help='decision parameters value can not go beyond this value', type = int)
        
        RWARE.add_argument("--parameters_upper_bound", default="100", help='decision parameters value can not go beyond this value', type = int)
            
        RWARE.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
        
        RWARE.add_argument("--max_iteration", default="5", help='max iteration for evolutionary method', type = int)
            
        RWARE.add_argument("--initial_sigma", default="1e-3", help='initial standard deviation for calculation initial covariances', type = float)
        
        RWARE.add_argument("--symmetry", action="store_false", help='devide population to 2 part which are symmetry to each other --> parameters in second part of population is negative of first part of population')
        
        RWARE.add_argument("--num_parents_prob", default=".5", help='rate of parent number to population size in each generation to create off springs', type = float)
        
        RWARE.add_argument("--step_size", default=".5", help='covariance calculation parameter --> (refer to paper)', type = float)
        # Pressure Plate;
        
        PressurePlate.add_argument("--agents", default=4, help='it is best to have 3 agents', type = int)
        
        PressurePlate.add_argument("--warmup", default=2,  help='warm up to fill buffer not be empty for sampling', type = int)
        
        PressurePlate.add_argument("--before_rnn_layers",  nargs = "+", default = [64], help='nodes in fully connected network before rnn network', type = int)
        
        PressurePlate.add_argument("--after_rnn_layers", nargs = "+", default= [64],  help='nodes in fully connected network after rnn network', type = int)
        
        PressurePlate.add_argument("--rnn_hidden_dim", default= 32,  help='nodes rnn network', type = int)
        
        PressurePlate.add_argument("--grad_clip", default=10, help='clipping gradient norm to prevent diverging', type = float)
        
        PressurePlate.add_argument("--lr_critic", default=0.001,  help='learning_rate of critic network', type = float)
        
        PressurePlate.add_argument("--lr_actor", default=0.005,  help='learning_rate of actor network', type = float)
        
        PressurePlate.add_argument("--critic_nodes_hidden_layers", nargs = "+", default = [128, 128], help='hidden nodes in layers of critic network', type = int)
        
        PressurePlate.add_argument("--tau", default=0.01,  help='percentage of grasping main network weights by targets networks', type = float)
        
        PressurePlate.add_argument("--requlirization", default=0.001, help='reqularization term added in calculating loss to preventing overfitting', type = float)
        
        PressurePlate.add_argument("--epsilon", default=1,  help='initial epsilon used in epsilon-greedy policy', type = float)
        
        PressurePlate.add_argument("--min_epsilon", default=0.05, help='minimum epsilon reacheable by epsilon-greedy policy', type = float)
        
        PressurePlate.add_argument("--epsilon-interval", default=1,  help='step to pass to update epsilon', type = int)
        
        PressurePlate.add_argument("--epsilon_range", default=50000, help='in how mant step epsilon reach it minimum amount', type = int)
        
        PressurePlate.add_argument("--episode_limit", default=1000,  help='one epsidoe how many step should be --> better be 25', type = int)
        
        PressurePlate.add_argument("--train_interval", default=1, help='step to pass to train once', type = int)
        
        PressurePlate.add_argument("--step_to_run", default=2000000,  help='how many steps algorithm will run in total', type = int)   
                
        PressurePlate.add_argument("--num_landmark", default=3, help='number of targets in rl environment --> best is 3', type = int)
        
        PressurePlate.add_argument("--gamma", default=0.99,  help='gamma as percentage we care about future rewards', type = float)
        
        PressurePlate.add_argument("--target_update_interval", default=50, help='steps to pass to update target networks weights', type = int)
                 
        PressurePlate.add_argument("--curiosity_hidden_layers", nargs = "+", default= [512, 256, 128, 64, 8], help='number of hidden nodes in curiosity purpose created network', type = int)
        
        PressurePlate.add_argument("--intrinsic_reward_scale", default = 0,  help='percentage of loss in curiosity driven method to get as reward', type = float)
        
        PressurePlate.add_argument("--evaluate_episode", default=30, help='step to pass to evaluate trained network', type = int)
        
        PressurePlate.add_argument("--batch_size", default=64,  help='number of batch to input networks in one tarining step', type = int)
        
        PressurePlate.add_argument("--mode", default="very_easy",  help='RWARE environment difficulity level', type = str)
        
        PressurePlate.add_argument("--PER", action = "store_true",  help='use prioritzed reply buffer or not --> do not touch this is not acomplished')
        
        PressurePlate.add_argument("--parameters_lower_bound", default="-100", help='decision parameters value can not go beyond this value', type = int)
        
        PressurePlate.add_argument("--parameters_upper_bound", default="100", help='decision parameters value can not go beyond this value', type = int)
            
        PressurePlate.add_argument("--pop_size", default="20",  help='number of members in population', type = int)
        
        PressurePlate.add_argument("--max_iteration", default="5", help='max iteration for evolutionary method', type = int)
            
        PressurePlate.add_argument("--initial_sigma", default="1e-3", help='initial standard deviation for calculation initial covariances', type = float)
        
        PressurePlate.add_argument("--symmetry", action="store_false", help='devide population to 2 part which are symmetry to each other --> parameters in second part of population is negative of first part of population')
        
        PressurePlate.add_argument("--num_parents_prob", default=".5", help='rate of parent number to population size in each generation to create off springs', type = float)
        
        PressurePlate.add_argument("--step_size", default=".5", help='covariance calculation parameter --> (refer to paper)', type = float)
        
        return parser
        