import numpy as np, os, time, random
from envs_repo.constructor import EnvConstructor
from models.constructor import ModelConstructor
from core.params import Parameters
import argparse, torch
from algos.erl_trainer import ERL_Trainer
import pdb
from loguru import logger
logger.remove(0)
new_level_1 = logger.level("EVALUATE", no=38)
new_level_2 = logger.level("TRAINING", no=38)
new_level_3 = logger.level("EXTRA", no=38)
new_level_4 = logger.level("EVOLUTION", no=38)
logger.add("evaluate.log", filter = lambda record : record["level"].name == "EVALUATE", format = "{level}-{message}-elapsed time is {elapsed}-local_time is {time}")
logger.add("evolution.log", filter = lambda record : record["level"].name == "EVOLUTION", format = "{level}-{message}")
logger.add("training.log", filter = lambda record : record["level"].name == "TRAINING", format = "{level}-{message}-elapsed time is {elapsed}-local_time is {time}")
logger.add("extra.log", filter = lambda record : record["level"].name == "EXTRA", format = "{level}-{message}")


        
parser = argparse.ArgumentParser()
        
# discriminate environment args

subparser = parser.add_subparsers(dest= "env")

Robot = subparser.add_parser("RobotManipulator")
        
# environment parameters

# ROBOT;
Robot.add_argument("--agents", default=3,  help='up to 5 agent can take', type = int)

Robot.add_argument("--warmup", default=20,  help='warm up to fill buffer not be empty for sampling', type = int)

Robot.add_argument("--before_rnn_layers",  nargs = "+", default = [64], help='nodes in fully connected network before rnn network', type = int)

Robot.add_argument("--after_rnn_layers", nargs = "+", default= [64],  help='nodes in fully connected network after rnn network', type = int)

Robot.add_argument("--rnn_hidden_dim", default=64,  help='nodes rnn network', type = int)

Robot.add_argument("--grad_clip", default=10, help='clipping gradient norm to prevent diverging', type = float)

Robot.add_argument("--lr_critic", default=0.001,  help='learning_rate of critic network', type = float)

Robot.add_argument("--lr_actor", default=0.005,  help='learning_rate of actor network', type = float)

Robot.add_argument("--critic_nodes_hidden_layers", nargs = "+", default = [128, 128], help='hidden nodes in layers of critic network', type = int)

Robot.add_argument("--tau", default=0.01,  help='percentage of grasping main network weights by targets networks', type = float)

Robot.add_argument("--reqularization", default=0.001, help='reqularization term added in calculating loss to preventing overfitting', type = float)

Robot.add_argument("--epsilon", default=1,  help='initial epsilon used in epsilon-greedy policy', type = float)

Robot.add_argument("--min_epsilon", default=0.05, help='minimum epsilon reacheable by epsilon-greedy policy', type = float)

Robot.add_argument("--epsilon_interval", default=1,  help='step to pass to update epsilon', type = int)

Robot.add_argument("--epsilon_range", default=50000, help='in how mant step epsilon reach it minimum amount', type = int)

Robot.add_argument("--episode_limit", default=500,  help='one epsidoe how many step should be --> better be 500', type = int)

Robot.add_argument("--num_landmark", default=3, help='number of targets in rl environment --> best is 3', type = int)

Robot.add_argument("--train_interval", default=1, help='step to pass to train once', type = int)

Robot.add_argument("--step_to_run", default=2000000,  help='how many steps algorithm will run in total', type = int)

Robot.add_argument("--gamma", default=0.99,  help='gamma as percentage we care about future rewards', type = float)

Robot.add_argument("--target_update_interval", default=50, help='steps to pass to update target networks weights', type = int)

Robot.add_argument("--curiosity_hidden_layers", nargs = "+", default= [64, 64, 64], help='number of hidden nodes in curiosity purpose created network', type = int)

Robot.add_argument("--robots", nargs = "+", default= ["Panda"], help='robots included', type = str)

Robot.add_argument("--grippers", nargs = "+", default= ["default"], help='robots grippers', type = str)

Robot.add_argument("--intrinsic_reward_scale", default=0,  help='percentage of loss in curiosity driven method to get as reward', type = float)

Robot.add_argument("--evaluate_episode", default=20, help='step to pass to evaluate trained network', type = int)

Robot.add_argument("--batch_size", default=64,  help='number of batch to input networks in one tarining step', type = int)

Robot.add_argument("--mode", default="very_easy",  help='Robot environment difficulity level', type = str)

Robot.add_argument("--task_name", default="Lift",  help='Robot environment implemented on which task', type = str)

Robot.add_argument("--robots_position", default="single-arm-opposed",  help='Robot position in ralation to ecah other', type = str)

Robot.add_argument("--reward_shaping", action="store_false",  help='use reward_shaping or not')

Robot.add_argument("--contro_freq", default=20, help='number of controller action on each second', type = int)

Robot.add_argument("--PER", action = "store_true",  help='use prioritzed reply buffer or not --> do not touch this is not acomplished')

Robot.add_argument("--seed", default= 991, help='seed', type = int)

Robot.add_argument("--savetag", default = "",  help='#Tag to append to savefile', type = str)

Robot.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)

Robot.add_argument('--buffer', type=float, help='Buffer size in million',  default=1.0)

Robot.add_argument('--alpha', type=float, help='Alpha for Entropy term ',  default=0.1)

Robot.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)

Robot.add_argument('--learning_start', type=int, help='Frames to wait before learning starts',  default=5000)

Robot.add_argument('--popsize', type=int, help='#Policies in the population',  default=10)

Robot.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=1)

Robot.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)

Robot.add_argument('--num_test', type=int, help='#Test envs to average on',  default=0) 

Robot.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=2)  


#######################  COMMANDLINE - ARGUMENTS ######################
# parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=2)
# parser.add_argument('--frameskip', type=int, help='Frameskip',  default=1)

#pdb.set_trace()

#Figure out GPU to use [Default is 0]
os.environ['CUDA_VISIBLE_DEVICES']=str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser)


#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

################################## Find and Set MDP (environment constructor) ########################
env_constructor = EnvConstructor(args.env_name, args) 


#######################  Actor, Critic and ValueFunction Model Constructor ######################
model_constructor = ModelConstructor(env_constructor.obs_dim, env_constructor.action_dim, 
                                     args.agents, args.before_rnn_layers, args.after_rnn_layers,
                                     args.rnn_hidden_dim, args.critic_nodes_hidden_layers)

ai = ERL_Trainer(args, model_constructor, env_constructor)
pdb.set_trace()
ai.train(args.total_steps)


