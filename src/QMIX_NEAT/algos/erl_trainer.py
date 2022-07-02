
from turtle import pd
import numpy as np, os, time, random, torch, sys
import pickle as pkl
from zmq import device
from src.QMIX_NEAT.algos.neuroevolution import SSNE
from src.QMIX_NEAT.core import utils
from src.QMIX_NEAT.core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from src.QMIX_NEAT.core.buffer import Buffer
from src.QMIX_NEAT.neat.evolve import NEAT_EVO
import torch
import pdb
import einops
import gc
from collections import namedtuple
from loguru import logger



class ERL_Trainer:

	def __init__(self, args, model_constructor, env_constructor):
     
		logger.log("EXTRA", f"-random seed is 0 \n")
     
		self.epsilons = namedtuple("epsilons", field_names=["main_epsilon", "epsilon_interval", "min_epsilon", "epsilon_range","epsilon_decrease_rate", "epsilon"])
		self.epsilons = self.epsilons(0, 0, 0, 0, 0, 0)
		self.epsilons = self.epsilons._replace(main_epsilon = args.epsilon, epsilon_interval = args.epsilon_interval, min_epsilon = args.min_epsilon,
                                         epsilon_range = args.epsilon_range, epsilon = args.epsilon, epsilon_decrease_rate = ((args.epsilon - args.min_epsilon) / (args.epsilon_range)))
		self.args = args
		self.crash =  False
		self.fitness_add = {}
		self.fitness_add_index = 0
		self.crash_store = 0
		self.env_constructor = env_constructor
		self.policy_string = 'QMIX' if env_constructor.is_discrete else 'unknown'
		self.num_learn = 0
		self.besties = []
		#self.manager = Manager()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		#Initialize population
		self.population = []
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model(self.policy_string)[0].to(device= "cpu"))

		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)[0].to(device= "cpu")

		#PG Learner
		if env_constructor.is_discrete:
			from algos.qmix import QMIX
			self.learner = QMIX(args, model_constructor, env_constructor.action_dim)
		else:
			pass
   
		#wandb.watch(self.learner.actor, log_freq=10, log="gradient")

		#Replay Buffer
		self.replay_buffer = Buffer(args.buffer_size)
  
  		#Evolution
		self.evolver = NEAT_EVO(buffer = self.replay_buffer, num_run = 5, rl_network = self.learner.qmix_critic, args = args) 

		#Initialize Rollout Bucket
		self.rollout_bucket = []
		for _ in range(args.rollout_size):
			self.rollout_bucket.append(model_constructor.make_model(self.policy_string)[0].to(device= "cpu"))

		self.evo_flag = [True for _ in range(args.pop_size)]
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = []
		self.test_bucket.append(model_constructor.make_model(self.policy_string)[0].to(device="cpu"))
		self.test_flag = True

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None
  
  
		self.average_fitness = []
		self.eps_terminate = {}

  
    
            



	def forward_generation(self, gen, tracker, average_fitness):

		gen_max = -float('inf')
  
  
		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		logger.log("EXTRA", f"gen {gen}")
		logger.log("EXTRA", f"num updates is {self.learner.num_updates}")
		for _ in range(self.num_learn):
			keys, data_batch, buffer_indices = self.replay_buffer.sample(self.args.batch_size)
			self.learner.updateParameters(data_batch, buffer_indices, self.eps_terminate)
			self.average_fitness.append(0)
		self.gen_frames = 0
		self.num_learn = 0

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_fitness = []; rollout_eplens = []
		self.learner.qmix_critic.cpu()
		if self.args.rollout_size > 0:
			for id in range(self.args.rollout_size):
				utils.hard_update(self.rollout_bucket[id], self.learner.qmix_critic)
				_, fitness, pg_frames, transitions, self.epsilons, eps_terminate, store_frame = rollout_worker(id, "pg", True, self.rollout_bucket,
                                                                       self.env_constructor, self.learner, self.epsilons)
				logger.log("TRAINING", f"gen {gen} id {id} fitness {fitness}")
				self.replay_buffer.add(transitions)
				if eps_terminate:
					self.eps_terminate[self.replay_buffer.store_index] = store_frame
				self.gen_frames += pg_frames; self.total_frames += pg_frames
				if fitness > self.best_score: torch.save(self.rollout_bucket[id].state_dict(), 
                                             "./result_actor_model_" + str(gen)); torch.save(self.learner.qmixnet.state_dict(), "./result_critic_model_" + str(gen))
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				rollout_fitness.append(fitness); rollout_eplens.append(pg_frames)
				self.num_learn += 1
		self.learner.qmix_critic.to(device = self.device)

		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			self.replay_buffer = self.evolver.Run()
		self.evolver.add_rl_genome(self.learner.qmix_critic, None)


		print(f"rollout fitness is {rollout_fitness[-1]}")

		print(f"best score is {self.best_score}")

		print(f"store index is {self.replay_buffer.store_index}")

  
		if gen > 1:

			print(f"taken transitions are {keys}")

		self.besties.append(self.best_score)

		return gen_max, _, _, _, _, rollout_fitness, rollout_eplens


	def train(self, frame_limit):
     
		self.learner.qmix_critic.to(device= self.device)
		self.learner.qmix_critic_target.to(device= self.device)
		self.learner.qmixnet.to(device= self.device)
		self.learner.qmixnet_target.to(device= self.device)
		# Define Tracker class to track scores
		#pdb.set_trace()
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()
		average_fitness = []
  
			

		for gen in range(1, 40000):  # Infinite generations
      
			print(f"generation is {gen} ")

			if gen % 50 == 0:

				torch.save(self.learner.qmix_critic.state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.qmixnet.state_dict(), "./result_critic_model_" + str(gen))
    
			if gen % 200 == 0:
				torch.save(self.learner.qmix_critic.state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.qmixnet.state_dict(), "./result_critic_model_" + str(gen))
				print(f"total frame is {self.total_frames}")
				self.average_fitness = []
			# Train one iteration
			max_fitness, champ_len, all_eplens, test_mean, test_std, rollout_fitness, rollout_eplens = self.forward_generation(gen, test_tracker, average_fitness)
			if test_mean: self.args.writer.add_scalar('test_score', test_mean, gen)

			gc.collect()
   
			try:
       
				print(self.besties[-1000])
       
				if np.mean(self.besties[-1000:]) <= self.besties[-1] + 3 and np.mean(self.besties[-1000:]) >= self.besties[-1] - 3:
        
					self.epsilons = self.epsilons._replace(epsilon = .3)

					self.epsilons = self.epsilons._replace(epsilon_range = 30000)

					self.besties = []
     
			except:

				print(self.epsilons.epsilon)
       
				pass

			if gen % 100 == 0 and gen>= 0:
					for eval_count in range(100):
						_, fitness, _, _, _, _ = rollout_worker(None, "test_add", True, None, self.env_constructor, self.learner, self.epsilons)
						logger.log("EVALUATE", f"gen {gen} step {eval_count} fitness {fitness}")
   
   
