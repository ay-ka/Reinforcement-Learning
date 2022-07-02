
from turtle import pd
import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from core.buffer import Buffer, PrioritizedBuffer
import torch
import pdb
import einops
import gc
from collections import namedtuple
np.random.seed(10), random.seed(10), torch.manual_seed(10)
from loguru import logger



class ERL_Trainer:

	def __init__(self, args, model_constructor, env_constructor):

		logger.log("EXTRA", f"-random seed is 0 \n")

		self.args = args
		self.crash =  False
		self.fitness_add = {}
		self.fitness_add_index = 0
		self.store_indices = []
		self.crash_store = 0
		self.env_constructor = env_constructor
		self.policy_string = 'MADDPG' if env_constructor.is_discrete else 'unknown'
		self.num_learn = 0
		#self.manager = Manager()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model_constructor = model_constructor
		self.limit_buffer = 300

		#Evolution
		self.evolver = SSNE(self.args) 

		#Initialize population
		self.population = []
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model(self.policy_string)[0].to(device=self.device))

		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)[0].to(device=self.device)

		#PG Learner
		if env_constructor.is_discrete:
			from algos.maddpg import MADDPG
			self.learner = MADDPG(args, model_constructor, env_constructor.action_dim)
		else:
			pass
   
		#wandb.watch(self.learner.actor, log_freq=10, log="gradient")

		#Replay Buffer
		self.replay_buffer = PrioritizedBuffer(buffer_capacity = args.buffer_size, episode_limit = args.episode_limit)

		#Initialize Rollout Bucket
		self.rollout_bucket = []
		for _ in range(args.rollout_size):
			self.rollout_bucket.append(model_constructor.make_model(self.policy_string)[0].to(device=self.device))

		self.evo_flag = [True for _ in range(args.pop_size)]
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = []
		self.test_bucket.append(model_constructor.make_model(self.policy_string)[0].to(device=self.device))
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
		if gen > 1:
			self.num_learn = 10
		for _ in range(self.num_learn):
			keys, data_batch, buffer_indices = self.replay_buffer.sample(self.args.batch_size)
			self.learner.update_parameters(data_batch)
		self.gen_frames = 0
		self.num_learn = -1



		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for id in range(self.args.pop_size):
				_, fitness, frames, transitions, eps_terminate, store_frame = rollout_worker(id, "evo", self.args.rollout_size > 0, 
                                                                    self.population, self.env_constructor, self.learner, self.args)
				all_fitness.append(fitness); all_eplens.append(frames)

				if fitness >= self.best_score - self.limit_buffer: #and fitness <= self.best_score + 5:
					print(f"{fitness} is greater than {self.best_score}")
					if self.env_constructor.env_name == "RWARE":
						if fitness > 0:
							self.replay_buffer.add(transitions, fitness)
					else:
						self.replay_buffer.add(transitions, fitness)
					self.fitness_add[self.fitness_add_index] = fitness
					self.fitness_add_index += 1
					if eps_terminate:
						self.eps_terminate[self.replay_buffer.store_index] = store_frame	
				if fitness > self.best_score + 10:
					torch.save(self.population[id].state_dict(), "./result_actor_model_evo_best" + str(gen)); torch.save(self.learner.critic.state_dict(), 
                                                                                                  "./result_critic_model_evo_best" + str(gen))																		  
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				self.num_learn += 1

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_fitness = []; rollout_eplens = []
		self.learner.actor.cpu()
		if self.args.rollout_size > 0:
			for id in range(self.args.rollout_size):
				utils.hard_update(self.rollout_bucket[id], self.learner.actor)
				_, fitness, pg_frames, transitions, eps_terminate, store_frame = rollout_worker(id, "pg", True, self.rollout_bucket,
                                                                       self.env_constructor, self.learner, self.args)
				logger.log("TRAINING", f"gen {gen} id {id} fitness {fitness}")
				if gen > 1:
					if fitness >= self.best_score - self.limit_buffer: #and fitness <= self.best_score + 5:
						print(f"{fitness} is greater than {self.best_score}")
						if self.env_constructor.env_name == "RWARE":
							if fitness > 0:
								self.replay_buffer.add(transitions, fitness)
						else:
							self.replay_buffer.add(transitions, fitness)						
						if eps_terminate:
							self.eps_terminate[self.replay_buffer.store_index] = store_frame
					self.gen_frames += pg_frames; self.total_frames += pg_frames

					if fitness > self.best_score + 10:
						torch.save(self.learner.actor.state_dict(), "./result_actor_model_pg_best" + str(gen)); torch.save(self.learner.critic.state_dict(), 
                                                                                                  "./result_critic_model_pg_best" + str(gen))
					self.best_score = max(self.best_score, fitness)
					#print(f"roll fitness is {fitness}")
				else:
					if fitness >= self.best_score - self.limit_buffer: #and fitness <= self.best_score + 5:
						if self.env_constructor.env_name == "RWARE":
							self.replay_buffer.add(transitions, fitness)
						else:
							self.replay_buffer.add(transitions, fitness)
						self.gen_frames += pg_frames; self.total_frames += pg_frames
						self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				rollout_fitness.append(fitness); rollout_eplens.append(pg_frames)
				self.num_learn += 1
				self.average_fitness.append(fitness)
		self.learner.actor.to(device = self.device)


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket)

		#Compute the champion's eplen
		champ_len = all_eplens[all_fitness.index(max(all_fitness))] if self.args.pop_size > 1 else rollout_eplens[rollout_fitness.index(max(rollout_fitness))]

		print(f"rollout fitness is {rollout_fitness[-1]}")
		print(f"best score is {self.best_score}")
		print(f"store index is {self.replay_buffer.store_index}")
		print(self.num_learn)

		return gen_max, champ_len, all_eplens, _, _, rollout_fitness, rollout_eplens


	def train(self, frame_limit):
		
		self.learner.actor.to(device= self.device)
		self.learner.actor_target.to(device= self.device)
		self.learner.critic.to(device= self.device)
		self.learner.critic_target.to(device= self.device)


		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()
		average_fitness = []

		for gen in range(1, 2000):  # Infinite generations

			print(f"generation is {gen}")
			
			if gen % 8000 == 0:

				torch.save(self.learner.qmix_critic.state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.qmixnet.state_dict(), "./result_critic_model_" + str(gen))
    
			if gen % 200 == 0:
				torch.save(self.learner.actor.state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.critic.state_dict(), "./result_critic_model_" + str(gen))
				print(f"total frame is {self.total_frames}")
				with open("ant_logg", "a") as f:
					f.write(f"average 5000 is {np.mean(self.average_fitness)}")
					f.write(f"std for 5000 is {np.std(self.average_fitness)}")
				self.average_fitness = []
			# Train one iteration
			max_fitness, champ_len, all_eplens, test_mean, test_std, rollout_fitness, rollout_eplens = self.forward_generation(gen, test_tracker, average_fitness)
			if test_mean: self.args.writer.add_scalar('test_score', test_mean, gen)


			gc.collect()

			#control limit buffer
			if gen % 200 == 0:
				try:
					if self.store_indices[-1] - self.store_indices[-10]  <= 3:
						if self.limit_buffer <= 10:
							self.limit_buffer += 2
							with open("logs", "a") as f:
								f.write(f"limit is :{self.limit_buffer}")
								f.write(f"\n")
				except:
					pass


			#control epsilon
			if self.replay_buffer.episode_record >= 7000 and self.bool == True :
				with open("logs", "a") as f:
					f.write("ok")
					self.epsilons = self.epsilons._replace(min_epsilon = .05)
					f.write(str(self.epsilons.min_epsilon))
				self.bool = False

			if gen % 100 == 0 and gen>= 0:
				for eval_count in range(100):
					_, fitness, _, _, _, _ = rollout_worker(None, "test_add", True, None, self.env_constructor, self.learner, self.epsilons)
					logger.log("EVALUATE", f"gen {gen} step {eval_count} fitness {fitness}")



