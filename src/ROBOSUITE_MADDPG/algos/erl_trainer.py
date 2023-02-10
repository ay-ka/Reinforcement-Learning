import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from core.buffer import Buffer
import torch
import pdb
import wandb
from loguru import logger


class ERL_Trainer:
    

	def __init__(self, args, model_constructor, env_constructor):
     
		logger.log("EXTRA", f"-random seed is 0 \n")

		self.args = args
		self.env_constructor = env_constructor
		self.policy_string = 'MADDPG' if env_constructor.is_discrete else 'MADDPG'
		self.num_learn = 0
		#self.manager = Manager()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		#Evolution
		self.evolver = SSNE(self.args) 

		#Initialize population
		self.population = []
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model(self.policy_string)[0])

		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)[0]

		#PG Learner
		if env_constructor.is_discrete:
			from algos.maddpg import MADDPG
			self.learner = MADDPG(args, model_constructor, env_constructor.action_dim)
		else:
			from algos.maddpg import MADDPG
			self.learner = MADDPG(args, model_constructor, env_constructor.action_dim)
   
		#wandb.watch(self.learner.actor, log_freq=10, log="gradient")

		#Replay Buffer
		self.replay_buffer = Buffer(args.buffer_size)

		#Initialize Rollout Bucket
		self.rollout_bucket = []
		for _ in range(args.rollout_size):
			self.rollout_bucket.append(model_constructor.make_model(self.policy_string)[0])

		self.evo_flag = [True for _ in range(args.pop_size)]
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = []
		self.test_bucket.append(model_constructor.make_model(self.policy_string)[0])
		self.test_flag = True

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None
  
  
		self.average_fitness = []
		


	def forward_generation(self, gen, tracker, average_fitness):

		gen_max = -float('inf')
  
		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		logger.log("EXTRA", f"gen {gen}")
		logger.log("EXTRA", f"num updates is {self.learner.num_updates}")
		if gen>1:
			for _ in range(1):
				data_batch = self.replay_buffer.sample(self.args.batch_size)
				self.learner.update_parameters(data_batch)
			self.gen_frames = 0
			self.num_learn = 0
		#pdb.set_trace()


		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for id in range(self.args.pop_size):
				_, fitness, frames, transitions = rollout_worker(id, "evo", self.args.rollout_size > 0, self.population, self.env_constructor, self.learner, self.args)
				all_fitness.append(fitness); all_eplens.append(frames)
				self.gen_frames+= frames; self.total_frames += frames
				self.replay_buffer.add(transitions)
				if fitness > self.best_score:
					torch.save(self.population[id].state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.critic.state_dict(), "./result_critic_model_" + str(gen))
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				self.num_learn += 1
		#pdb.set_trace()

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_fitness = []; rollout_eplens = []
		if self.args.rollout_size > 0:
			for id in range(self.args.rollout_size):
				utils.hard_update(self.rollout_bucket[id], self.learner.actor)
				_, fitness, pg_frames, transitions = rollout_worker(id, "pg", True, self.rollout_bucket, self.env_constructor, self.learner, self.args)
				logger.log("TRAINING", f"gen {gen} id {id} fitness {fitness}")
				self.replay_buffer.add(transitions)
				self.gen_frames += pg_frames; self.total_frames += pg_frames
				if fitness > self.best_score: torch.save(self.rollout_bucket[id].state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.critic.state_dict(), "./result_critic_model_" + str(gen))
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				rollout_fitness.append(fitness); rollout_eplens.append(pg_frames)
				self.num_learn += 1
		#pdb.set_trace()

		######################### END OF PARALLEL ROLLOUTS ################

		############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
		if self.args.pop_size > 1:
			champ_index = all_fitness.index(max(all_fitness))
			utils.hard_update(self.test_bucket[0], self.population[champ_index])
			if max(all_fitness) > self.best_score:
				index = np.argmax(all_fitness)
				torch.save(self.population[index].state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.critic.state_dict(), "./result_critic_model_" + str(gen))
				self.best_score = max(all_fitness)
				utils.hard_update(self.best_policy, self.population[champ_index])
				torch.save(self.population[champ_index].state_dict(), self.args.aux_folder + '_best'+self.args.savetag)
				print("Best policy saved with score", '%.2f'%max(all_fitness))

		else: #If there is no population, champion is just the actor from policy gradient learner
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])



		test_mean, test_std = None, None

		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket)

		#Compute the champion's eplen
		champ_len = all_eplens[all_fitness.index(max(all_fitness))] if self.args.pop_size > 1 else rollout_eplens[rollout_fitness.index(max(rollout_fitness))]


		return gen_max, champ_len, all_eplens, 0, 0, rollout_fitness, rollout_eplens


	def train(self, frame_limit):
		# Define Tracker class to track scores
		#pdb.set_trace()
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()
		average_fitness = []

		for gen in range(1, 2000000000000):  # Infinite generations

			print(f"generation {gen}")
    
			if gen % 2000 == 0:
				torch.save(self.learner.actor.state_dict(), "./result_actor_model_" + str(gen)); torch.save(self.learner.critic.state_dict(), "./result_critic_model_" + str(gen))
				print(f"total frame is {self.total_frames}")

			max_fitness, champ_len, all_eplens, test_mean, test_std, rollout_fitness, rollout_eplens = self.forward_generation(gen, test_tracker, average_fitness)
			if test_mean: self.args.writer.add_scalar('test_score', test_mean, gen)
   



			if gen % 100 == 0 and gen>= 0:
				for eval_count in range(100):
					_, fitness, _, _, _, _ = rollout_worker(None, "test_add", True, None, self.env_constructor, self.learner, self.epsilons, self.args)
					logger.log("EVALUATE", f"gen {gen} step {eval_count} fitness {fitness}")




