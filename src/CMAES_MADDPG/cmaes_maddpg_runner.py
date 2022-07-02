import torch
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper as pettingzoowrapper
from gym.wrappers.order_enforcing import OrderEnforcing as normalwrapper
import numpy as np
import pressureplate
from collections import namedtuple
from einops import rearrange
import gym
import rware
import pdb
from cmaes_maddpg_buffer import Buffer, PrioritizedBuffer
from cmaes_maddpg_trainer import Trainer
import time
import pickle
import einops


class Runner:
    
    def __init__(self, args):

        """
         
        runner object to make agents intract with envioronment and crate data (transitions and episodes)

        Args:
        
            args : get all argument needed

        Returns:
        
            no return --> initialize buffer and environmnet and trainer

        """
        
        self.crash = False
        
        self.crash_store = 0

        self.evaluate_episode = args.evaluate_episode
        
        self.episodes_avg_reward = []
        
        self.all_fitness = []
        
        self.num_migration = 0
        
        self.num_agents = args.agents
        
        self.num_landmarks = args.num_landmark
        
        self.num_warmup_eps = args.warmup
        
        self.train_interval = args.train_interval
        
        self.target_update_interval = args.target_update_interval
        
        self.step_to_run = args.step_to_run
        
        self.requlirization = args.requlirization
        
        self.total_episodes = 0
        
        self.transition = namedtuple("Transition", field_names=["obs", "state", "action", "next_obs",
                                           "next_state", "done", "reward", "logit", "action_onehot", "intrinsic_reward"])
        
        self.main_epsilon = args.epsilon
                                     
        self.eval_epsilon = 0 # in evaluation process all action taken from max operation
                                     
        self.epsilon = self.main_epsilon
                                     
        self.epsilon_interval = args.epsilon_interval
                                     
        self.min_epsilon = args.min_epsilon

        self.epsilon_range = args.epsilon_range
                                     
        self.epsilon_decrease_rate = (self.main_epsilon - self.min_epsilon) / (self.epsilon_range)
        
        self.intrinsic_reward_scale = args.intrinsic_reward_scale
                                     
        self.transition = self.transition(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        self.total_step = 0
                                     
        self.train_step = 0
        
        self.keep = True

        self.grad_clip = args.grad_clip
        
        self.finished_first_eps = False
        
        self.episode_limit = args.episode_limit
                                     
        self.lr_actor = args.lr_actor
                                     
        self.lr_critic = args.lr_critic
                                     
        self.gamma = args.gamma
        
        self.tau = args.tau
        
        self.keep_obs = None  # storing last observation (episode limit) from 
                              # n'th episode to continue from that spesific point for n+1"th episode
        
        self.env = self.CreateEnv(args.env, mode = args.mode)

        if isinstance(self.env, pettingzoowrapper):

            self.num_actions = self.env.action_space("agent_0").n

            self.observation_dim = self.env.observation_spaces["agent_0"].shape[0]

        else:

            self.num_actions = self.env.action_space[0].n

            self.observation_dim = self.env.observation_space[0].shape[0]

        self.state_dim = (self.observation_dim * self.num_agents) + (self.num_agents * self.num_actions) + self.num_agents
        
        self.critic_nodes_hidden_layers = args.critic_nodes_hidden_layers
        
        self.trainer =  Trainer(obs_dim = self.observation_dim, before_rnn_layers = args.before_rnn_layers,
                        after_rnn_layers =args. after_rnn_layers, rnn_hidden_dim = args.rnn_hidden_dim, actor_lr = args.lr_actor, critic_lr = args.lr_critic
                        ,n_actions = self.num_actions, target_update_interval = self.target_update_interval, num_agent = self.num_agents,
                        curiosity_hidden_layers = args.curiosity_hidden_layers, critic_input_dim = self.state_dim, 
                        critic_nodes_hidden_layers = self.critic_nodes_hidden_layers, tau = self.tau, grad_clip = self.grad_clip)
        
        if args.PER:
            
            self.Buffer = PrioritizedBuffer(batch_size = args.batch_size)
            
        else:
            
            self.Buffer =  Buffer(batch_size = args.batch_size)
            
        if self.crash:
            
            self.fillBuffer()
            
    def checkpoint(self):

        """
         
        a function to save transition in hardware for reusing

        Args:
        
            no argument

        Returns:
        
            no return --> save buffer contain into hardware

        """

        obs_list, action_onehot_list, done_list = [], [], []

        reward_list, intrinsic_reward_list, logits_list = [], [], []
        
        buffer_dict = {}
        
        for transition_index, target_index in enumerate(range(self.crash_store - 10, self.crash_store)):
            
            buffer_dict[transition_index] = self.Buffer.buffer[target_index]

        for store_index, (key, transition) in enumerate(buffer_dict.items()):

            if store_index > len(list(buffer_dict.keys())) - 1:

                break

            for step_index, step in enumerate(transition):

                obs_list.append(step.obs)

                if step_index == len(transition) - 1:

                    obs_list.append(step.next_obs)

                action_onehot_list.append(step.action_onehot)

                done_list.append(step.done)

                reward_list.append(step.reward)

                intrinsic_reward_list.append(step.intrinsic_reward)

                logits_list.append(step.logit)

        results = [np.concatenate(obs_list), np.concatenate(action_onehot_list),
                   np.concatenate(done_list), np.concatenate(reward_list),
                   np.concatenate(intrinsic_reward_list), np.concatenate(logits_list)]

        for file_counter, result in enumerate(results):

            with open("res/buf_res_" + str(file_counter) + "_" + str(self.crash_store) + ".pkl", "wb") as file:

                pkl.dump(result, file)
        
            
    def fillBuffer(self):
        
        """
         
        a function to fill buffer with saved transition in case of crash in previously tried training 

        Args:
        
            no argument

        Returns:
        
            no return --> fill buffer

        """
        
        self.Buffer.buffer = dict()
        
        experience = namedtuple("Experience", field_names=["obs", "state", "action", "next_obs",
                                           "next_state", "done", "reward", "logit", "action_onehot", "intrinsic_reward"])
        
        #initialize buffer
        
        for episode in range(self.Buffer.buffer_capacity):
    
            self.Buffer.buffer[episode] = []
    
            for limit in range(self.episode_limit):
        
                self.Buffer.buffer[episode].append(experience(0,0,0,0,0,0,0,0,0,0))
        
        self.Buffer.store_index, alg.Buffer.episode_record = 0, 0
        
        for step_of_store in range(10, 10010, 10):
            
            saved_transitions = []
            
            try:

                for counter in range(6):

                    with open("res/buf_res_" + str(counter) + "_" + str(step_of_store) + ".pkl", "rb") as file:

                        saved_transitions.append(pkl.load(file))
                        
            except:
                
                break

            saved_obs, saved_action_onehot = saved_transitions[0], saved_transitions[1]

            saved_done, saved_reward = saved_transitions[2], saved_transitions[3]

            saved_intrinsic_reward, saved_logits = saved_transitions[4], saved_transitions[5]

            chuncked = list(map(self.chunk, [saved_obs, saved_action_onehot, saved_done, saved_reward,
                              saved_intrinsic_reward, saved_logits]))

            obs_result, action_onehot_result, done_result = chuncked[0], chuncked[1], chuncked[2]

            reward_result, intrinsic_reward_result, logits_result = chuncked[3], chuncked[4], chuncked[5] 

            zipped_result = list(zip(obs_result, action_onehot_result, done_result,
                        reward_result, intrinsic_reward_result, logits_result))

            for obs_eps, action_onehot_eps, done_eps, reward_eps, intrinsic_reward_eps, logits_eps in zipped_result:

                one_eps_buffer = []

                for transition_index in range(len(logits_eps)):

                    transition = namedtuple("Transition", field_names=["obs", "next_obs", "state",
                                                            "next_state", "intrinsic_reward", "logit",
                                                            "action", "action_onehot", "done", "reward"])

                    #create obs and obs_next

                    obs = np.expand_dims(obs_eps[transition_index], axis = 0)

                    next_obs = np.expand_dims(obs_eps[transition_index + 1], axis = 0)

                    #create state

                    state = np.expand_dims(einops.rearrange(obs, "a1 a2 a3-> a1 (a2 a3)"), axis=0)

                    next_state = np.expand_dims(einops.rearrange(next_obs, "a1 a2 a3-> a1 (a2 a3)"), axis=0)

                    # create action

                    action_onehot = np.expand_dims(action_onehot_eps[transition_index], axis = 0)

                    action = np.expand_dims(np.argmax(action_onehot, axis = 2), axis=0)

                    #create others

                    reward = np.expand_dims(reward_eps[transition_index], axis = 0)

                    logit = np.expand_dims(logits_eps[transition_index], axis = 0)

                    intrinsic_reward = np.expand_dims(intrinsic_reward_eps[transition_index], axis = 0)

                    done = np.expand_dims(done_eps[transition_index], axis = 0)

                    transition = transition(obs = obs, next_obs = next_obs, state = state,
                                    next_state = next_state, action = action, action_onehot = action_onehot, 
                                    reward = reward, done = done, logit = logit, intrinsic_reward = intrinsic_reward)

                    one_eps_buffer.append(transition)

                self.Buffer.buffer[self.Buffer.store_index] = one_eps_buffer

                self.Buffer.episode_record += 1

                self.Buffer.store_index = self.Buffer.episode_record % self.Buffer.buffer_capacity
        
        
    def Run(self):
        
        """
         
        execute main loop of rl (collecting data and training)

        Args:
        
            no argument

        Returns:
        
            no return --> train used reinfoecement learning method to find best policies (MADDPG)

        """
        
        # fill buffer with couple of episodes
        self.warmup()
                                     
        # train
        self.Train()                             
        
    
    def warmup(self):
        
        """
         
        initially fill buffer for couple of episodes to not remain empty

        Args:
        
            no argument

        Returns:
        
            no return --> initially collect data from environment and fill buffer

        """
        
        warmup_rewards = []
        
        for episode in range(self.num_warmup_eps):
            
            eps_reward = self.collector(warmup = True)
            
            warmup_rewards.append(eps_reward)
                                     
                                     
    def Train(self):
        
        """
         
        train reinforcement learning method (MADDPG) for fixed times (total step)

        Args:
        
            no argument

        Returns:
        
            no return --> train MADDPG method for fixed times (total step)

        """
                                     
        episodes_avg_reward = []

        self.all_fitness = []
        
        self.num_migration = 0
                                     
        while self.total_step < self.step_to_run:
                                     
            eps_reward = self.collector(train = True)
                                     
            self.episodes_avg_reward.append(eps_reward)
            
            self.all_fitness.append(eps_reward)
                                     
            if len(self.episodes_avg_reward) == 2000:

                self.num_migration += 1
                                     
                five_eps_average_reward = np.mean(episodes_avg_reward)
                
                five_eps_average_reward_std = np.std(episodes_avg_reward)
                
                with open("ant_loggg", "a") as f:
                
                    f.write("average reeward for 50000 is" + str(five_eps_average_reward) + "\n")
                    
                    f.write("average std for five episode is" + str(five_eps_average_reward_std) + "\n")

                torch.save(self.trainer.actor.state_dict(), "./result_actor_model_" + str(self.num_migration))
                
                torch.save(self.trainer.critic.state_dict(), "./result_critic_model_" + str(self.num_migration))

                self.episodes_avg_reward = []
                
                self.evaluate(ea = False)
                
                
    def evaluate(self, ea):
            
        """
         
        evaluate our method

        Args:
        
            ea: do we evaluate network from ea side or not

        Returns:
        
            no return --> evaluate some spesific network

        """
                                     
        episodes_avg_reward = []
                                     
        for episode in range(self.evaluate_episode):
                                     
            eps_reward = self.collector(evaluation = True, ea = ea)
                                     
            episodes_avg_reward.append(eps_reward)
                                     
        five_eps_average_reward = np.mean(episodes_avg_reward)
                                     
        print(f" average reeward for five episode is {five_eps_average_reward} and rewards are : {episodes_avg_reward}")
            
            
    def collector(self, warmup = False, train = False, evaluation = False, ea = False):
        
        """
         
        roll agents on environment and collect data and for training or evaluation

        Args:
        
            warmup: True or False --> collector is called for initially fill buffer for couple of episodes or not
            
            train: True or False --> collector is called for training purposes
            
            evaluatio: True or False --> collector is called for avluation purposes
            
            ea : True or Flase --> use critic model assined to ea networks or not

        Returns:
        
            no return --> colect data and with respect to (warmup, train, evaluation operate differenetly)
        """
        
        if evaluation:

            if isinstance(self.env, pettingzoowrapper):
            
                self.env.reset()

                obs = []

                self.env.agent_selection = "agent_0"

                for agent in self.env.agents:

                    self.env.agent_selection = agent

                    observation, _, _, _ = self.env.last()

                    obs.append(observation)

                self.env.agent_selection = "agent_0"

            else:

                obs = self.env.reset()
            
            #convert list of array to numpy.array 
            
            obs = self.ToNumpy_(obs)

            # convert to batch like shape
            obs = self.Unsqueeze(obs, dim = 0)

            #get state  -> we mix second and third dimenstion together to combine obs for creating state
            state = rearrange(obs, "d0 d1 d2 -> d0 (d1 d2)")

            state = self.Unsqueeze(state, dim = 0)
        
        else:

            if isinstance(self.env, pettingzoowrapper):
            
                self.env.reset()

                obs = []

                self.env.agent_selection = "agent_0"

                for agent in self.env.agents:

                    self.env.agent_selection = agent

                    observation, _, _, _ = self.env.last()

                    obs.append(observation)

                self.env.agent_selection = "agent_0"

            else:

                obs = self.env.reset()
        
            #convert list of array to numpy.array 
            
            obs = self.ToNumpy_(obs)

            # convert to batch like shape
            obs = self.Unsqueeze(obs, dim = 0)
                    
                         

        episode_reward = []
        
        transitions = []
        
        self.trainer.GetHiddenStates(1)

        #img = plt.imshow(self.env.render(mode='rgb_array'))

        for step in range(self.episode_limit):
                                     
            self.ActiveEvalMode(actor = True)
            
            self.ActiveEvalMode(critic = True)

            action, action_onehot, logit = self.trainer.GetActions(self.Unsqueeze(obs, dim = 0), 
                                                                    collecting_data = True, ea = ea,
                                                                    epsilon = self.epsilon)
            
            #pdb.set_trace()
            
            action, action_onehot, logit = self.Squeeze(action), self.Squeeze(action_onehot), self.Squeeze(logit)

            if isinstance(self.env, normalwrapper):

                next_obs, rewards, dones, info = self.env.step(self.Squeeze(action, dim = 0))

                rewards = self.ToNumpy_(rewards).reshape(-1, 1).tolist()

                #rewards = [[np.sum(rewards)]] * self.num_agents
                
                interinsic_rewards = self.CalculateIntrinsicRewards(np.expand_dims(next_obs, axis = 0))

            else:

                next_obs, rewards, dones = [], [], []

                for agent_index, agent in enumerate(self.env.agents):

                    self.env.agent_selection = agent

                    self.env.step(action[0][agent_index][0]) # (1 x num_agents X 1)
                
                for agent_index, agent in enumerate(self.env.agents): 

                    self.env.agent_selection = agent

                    next_observation, reward, done, _ = self.env.last()

                    next_obs.append(next_observation)

                    rewards.append([reward])

                    dones.append(done)
                    
                reward = np.sum(rewards)
                
                rewards = [[reward]] * self.num_agents
                    
                interinsic_rewards = self.CalculateIntrinsicRewards(np.expand_dims(next_obs, axis = 0))

            #img.set_data(self.env.render(mode='rgb_array')) # just update the data

            #display.display(plt.gcf())
            
            #clear_output(wait=True)
            
            #time.sleep(1)
            
            #pdb.set_trace()
            
            done_env = np.all(dones)

            episode_reward.append( sum(np.array(rewards)) )

            reward = self.Unsqueeze(rewards, dim = 0)
            
            intrinsic_reward = self.Unsqueeze(interinsic_rewards, dim = 0)

            done = self.Unsqueeze(self.Unsqueeze(dones, dim = 1), dim = 0)

            done_env = done_env.reshape(1,1,1)

            if done_env.item() == 1 and evaluation:
                                     
                self.finished_first_eps = False
                
                episode_reward = np.sum(episode_reward)
                
                return episode_reward / 100
            
            next_obs = self.Unsqueeze(next_obs, dim = 0)
            
            #get state and next_state  -> we mix second and third dimenstion together to combine obs for creating state
            
            state = rearrange(obs, "d0 d1 d2 -> d0 (d1 d2)")

            state = self.Unsqueeze(state, dim = 0)
            
            next_state = rearrange(next_obs, "d0 d1 d2 -> d0 (d1 d2)")

            next_state = self.Unsqueeze(next_state, dim = 0)
            
            # strore current step transition to buffer
            
            if not evaluation:
            
                self.transition = self.transition._replace(obs = obs, next_obs = next_obs, state = state,
                                        next_state = next_state, action = action, action_onehot = action_onehot, 
                                        reward = reward, done = done, logit = logit, intrinsic_reward = intrinsic_reward)

                transitions.append(self.transition)
            
            #preperation next_step
            
            obs = next_obs
            
            # keep observation and state for next 
            
            if not evaluation and not ea:
                                     
                self.total_step += 1
                            
                if self.total_step % self.epsilon_interval == 0:
                            
                    #update epsilon

                    self.UpdateEpsilon()
                    
            # train curiosity driven network (predictor network)
            
            self.trainer.TrainPredictorNetwork(next_obs)
                                                            
        # train
            
        if train == True:
                                     
            if (warmup == False) and (self.total_episodes % self.train_interval == 0):

                    self.ActiveTrainMode(actor = True)

                    self.ActiveTrainMode(critic = True)
                                     
                    self.train_step += 1
                                     
                    data_batch = self.Buffer.sample()
                    
                    self.trainer.TrainCritic(input_batch = data_batch, gamma = self.gamma,
                                             train_step = self.train_step, epsilon = self.epsilon)
                                     
                    self.trainer.TrainActor(input_batch = data_batch, train_step = self.train_step,
                                            epsilon = self.epsilon, requlirization = self.requlirization)
                                    
                        
                        
                    
                    
        # save buffer  
        
        if isinstance(self.env, normalwrapper):        
           
            if not evaluation and not ea and np.sum(episode_reward) >= -1 * np.math.inf:    
                                             
                self.Buffer.store(transitions)  

            if not evaluation and ea and np.sum(episode_reward) >= -1 * np.math.inf:  

                self.Buffer.store(transitions)
        else:
            
                if not evaluation and not ea:
                    
                    self.Buffer.store(transitions)  
                
                if not evaluation and ea:

                    self.Buffer.store(transitions)  
                    

                     

        if not warmup and not ea:

            self.total_episodes += 1


        episode_reward =  np.sum(episode_reward)
              
        return episode_reward
            
              
    def CreateEnv(self, env_name, mode = "very_easy"):
        
        """
         
        create environment for agent to intract

        Args:
        
            env_name: environment name --> possible choices (RWARE, MPE)
            
            mode: for RWARE env --> possible choices (very_easy, easy, medium, hard)

        Returns:
        
            env: env object (agents intract with this environment)

        """
        
        # env name must be <RWARE> or <MPE>
        
        if env_name == "RWARE":
            
            if mode == "very_easy":
                
                env = gym.make("rware-tiny-" + str(self.num_agents) + "ag-v1")
                
            if mode == "easy":
                
                env = gym.make("rware-small-" + str(self.num_agents) + "ag-v1")
                
            if mode == "medium":
                
                env = gym.make("rware-medium-" + str(self.num_agents) + "ag-v1")
                
            if mode == "hard":
                
                env = gym.make("rware-large-" + str(self.num_agents) + "ag-v1")
                
        elif env_name == "MPE":

            env = simple_spread_v2.env(N = self.num_agents, max_cycles=25, local_ratio = 0)
            
            #env = MPEEnv(self.num_agents, self.num_landmarks) 

        elif env_name == "PressurePlate":

            env = gym.make('pressureplate-linear-4p-v0')
            
        return env
    
    
    def CalculateIntrinsicRewards(self, next_obs):
            
        """
         
        calculate intrinsic rewards

        Args:
        
            next_obs: curiosity driven networks networks inputs (next_observations of current step)

        Returns:
        
            return: interinsic_reward

        """
        
        inputs = self.ToTensor_(next_obs)
        
        inputs = rearrange(inputs, "d0 d1 d2 -> (d0 d1 d2)")
        
        intrinsic_rewards = []
        
        criterion = torch.nn.MSELoss()
        
        main_features = self.trainer.target_network(inputs)
        
        predicted_features = self.trainer.predictor_network(inputs)
        
        loss = criterion(main_features, predicted_features)
        
        intrinsic_reward = self.intrinsic_reward_scale * loss.item()
        
        intrinsic_rewards.append([intrinsic_reward])
        
        intrinsic_rewards = intrinsic_rewards * self.num_agents
        
        return intrinsic_rewards
    
    
    def UpdateEpsilon(self, evaluation = False):
        
        """
         
        update epsilon (epsilon greedy method for exploraton effectively)

        Args:
        
            evaluation: are we in evaluation mode or not

        Returns:
        
            no return: update epsilon

        """
                                     
        if evaluation:
                  
            self.epsilon = self.eval_epsilon
                                     
            return
            
        if self.main_epsilon > self.min_epsilon:

            self.main_epsilon = self.main_epsilon - self.epsilon_decrease_rate

        else:

            self.main_epsilon = self.main_epsilon

        self.epsilon = self.main_epsilon
                                     
                                     
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
                                     
    

    
    
    
    def ActiveTrainMode(self, actor = False, critic = False):
        
        """
         
        switch to train mode in pytorch network

        Args:
        
            actor: True or False --> True means switch actor network to train mode 
            
            critic: True or False --> True means switch critic network to train mode 

        Returns:
        
            no return

        """
        
        if actor:
            
            self.trainer.actor.train()
            
        if critic:
            
            self.trainer.critic.train()
        
        
    def ActiveEvalMode(self, actor =  False, critic = False):
        
        """
         
        switch to evaluation mode in pytorch network

        Args:
        
            actor: True or False --> True means switch actor network to evaluation mode 
            
            critic: True or False --> True means switch critic network to evaluation mode 

        Returns:
        
            no return

        """
        
        if actor:
            
            self.trainer.actor.eval()
            
        if critic:
            
            self.trainer.critic.eval()
            
            
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
        
        for element in inpt:
            
            yield torch.cat(element, dim = dim)
            
            
    def TensorStack(self, *inpt, dim = 0):
        
        """
         
        stack tensors along one axis 

        Args:
        
            inpt: list of tensors

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
            
            result = inpt.squeeze(dim = dim)
                                     
        else:
            
            result = np.expand_dims(inpt, axis = dim)
                                     
        return result
    
    def chunk(self, input_, divition_number = 10):
        
        """
         
        create chunk of input data

        Args:
        
            division_number : how many chunk input should be

        Returns:
        
            chubked_data

        """
        
        chunked_data = np.split(input_, divition_number)
    
        return chunked_data
