from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pdb
import copy

import cem_maddpg_buffer
import cem_maddpg_models
import cem_maddpg_trainer
import cem_maddpg_runner
import cem_maddpg_env
import cem_maddpg_utils


class CEM:
    
    def __init__(self, network_model, rl_algorithm_runner, parameters_lower_bound = -1000, num_parents_prob = .3,
                 max_iteration = 2, parameters_upper_bound = 100,  num_genomes = 40, initial_sigma = 1e-3,
                 symmetry = True, initial_epsilon = 1e-3, epsilon_limit = 1e-5, tau = 0.95):
        
        """ CEM algorithm main class

        Args:
        
            network_model: nueral network model (from reinforcement part --> actor or critic) (pytorch model)
            
            rl_algorithm_runner: reinforcement learning main runner which encompass all other functioality of 
                                 reinforcement learning like (collecting data, evaluation, trainer, buffer)
            
            parameters_lower_bound: minimum value parameters (gens) can take inside genome
            
            parameters_upper_bound: maximum value parameters (gens) can take inside genome
            
            max_iteration: number of iteration (generation) to run algorithm
            
            num_genomes: how many member is in population
            
            initial_sigma: initial standard deviation for calculation initial covariances
            
            num_parent_prob: rate of parent number to population size in each generation to create off springs
            
            symmetry: devide population to 2 part which are symmetry to each other --> parameters in second part of 
                        population is negative of first part of population
                        
            tau: constant for calculating epsilon (noise part in cov calculation) --> (refer to paper)
            
            initial_epsilon: small number to add noise in our covariance calculation
            
            epsilon_limit: minimum number epsilon can take (with progressing noise term will decrease)
                        

        Returns:
        
            no return --> initializing (create initial CEM population also store current best genome)

        """
        
        
        # defining genome network settings
        
        self.network = network_model
        
        self.rl_algorithm_runner = rl_algorithm_runner
        
        self.activation_func = [nn.Tanh(), nn.ReLU(), nn.ELU(), nn.Sigmoid(), nn.SELU()]
        
        self.last_layer_act_func = [nn.ReLU(), nn.ELU()]
        
        self.lst_activation_parameters = []
        
        self.lst_activation_parameters.append(self.activation_func)
        
        self.lst_activation_parameters.append(self.last_layer_act_func)
        
        self.length_activation_parameters = len(self.lst_activation_parameters)
        
        weight_list = CEM.GetWeightsRL(self.network)
        
        rl_genome_parameters = CEM.CreateRLGenome(weight_list)
        
        self.number_of_parameters = len(rl_genome_parameters)
        
        
        # CEM settings
        
        self.mu = np.array(rl_genome_parameters)
        
        self.old_mu = np.array(rl_genome_parameters)
        
        self.sigma = initial_sigma
        
        self.epsilon = initial_epsilon
        
        self.epsilon_limit = epsilon_limit
        
        self.tau = tau
        
        self.cov = self.sigma * np.ones(self.number_of_parameters)
        
        self.symmetry = symmetry
        
        if self.symmetry:
            
            assert (num_genomes % 2 == 0), "Population size must be even"
            
            self.num_parents = num_genomes // 2
            
        else:
            
            self.num_parents = round(num_genomes * self.num_parents_prob)
            
            
            
        
        # problem settings
        
        self.pop_size  = num_genomes
        
        self.num_parents_prob = num_parents_prob
        
        self.max_iteration =  max_iteration
        
        self.upper_bound = parameters_upper_bound
        
        self.lower_bound = parameters_lower_bound
        
        self.genome = namedtuple("Genome", field_names = ["parameters", "fitness"])
        
        self.genomes = {"genome_" + str(genome_index) : self.genome([], None) for genome_index in range(self.pop_size)}
        
        self.parents_parameters = np.zeros([self.num_parents, self.number_of_parameters])
        
        self.best_fitness = np.zeros([self.max_iteration])
        
        
        
        # create initial population 
        
        self.CreatePop()
        
        self.CostFunction()
                        
        self.SortPop()
               
            
            
        # calculate probs 
        
        self.CalculateProbs()
        
        
        
        # store best fitness of initial generation
        
        initial_generation_index = 0
        
        self.best_fitness[initial_generation_index] = self.genomes["genome_0"].fitness # population is sorted inside CreateInitPop()
        
        self.all_fitness = []
        
        
    def MainLoop(self):
        
        """ main loop for CEM (evolveing through generations)
        
        Args:
        
            no argument

        Returns:
        
            no return --> evolution  through generations

        """
        
        for iteration in range(1, self.max_iteration):
            
            #calculate new Mean
            
            self.CalculateMeans()
            
            #calculate new covariance matrix
            
            self.CalculateCovariance()
            
            #create new genomes
            
            self.CreatePop()
            
            #evaluate new genomes
            
            self.CostFunction()
            
            #sort new_genomes
            
            self.SortPop()
            
            # save best fitness
            
            self.best_fitness[iteration] = self.genomes["genome_0"].fitness

            self.all_fitness.append(self.genomes["genome_0"].fitness)
            
        
    def CalculateMeans(self):
        
        """ calculate mean
        Args:
        
            no argument

        Returns:
        
            no return --> calculate mean --> (equation in paper)

        """
        
        self.old_mu = self.mu
        
        self.parents_parameters = np.array([self.genomes["genome_" + str(genome_index)].parameters for genome_index 
                                                                                    in range(self.num_parents)]) # pop is sorted in last operation
        
        self.mu = self.probs @ self.parents_parameters
    
    
    def CalculateCovariance(self):
        
        """ calculate covarianes

        Args:
        
            no argument

        Returns:
        
            no return --> calculate covariance --> (equation in paper)

        """
        
        self.epsilon = self.epsilon * self.tau + (1 - self.tau) * self.epsilon_limit
        
        z = (self.parents_parameters - self.old_mu)
        
        self.cov = ( 1 / self.num_parents ) * ( self.probs @ (z * z) + self.epsilon * np.ones(self.number_of_parameters) )

        
    def CreatePop(self):
        
        """ create population with respect to current mean and covariances

        Args:
        
            no argument

        Returns:
        
            no return --> create population and set 

        """
        
        if self.symmetry:
            
            normal_random_dist = np.random.randn(self.num_parents, self.number_of_parameters) # to the number of parents
            
            normal_random_dist = np.concatenate([normal_random_dist, - normal_random_dist]) # to the number of pop_size

        else:
            
            normal_random_dist = np.random.randn(self.pop_size, self.number_of_parameters) # to the number of pop_size

        population = self.mu + normal_random_dist * np.sqrt(self.cov)
        
        # select activation function
        
        population[:, -2] = np.random.randint(0, len(self.activation_func), size = self.pop_size)
        
        population[:, -1] = np.random.randint(0, len(self.last_layer_act_func), size = self.pop_size)
        
        self.genomes = {"genome_" + str(genome_index) : self.genome(parameters = genome.tolist(), fitness = None) 
                                                for genome_index, genome in enumerate(population)}
        
    def GetDistParams(self):
        
                
        """ get mean and covariance 

        Args:
        
            no argument

        Returns:
        
            no return --> get mean and covariance

        """
        
        return self.mu, self.cov
        
    @staticmethod   
    def GetWeightsRL(network_model):
        
        """ get weights list from reinforcement nurral network (critic or hypernets weights)

        Args:
        
            network_model: nueral network model (from reinforcement part --> actor or critic) (pytorch model)

        Returns:
        
            weights_lst: list of all weights inside nueral network model used in reinforcement learning 
                         inputed as argument to this function

        """
        
        weights_lst = []

        for layer_name, weights in network_model.state_dict().items():

            weights = weights.numpy()

            weights = weights.flatten()

            weights = weights.tolist()

            weights_lst = weights_lst + weights
            
        return weights_lst
            
    @staticmethod     
    def CreateRLGenome(weights_lst):
        
        """ create one genome from a weight list 

        Args:
        
            weights_lst: list of weights (nueral network weights as a list)
                         

        Returns:
        
            genome_parameters: a list of float numbers (list of weights --> nueral network weights +
                                                                    other functioloty of network like activatio function)

        """
        
        genome_parameters = weights_lst
        
        # first zero for hidden layers activation and second_zero for last_layer activation function
        
        select_activation = [0, 0]
        
        # create rl_wolf
        
        genome_parameters = genome_parameters + select_activation
        
        return genome_parameters
    
    
    def CalculateProbs(self):
        
        """ calculate selection probabilities with respect to genome ranks

        Args:
        
            no_argument

        Returns:
        
            no return --> calculate probabilities and set to each genome (parent genomes)

        """
        
        weights = np.array([np.log(self.num_parents + 1) / parent_index for parent_index in range(1,
                                                                                            self.num_parents + 1)])
        
        self.probs = (weights / np.sum(weights))
        
        self.probs = self.probs.reshape(1, -1)
        

                        
    def CostFunction(self, genome_parameters = None):
        
        """ our problem cost function for CEM algorithm

        Args:
        
            genome_parameters: list of float number which represent a genome (can be used to craete nueral network) -->
                            if is not passed calculate all genomes fitness inside main population

        Returns:
        
            fitness: fitness of nueral network creatd from genome_parameters inputed as argument  (if genome_parameters not 
                     passed there is no return and just calculate all genomes fitness inside main population)

        """
        
        if genome_parameters != None:
            
            self.GetGenomefNetwork(genome_parameters)
            
            fitness = self.Evaluate()
            
            return fitness
        
        else:
            
            for genome_index, genome in self.genomes.items():
                
                self.GetGenomeNetwork(genome.parameters)
                
                fitness = self.Evaluate()
                
                self.genomes[genome_index] = self.genomes[genome_index]._replace(fitness = fitness)
                        
                        
    
                        
    def GetGenomeNetwork(self, genome_parameters):
        
        """ created nueral network from parameters of an genome

        Args:
        
            genome_parameters: list of float number which represent an genome (can be used to craete nueral network) 
            
        Returns:
        
            no return --> set CEM class own network (self.network) to nueral network created from inputed genome_parameters

        """
        
        
        start_index, end_index, track_index, one_dim = 0, 0, 0, 0

        for layer_name , weights in self.network.state_dict().items():

            start_index = start_index + track_index

            layer_first_dim = self.network.state_dict()[layer_name].shape[0]

            try:

                layer_second_dim =  self.network.state_dict()[layer_name].shape[1]

            except:

                one_dim = True

                layer_second_dim = 1

            track_index = layer_first_dim * layer_second_dim

            end_index = end_index + track_index

            if one_dim :

                self.network.state_dict()[layer_name].data[:] = torch.tensor(genome_parameters[start_index :
                                                                                                 end_index])

            else:

                self.network.state_dict()[layer_name].data[:] = torch.tensor(genome_parameters[start_index : 
                                                                                    end_index]).reshape(
                                                                                    layer_first_dim, layer_second_dim)

            one_dim = False
            
        self.network.hidden_layer_act, self.network.last_layer_act = self.activation_func[int(genome_parameters[-2])],                                                                      self.last_layer_act_func[int(genome_parameters[-1])] 
                        
                        
    def SortPop(self):
        
        """ sort population with respect to their fitness

        Args:
        
            no argument

        Returns:
        
            no return --> sort population with respect to their fitness

        """
        
        fitneses = []

        for genome_index, genome in self.genomes.items():

            fitneses.append((genome.fitness, genome.parameters, genome_index))

        sorted_fitneses = list(reversed(sorted(fitneses)))

        for index, (genome_fitness, genome_parameters, past_genome_index) in enumerate(sorted_fitneses):
            
            genome_index = "genome_" + str(index)

            self.genomes[genome_index] = self.genomes[genome_index]._replace(parameters = genome_parameters, 
                                                                fitness = genome_fitness)
            
            
    def Evaluate(self, genome_network_model = None):
        
        """ evaluate one genomes by passing genome (parameters inside it as weights) to reinforcement algorithm

        Args:
        
            genome_network_model: nueral network created from spesific genome (parameters inside it) (pytorch model) --> 
                               if Not given, genome_network_model is set to CEM class own network which is initialized 
                               by spesific genome parameters

        Returns:
        
            fitness: fitness of targeted genome

        """
        
        if genome_network_model is None:
            
            genome_network_model = self.network
        
        else:
            
            genome_network_model = genome_network_model
        
        self.rl_algorithm_runner.trainer.actor_ea = genome_network_model

        fitneses = []
        
        for iteration in range(5):
        
            fitness = self.rl_algorithm_runner.collector(evaluation = True, ea = True)
            
            fitneses.append(fitness)

        fitness = np.mean(fitneses)
        
        return fitness
            
        
    def InsertRL(self, rl_network):
        
        """ transfer reinforcement learning network model to (EA) algorithm and insert it to population replacement of 
            current generation's population worst member

        Args:
        
            rl_network: nueral network model from reinforcement learning part (pytorch model) (actor or critic or ...)

        Returns:
        
            no return --> convert rl network model to a genome and replace it with current generation worst member 

        """
        
        # get position (parameters as list)
        
        weight_lst = CEM.GetWeightsRL(rl_network)
        
        rl_genome_parameters = CEM.CreateRLGenome(weight_lst)
        
        # get fitness
        
        rl_genome_fitness = self.Evaluate(rl_network)
        
        #sort genome pop
        
        self.SortPop()
        
        # worst genome index
        
        worst_genome_index = len(self.genomes.keys())
        
        worst_genome_index = "genome_" + str(worst_genome_index - 1) # because is sorted
        
        #insert to population
        
        self.genomes[worst_genome_index] = self.genomes[worst_genome_index]._replace(parameters = rl_genome_parameters, 
                                                                                  fitness = rl_genome_fitness)
        
        
        # sort solution archive
        
        self.SortPop()


    def InsertSolutions(self, num_migration):
        
        """ insert network saved in rl part to fill solution archive

        Args:
        
            num_migration : number of network should be transfered

        Returns:
        
            no return --> fill solution archive with rl network

        """ 
        
        
        
        if num_migration < len(self.genomes.keys()):
            
            # we add 1 because in range function we have started from number 1 and it makes our range 1 number samller (there is also one place for untransfered network in 
            # archive)
        
            new_num_migration = num_migration + 1
                
        else:
            
            # in here we want to give place for one network not transfered from rl in solutio archive
            
            new_num_migration = len(self.genomes.keys())
            
        for saved_net_index, genome_index in enumerate(range(1, new_num_migration)):
            
            # reverse inputed network order (most recenet saved inserted) --> network_n, network_n-1, ..... not network_0, network_1, ...
            
            saved_net_index = num_migration - saved_net_index
            
            # load targeted network
                    
            host_network = copy.deepcopy(self.network)
            
            host_network.load_state_dict(torch.load("result_actor_model_" + str(saved_net_index)))
            
            # get position (parameters as list)
        
            weight_lst = CEM.GetWeightsRL(host_network)
            
            rl_genome_parameters = CEM.CreateRLAnt(weight_lst)
            
            # get fitness
        
            rl_genome_fitness = self.Evaluate(host_network)
            
            #insert to population
        
            self.genomes["genome_" + str(len(self.genomes.keys()) - genome_index)] = self.genomes[
                                                    "genome_" + str(len(self.genomes.keys()) - genome_index)]._replace(parameters = rl_genome_parameters, 
                                                     fitness = rl_genome_fitness) 
                                                    
                                                    
            self.SortPop()       

