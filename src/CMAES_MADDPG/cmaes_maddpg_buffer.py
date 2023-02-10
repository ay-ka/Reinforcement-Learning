import numpy as np
from collections import namedtuple
from cmaes_maddpg_utils import  SampleError


class Buffer:
    
    def __init__(self, buffer_capacity : int = 50000, batch_size : int = 16, episode_limit : int = 25):
        
        """
        buffer to store transition.

        Args:
        
            buffer_capacity: number of transition can be stored.
            
            batch_size: number of transition to select when sampling.
            
        other parameters:
        
            episode_record: number of transition which are stored up to now
            
            store_index: index indicating where new transition should be stored in buffer (buffer index)

        Returns:
        
            no return --> initialize buffer

        """
        
        self.buffer_capacity = buffer_capacity
        
        self.batch_size = batch_size
        
        self.episode_record = 0
        
        self.store_index = 0
        
        self.episode_limit = episode_limit
        
        self.buffer = dict()
        
        self.experience = namedtuple("Experience", field_names=["obs", "state", "action", "next_obs",
                                           "next_state", "done", "reward", "logit", "action_onehot", "intrinsic_reward"])
        
        #initialize buffer
        
        for episode in range(self.buffer_capacity):
    
            self.buffer[episode] = []
    
            for limit in range(self.episode_limit):
        
                self.buffer[episode].append(self.experience(0,0,0,0,0,0,0,0,0,0))
        

        
        
        
        
    def store(self, transition):
                    
        """
        buffer to store transition.

        Args:
        
            buffer_capacity: number of transition can be stored.
            
            batch_size: number of transition to select when sampling.
            
            episode_limit: number of step is allowed to take in one episode
            
        other parameters:
        
            episode_record: number of transition which are stored up to now
            
            store_index: index indicating where new transition should be stored in buffer (buffer index)

        Returns:
        
            no return --> initialize buffer

        """
           
        #store transition on buffer
        
        self.buffer[self.store_index][:] = transition
        
        
        
        #update store index
        
        self.episode_record += 1
            
        self.store_index = self.episode_record % self.buffer_capacity
        
        
    def sample(self) -> dict:
        
        """
        sample batch of transition from buffer

        Args:
        
            no argument

        Returns:
        
           batch: dictionary of transition returned as batch

        """
        
        batch = {}
        
        if self.episode_record < self.buffer_capacity:
            
            if self.episode_record < self.batch_size:
                
                indicies = list(self.buffer.keys())[0 : self.episode_record]
                
                batch_indicies = np.random.choice(indicies, size = self.episode_record)
                
                
            else:
                
                indicies = list(self.buffer.keys())[0 : self.episode_record]
                
                batch_indicies = np.random.choice(indicies, size = self.batch_size)
                
        else:
            
            indicies = list(self.buffer.keys())
            
            batch_indicies = np.random.choice(indicies, size = self.batch_size)
        
        #take experience batch
        
        for index, targeted_index in enumerate(batch_indicies):
            
            key = "transition_" + str(targeted_index) + "_idx_" + str(index)
            
            batch[key] = self.buffer[targeted_index] #befor it was indexs
            
        return batch


class PrioritizedBuffer:
    
    
    def __init__(self, buffer_capacity : int = 100, batch_size : int = 32):
        
        """
        Prioritized buffer to store transition.

        Args:
        
            buffer_capacity: number of transition can be stored.
            
            batch_size: number of transition to select when sampling.
            
        other parameters:
        
            episode_record: number of transition which are stored up to now
            
            store_index: index indicating where new transition should be stored in buffer (buffer index)
            
            alpha: constant --> priorizitation intensity --> refer to Prioritized Experience Reply Paper
            
            betha: constant --> refer to Prioritized Experience Reply Paper (not implemented here)
            
            epsilon: preventing priorities to be zero --> refer to Prioritized Experience Reply Paper

        Returns:
        
            no return --> initialize buffer

        """
        
        self.buffer_capacity = buffer_capacity
        
        self.batch_size = batch_size
        
        self.alpha = 0.5
        
        self.betha = .75 # --> relate to calculating weights not implemented here
        
        self.epsilon = 1e-6
        
        self.episode_record = 0
        
        self.store_index = 0

        self.per_attributes = namedtuple("Data", field_names=["priority", "probability", "weight"])
        
        
        self.experience = namedtuple("Experience", field_names=["obs", "state", "action", "next_obs",
                                           "next_state", "done", "reward", "logit", "action_onehot", "intrinsic_reward"])
        

        #initialize buffer
        
        for episode in range(self.buffer_capacity):
    
            self.buffer[episode] = []
    
            for limit in range(self.episode_limit):
        
                self.buffer[episode].append(self.experience(0,0,0,0,0,0,0,0,0,0))
            
            
        
        self.PER_attributes = {index: self.per_attributes(0, 0, 0) for index in range(self.buffer_capacity)}
        
        
    def store(self, transition, episode_error: float):
        
        """
        store one transition inside buffer

        Args:
        
            transition: a transition taken from environment when agent take one step ("obs", "state", "action", "next_obs",
                                           "next_state", "done", "reward", "logit", "action_onehot")
                                           
            error: difference between current value of state and target for that state (like TD error)

        Returns:
        
            no return --> (action of storing one transition inside buffer and updating related parameters)

        """
         
        #store transition on buffer
        
        self.buffer[self.store_index][:] = transition

        
        #update per atributes for all transitions (priority, probability, weights)
        
        self.PER_attributes[self.store_index] = self.PER_attributes[self.store_index]._replace(priority = 
                                                                                        episode_error + self.epsilon)
        
        #update probabilities
        
        self.UpdateBuffer()
        
        
        #update store index
        
        self.episode_record +=1
        
        self.store_index = self.episode_record % self.buffer_capacity
        
    
    def sample(self) -> dict:
        
        """
        sample batch of transition from buffer

        Args:
        
            no argument

        Returns:
        
           batch: dictionary of transition returned as batch

        """
        
        batch = {}
        
        if self.episode_record < self.buffer_capacity:
            
            #get candidate indicies
            
            indicies = list(self.buffer.keys())[0 : self.episode_record]

            attr_list = list(self.PER_attributes.values())[0 : self.episode_record]
            
            #get probabilities for candidate indicies

            probabilities = [attr_list[index].probability for index in range(len(attr_list))]
            
            #without storing any transition calling sample method throw error

            if sum(probabilities) == 0:

                raise SampleError("you should store one transition first")
                
            #take targeted indicies for current batch
            
            if self.episode_record < self.batch_size:
                
                batch_indicies = np.random.choice(indicies, size = self.episode_record, p = probabilities)    
                
            else:
                
                batch_indicies = np.random.choice(indicies, size = self.batch_size, p = probabilities)
                 
        else:
            
            #get candidate indicies
            
            indicies = list(self.buffer.keys())

            attr_list = list(self.PER_attributes.values())
            
            #get probabilities for candidate indicies

            probabilities = [attr_list[index].probability for index in range(len(attr_list))]
            
            #without storing any transition calling sample method throw error

            if sum(probabilities) == 0:

                raise SampleError("you should store one transition first")
                
            #take targeted indicies for current batch
                
            batch_indicies = np.random.choice(indicies, size = self.batch_size, p = probabilities)
        
        #take experience batch
        
        for index, targeted_index in enumerate(batch_indicies):
            
            key = "transition_" + str(targeted_index) + "_idx_" + str(index)
            
            batch[key] = self.buffer[targeted_index] #befor it was indexs
            
        return batch
    
    
    def UpdateBuffer(self):
        
        """
        Update selection probabilities of each transition inside buffer

        Args:
        
            no argument

        Returns:
        
            no return --> (action of updating selection probabilities of each transition inside buffer)

        """
        
        attr_list = list(self.PER_attributes.values())
        
        #get priorities 
        
        priorities = [attr_list[index].priority for index in range(self.buffer_capacity)]
        
        #compute probabilities (wrt paper)
        
        nominator = list( map(self.Power, priorities) ) 
        
        dinaminator = list( np.sum(nominator).reshape(-1) ) * len(nominator)
        
        probabilities = list( map(self.Devide, nominator, dinaminator) )
        
        #add probabilities to buffer
        
        for index in range(self.buffer_capacity):
            
            self.PER_attributes[index] = self.PER_attributes[index]._replace(probability = probabilities[index])
            
            
    def Power(self, inpt):
        
        """
         
        do power operation on input

        Args:
        
            inpt: a float number (priority in here)

        Returns:
        
            return --> power two of inputed number (power two of prioritie)

        """
            
        result = inpt ** self.alpha
    
        return result
    
    
    def Devide(self, nominator, dinaminator):
        
        """
         
        devide nominator to denaminator

        Args:
        
            nominator: a float number as nominator of a division
            
            denamionator : afloat mnumber as denaminator of division

        Returns:
        
            result --> division of numinator to dinaminator

        """
        
        result = nominator / dinaminator
        
        return result



