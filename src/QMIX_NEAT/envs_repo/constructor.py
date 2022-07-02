from src.QMIX_NEAT.envs_repo.gym_wrapper import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, num_agents):
        """
        A general Environment Constructor
        """
        self.env_name = env_name
        self.num_agents = num_agents

        dummy_env = self.make_env()
        self.is_discrete = dummy_env.is_discrete
        self.obs_dim = dummy_env.obs_dim
        self.action_dim = dummy_env.action_dim


    def make_env(self):
        """
        Generate and return an env object
        """
        env = GymWrapper(self.env_name, num_agents = self.num_agents, mode = "very_easy")
        return env



