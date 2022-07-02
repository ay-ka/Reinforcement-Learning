from envs_repo.gym_wrapper import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, args):
        """
        A general Environment Constructor
        """
        self.env_name = env_name

        #Dummy env to get some macros
        dummy_env = self.make_env(args)
        self.is_discrete = dummy_env.is_discrete
        self.obs_dim = dummy_env.obs_dim
        self.action_dim = dummy_env.action_dim


    def make_env(self, args):
        """
        Generate and return an env object
        """
        env = GymWrapper(self.env_name, args)
        return env



