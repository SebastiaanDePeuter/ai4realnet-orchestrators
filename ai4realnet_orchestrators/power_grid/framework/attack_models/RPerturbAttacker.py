import pandas as pd
import numpy as np
import copy
from perturbation_agents.random_perturb_agent import RandomPerturbationAgent
from Grid2OpUtils import Grid2OpUtils
from attack_models.BaseAttackerClass import BaseAttackerClass


class RPerturbAttacker(BaseAttackerClass):
    """
    Random Perturbation Attacker.
    This wrapper class applies adversarial perturbation to environment observations using a random perturbation strategy.
    Acts as an interface between the environment and the agent modifying observations.

    Attributes:
        model_name: Identifier for the attacker model.
        pickle_file: Default filename for serialization.
        env: Environment reference
        model: Perturbation model agent
    """
    def __init__(self, env, **kwargs):
        """
        Args:
            env: Environment reference
        """
        self.model_name = "RPerturbAttacker"
        self.pickle_file = "r_perturb_attacker.pkl"
        self.env = env

        # Random perturbation model working on the environments observation space
        self.model = RandomPerturbationAgent(self.env.observation_space, **kwargs)


    def perturb(self, obs):
        """
        Apply random adversarial perturbation to an observation.

        Args:
            obs: Original observation
        
        Returns:
            obs_perturbed: Perturbed observation created by the model.
        """
        obs_t = copy.deepcopy(obs) # Work on a copy to avoid mutating the original observation
        obs_perturbed = self.model.perturb(obs_t)
        return obs_perturbed
