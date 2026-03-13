import pandas as pd
import copy
from stable_baselines3 import PPO
from attack_models.BaseAttackerClass import BaseAttackerClass

class PPOAttacker(BaseAttackerClass):
    """
    Attacker that uses PPO (Proximal Policy Optimization) model to perturb environment observations.
    It serves as an interface between the environment and the PPO model.
    Args:
        model_path: Path to the pre-trained PPO model file
        model_name: Name identifier for this attacker
        pickle_file: Filename used for saving
    """
    def __init__(self, model_path, model_name="PPO", pickle_file="ppo_attacker.pkl"):
        self.model_path = model_path
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.model = PPO.load(self.model_path, device="cpu")


    def perturb(self, obs):
        """
        Apply PPO-based perturbation to an observation.

        Args:
            obs: Original environment observation.
        
        Returns:
            Perturbed observation object
        """
        obs_perturbed = copy.deepcopy(obs) # Wroks on a copy to avoid mutating the original observations

        # Build attribute indices for consistent mapping
        _, attr_start_idx = self._attribute_setup(obs_perturbed)

        # Get perturbation vector from PPO
        perturbation, _ = self.model.predict(obs_perturbed.to_vect(), deterministic=True)

        # Apply perturbation
        obs_perturbed._vectorized *= perturbation

        # Recompute Rho
        obs_perturbed.rho = obs_perturbed.to_vect()[attr_start_idx["rho"]:attr_start_idx["rho"] + len(obs_perturbed.rho)]
        return obs_perturbed