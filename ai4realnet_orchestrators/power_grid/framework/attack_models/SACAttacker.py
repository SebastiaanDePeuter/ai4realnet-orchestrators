import pandas as pd
import copy
import torch
from stable_baselines3 import SAC
from attack_models.BaseAttackerClass import BaseAttackerClass


class SACAttacker(BaseAttackerClass):
    """
    Soft Actor-Critic (SAC) Attacker
    This attacker perturbs environment observations using a pretrained SAC model. The perturbation is scaled by a configurable factor to control intensity

    Args:
        model_path: Path t the pretrained SAC model file
        factor: Scaling factor applied to perturbation
        model_name: Identifier for the attacker model
        pickle_file: Filename for serialization

    Attributes:
        model_path: Path to the saved SAC model.
        factor: Perturbation scaling factor.
        model_name: Identifier for the pretrained SAC model.
        pickle_file: Default filename for serialization.
        model: Perturbation model trained
        entropy_threshold (optional): Entropy threshold used as a heuristic to evaluate perturbation diversity (default = 0.85)
    """
    def __init__(self, model_path, factor, model_name, pickle_file, entropy_threshold=0.85):
        self.model_path = model_path
        self.factor = factor
        self.model_name = model_name
        self.pickle_file = pickle_file
        self.model = SAC.load(self.model_path)
        self.entropy_threshold = entropy_threshold


    def perturb(self, obs):
        """
        Apply SAC adversarial perturbation to an observation.
        The perturbation is predicted by the SAC policy and scaled by the factor. Only selected attributes are update.

        Args:
            obs: Original environment observations
        
        Returns:
            obs_perturbed: Perturbed copy of the observations.
        """
        obs_perturbed = copy.deepcopy(obs) # Work on a copy to avoid mutating the original observation
        
        # Attributes index setup for mapping back perturbed values
        _, attr_start_idx = self._attribute_setup(obs_perturbed)
        
        # Get perturbation vector from SAC policy
        perturbation, _ = self.model.predict(obs_perturbed.to_vect(), deterministic=True)
        
        # Apply perturbation scaled by a factor
        obs_perturbed._vectorized *= (1 + (perturbation * (self.factor / 10)))
        
        # Recompute rho
        obs_perturbed.rho = obs_perturbed.to_vect()[attr_start_idx["rho"]:attr_start_idx["rho"] + len(obs_perturbed.rho)]
        return obs_perturbed
    
