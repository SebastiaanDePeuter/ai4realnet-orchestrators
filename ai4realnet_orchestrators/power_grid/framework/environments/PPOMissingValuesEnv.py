import gym
from gym import spaces
import torch
import numpy as np
import pandas as pd
import copy
from environments.BaseAdversarialEnv import BaseAdversarialEnv

class MissingValuesEnv(BaseAdversarialEnv):
    """
    Custom Gym environment that simulates adversarial attacks by removing parts of the observation vector before passing it to the target agent.

    """
    def __init__(self, target_agent, env, log_dir):
        super(MissingValuesEnv, self).__init__(target_agent, env, log_dir)
        
        # Define action space specific to this environment
        self.action_space = spaces.MultiBinary(len(self.current_obs))

    def _apply_perturbation(self, perturbation):
        """
        Apply perturbation by removing observation values.
        
        Args:
            perturbation: Binary mask over the observation vector
            
        Returns:
            Perturbed observation object
        """
        obs_perturbed = copy.deepcopy(self.current_obs_obj)
        obs_perturbed._vectorized *= perturbation
        return obs_perturbed

    def save_logs(self):
        """
        Save training results to CSV.

        ### Check if the file name is correct.
        """
        df = pd.DataFrame(self.results[1:], columns=self.results[0])
        df.to_csv(self.log_dir, index=False)
        print(f"Training results for PPOMissingValuesEnv.py saved to {self.log_dir}")
        df = pd.DataFrame(self.all_perturbations[1:], columns=self.all_perturbations[0])
        df.to_csv(self.log_dir.replace(".csv", "_perturbations.csv"), index=False)
        print(f"Perturbation results for PPOMissingValuesEnv.py saved to {self.log_dir.replace('.csv', '_perturbations.csv')}")