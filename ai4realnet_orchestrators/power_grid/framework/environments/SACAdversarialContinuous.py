import gym
from gym import spaces
import torch
import numpy as np
import pandas as pd
import copy
from environments.BaseAdversarialEnv import BaseAdversarialEnv

class SACAdversarialContinuous(BaseAdversarialEnv):
    """
    Custom Gym environment that applies continuous adversarial perturbations to the observation vector before passing it to the target agent.

    Args:
        target_agent: Victim agent to be attacked
        env: Environment
        log_dir: Filepath to save training logs
        epsilon (optional): Maximum perturbation magnitude (default = 0.1)
    """
    def __init__(self, target_agent, env, log_dir, epsilon=0.1):
        super(SACAdversarialContinuous, self).__init__(target_agent, env, log_dir)
        
        self.epsilon = epsilon
        
        # Define action space specific to this environment
        self.action_space = spaces.Box(
            low=-epsilon, high=epsilon, shape=self.current_obs.shape, dtype=np.float32
        )

    def _apply_perturbation(self, perturbation):
        """
        Apply continuous perturbation to observation.
        
        Args:
            perturbation: Array containing continuous noise in [-epsilon, epsilon] per feature
            
        Returns:
            Perturbed observation object
        """
        # Ensure perturbation respect the epsilon bounds
        perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)

        # Apply multiplicative perturbation to copy of current observation
        obs_perturbed = copy.deepcopy(self.current_obs_obj)
        obs_perturbed._vectorized *= (1 + perturbation)
        return obs_perturbed

    def _handle_episode_end(self):
        """
        Handle episode end with additional logging for continuous perturbations.
        """
        print(f"Episode {self.cur_episode} finished: {self.current_obs_obj.current_step} steps with reward {self.episode_reward}")
        
        # Get the last perturbation for average calculation
        if len(self.all_perturbations) > 1:
            last_perturbation = self.all_perturbations[-1][2]  # perturbation is at index 2
            print(f"average_perturbation: {np.mean(np.abs(last_perturbation))}")

        self.results += [(self.cur_episode, self.current_obs_obj.current_step, self.episode_reward)]
        self.cur_episode += 1
        self.episode_reward = 0

    def save_logs(self):
        """
        Save training results to CSV.
        """
        df = pd.DataFrame(self.results[1:], columns=self.results[0])
        df.to_csv(self.log_dir, index=False)
        print(f"Training results for SACAdversarialContinuous.py saved to {self.log_dir}")
        df = pd.DataFrame(self.all_perturbations[1:], columns=self.all_perturbations[0])
        df.to_csv(self.log_dir.replace(".csv", "_perturbations.csv"), index=False)
        print(f"Perturbation results for SACAdversarialContinuous.py saved to {self.log_dir.replace('.csv', '_perturbations.csv')}")