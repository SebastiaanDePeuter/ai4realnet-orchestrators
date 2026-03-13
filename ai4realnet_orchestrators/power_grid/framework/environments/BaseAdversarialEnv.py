import gym
from gym import spaces
import torch
import numpy as np
import pandas as pd
import copy

class BaseAdversarialEnv(gym.Env):
    """
    Base class for adversarial environments that attack target agents by perturbing observations.
    """
    def __init__(self, target_agent, env, log_dir):
        super(BaseAdversarialEnv, self).__init__()
        self.target_agent = target_agent
        self.env = env
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment and first observation
        obs_obj = self.env.reset()
        self.current_obs_obj = obs_obj 
        self.current_obs = obs_obj.to_vect()

        # Episode tracking
        self.cur_episode = 0
        self.all_perturbations = [("episode", "step", "perturbation", "reward")]
        self.results = [("episode", "n_steps", "reward")]
        self.episode_reward = 0

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.current_obs.shape, dtype=np.float32
        )

        # Map vectorized indices to attribute names
        self.attr_list = []
        for attr in self.current_obs_obj.attr_list_vect:
            if isinstance(getattr(self.current_obs_obj, attr), np.ndarray):
                n = getattr(self.current_obs_obj, attr).shape[-1]
            else:
                n = 1
            self.attr_list += [attr] * n
            
        # Get starting index for each attribute
        self.attr_start_idx = (pd.DataFrame(self.attr_list, columns=["attr"])
                        .reset_index()
                        .groupby("attr")["index"].min()
                        .to_dict())

    def reset(self):   
        """
        Reset the environment to initial state for a new episode

        Returns:
            A copy of the vectorized initial observations np.array
        """     
        obs_obj = self.env.reset()
        self.current_obs_obj = obs_obj
        self.current_obs = obs_obj.to_vect()

        return self.current_obs.copy()

    def step(self, perturbation):
        """
        Apply a perturbation to the observation and query the target agent.
        This method should be overridden by subclasses to implement specific perturbation logic.

        Args:
            perturbation: Perturbation to apply (format depends on subclass)
        
        Returns:
            A tuple containing: 
                - New observation array
                - Adversarial reward
                - Boolean indicating whether the episode terminated
                - A dictionary with Environment information
        """
        try:     
            # Apply perturbation (to be implemented by subclasses)
            obs_perturbed = self._apply_perturbation(perturbation)

            # Recompute rho
            obs_perturbed.rho = obs_perturbed.to_vect()[self.attr_start_idx["rho"]:self.attr_start_idx["rho"] + len(obs_perturbed.rho)]

            # Query victim agent with perturbed observation
            action = self.target_agent.act(obs_perturbed, 0, False)
            new_obs_obj, reward, done, info = self.env.step(action)

            #Update environment state
            self.current_obs_obj = new_obs_obj
            self.current_obs = new_obs_obj.to_vect()

            # Compute adversarial reward
            attack_reward = self._calculate_attacker_reward(reward, done, new_obs_obj.current_step, new_obs_obj.max_step)
            self.episode_reward += attack_reward
            
            # Log perturbation
            self.all_perturbations += [(self.cur_episode, new_obs_obj.current_step, perturbation, attack_reward)]
            
            # Episode finished
            if done:
                self._handle_episode_end()
                
            return self.current_obs, attack_reward, done, info
        
        except Exception as e:
            print(f"Error in step: {e}")
            raise e

    def _apply_perturbation(self, perturbation):
        """
        Apply perturbation to observation. To be implemented by subclasses.
        
        Args:
            perturbation: Perturbation to apply
            
        Returns:
            Perturbed observation object
        """
        raise NotImplementedError("Subclasses must implement _apply_perturbation method")

    def _handle_episode_end(self):
        """
        Handle episode end logic. Can be overridden by subclasses for custom behavior.
        """
        print(f"Episode {self.cur_episode} finished: {self.current_obs_obj.current_step} steps with reward {self.episode_reward}")
        
        self.results += [(self.cur_episode, self.current_obs_obj.current_step, self.episode_reward)]
        self.cur_episode += 1
        self.episode_reward = 0

    def _calculate_attacker_reward(self, victim_reward, done, step, max_steps):
        """
        Compute adversarial reward based on whether the victim agent failed.

        Args:
            victim_reward: Reward of the victim agent
            done: Boolean indicating whether the episode ended
            step: Current step number
            max_steps: Maximum possible steps in the environment.

        Returns:
            Reward for the adversarial agent.
        """
        is_failure = done and step < max_steps
        if is_failure:
            # Scale reward by how early failure happened
            return 10000 * ((max_steps - step) / max_steps) 
        elif done:
            # Penalty if survived
            return -5000.0
        else: 
            # Ongoing small penalty for each step
            return -10.0
        
    def save_logs(self):
        """
        Save training results to CSV.
        """
        df = pd.DataFrame(self.results[1:], columns=self.results[0])
        df.to_csv(self.log_dir, index=False)
        print(f"Training results saved to {self.log_dir}")
        df = pd.DataFrame(self.all_perturbations[1:], columns=self.all_perturbations[0])
        df.to_csv(self.log_dir.replace(".csv", "_perturbations.csv"), index=False)
        print(f"Perturbation results saved to {self.log_dir.replace('.csv', '_perturbations.csv')}")