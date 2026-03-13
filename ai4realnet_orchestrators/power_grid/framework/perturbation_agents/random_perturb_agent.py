"""
Random Perturbation Agent for Grid2Op Robustness Testing

Implements probabilistic perturbations to generation, load, and line flow data
with temporal persistence. Supports both Grid2Op observations and numpy arrays
for flexible integration with different testing frameworks.
"""

import os
from typing import Union, Dict, List, Optional
import numpy as np
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent


class RandomPerturbationAgent(BasePerturbationAgent):
    """
    Random perturbation agent with temporal persistence and component-specific noise.
    
    Applies stochastic perturbations to generation, load, and line flow measurements
    with configurable persistence durations. Simulates realistic sensor noise,
    measurement errors, and communication failures in power grid monitoring systems.
    """

    def __init__(self,
                 obs_space: grid2op.Observation.ObservationSpace,
                 prob_perturb: float = 0.4,
                 sigma_gen: float = 0.15,
                 sigma_load: float = 0.15,
                 sigma_rho: float = 0.15,
                 mean_perturb_duration_steps: float = 6.0,
                 save_dir: str = ""):
        """
        Initialize random perturbation agent.

        Args:
            obs_space: Grid2Op observation space
            prob_perturb: Probability of applying perturbation each step
            sigma_gen: Log-normal sigma for generation perturbations
            sigma_load: Log-normal sigma for load perturbations
            sigma_rho: Log-normal sigma for line flow perturbations
            mean_perturb_duration_steps: Mean duration of perturbation persistence
            save_dir: Directory to save perturbation history
        """
        super().__init__(obs_space, name="RandomPerturbationAgent")
        
        self.prob_perturb = prob_perturb
        self.save_dir = save_dir
        self.mean_perturb_duration_steps = mean_perturb_duration_steps
        
        # Component-specific noise parameters
        self.sigma_dict = {
            "gen": sigma_gen,
            "load": sigma_load,
            "rho": sigma_rho
        }
        
        # State tracking
        self.current_perturbations = []  # [component_name, index, factor, remaining_steps]
        self.perturbation_history = {}

    def perturb(self, 
                obs: Union[grid2op.Observation.BaseObservation, np.ndarray]
                ) -> Union[grid2op.Observation.BaseObservation, np.ndarray]:
        """
        Apply random perturbations to observation.

        Args:
            obs: Grid2Op observation or numpy array

        Returns:
            Perturbed observation of the same type as input
        """
        if isinstance(obs, np.ndarray):
            return self._perturb_numpy_array(obs)
        else:
            return self._perturb_grid2op_observation(obs)

    def _perturb_grid2op_observation(self, 
                                   obs: grid2op.Observation.BaseObservation
                                   ) -> grid2op.Observation.BaseObservation:
        """Apply perturbations to Grid2Op observation object."""
        obs_perturbed = obs.copy()
        
        # Apply existing persistent perturbations
        obs_perturbed = self._apply_persistent_perturbations(obs_perturbed)
        
        # Decide whether to add new perturbation
        if self.space_prng.random() <= self.prob_perturb:
            obs_perturbed = self._apply_new_perturbation(obs_perturbed)
        
        # Update internal matrices after perturbation
        obs_perturbed._reset_matrices()
        
        # Store perturbation state for analysis
        self.perturbation_history[f"step_{obs.current_step}"] = np.array(
            self.current_perturbations, dtype=object
        )
        
        return obs_perturbed

    def _perturb_numpy_array(self, obs_vector: np.ndarray) -> np.ndarray:
        """Apply perturbations to numpy observation vector."""
        # Convert to Grid2Op observation for consistent processing
        obs = self.obs_space.from_vect(obs_vector)
        obs_perturbed = self._perturb_grid2op_observation(obs)
        
        # Convert back to numpy array
        return obs_perturbed.to_vect()

    def _apply_persistent_perturbations(self, 
                                      obs: grid2op.Observation.BaseObservation
                                      ) -> grid2op.Observation.BaseObservation:
        """Apply currently active perturbations and update their duration."""
        if not self.current_perturbations:
            return obs
        
        expired_indices = []
        
        for i, (component_name, perturb_idx, perturb_factor, remaining_steps) in enumerate(self.current_perturbations):
            if remaining_steps <= 0:
                expired_indices.insert(0, i)  # Insert at front for reverse iteration
                continue
            
            # Apply perturbation to appropriate component
            self._apply_perturbation_to_component(obs, component_name, perturb_idx, perturb_factor)
            
            # Decrease remaining duration
            self.current_perturbations[i][3] -= 1
        
        # Remove expired perturbations
        for idx in expired_indices:
            self.current_perturbations.pop(idx)
        
        return obs

    def _apply_new_perturbation(self, 
                              obs: grid2op.Observation.BaseObservation
                              ) -> grid2op.Observation.BaseObservation:
        """Generate and apply a new random perturbation."""
        # Randomly select component to perturb
        component_prob = self.space_prng.random()
        if component_prob < 0.25:
            component_name = "gen"
            target_vector = obs.gen_p
        elif component_prob < 0.5:
            component_name = "load"
            target_vector = obs.load_p
        else:
            component_name = "rho"
            target_vector = obs.rho
        
        # Apply perturbation to selected component
        self._perturb_component_vector(target_vector, component_name)
        
        return obs

    def _perturb_component_vector(self, vector: np.ndarray, component_name: str) -> None:
        """
        Apply perturbation to a specific component vector.

        Args:
            vector: Component vector to perturb (modified in place)
            component_name: Name of the component ("gen", "load", or "rho")
        """
        # Select random element within component
        perturb_idx = self.space_prng.randint(0, len(vector))
        
        # Generate perturbation factor
        if self.space_prng.random() < 0.2:
            # 20% chance of complete data loss (sensor failure)
            perturb_factor = 0.0
        else:
            # Log-normal multiplicative noise
            perturb_factor = self.space_prng.lognormal(
                mean=0.0, 
                sigma=self.sigma_dict[component_name]
            )
        
        # Calculate perturbation duration (exponential distribution)
        perturb_duration = np.round(
            self.space_prng.exponential(self.mean_perturb_duration_steps)
        )
        
        # Apply perturbation immediately
        vector[perturb_idx] *= perturb_factor
        
        # Store for future persistence
        self.current_perturbations.append([
            component_name, perturb_idx, perturb_factor, perturb_duration
        ])

    def _apply_perturbation_to_component(self,
                                       obs: grid2op.Observation.BaseObservation,
                                       component_name: str,
                                       perturb_idx: int,
                                       perturb_factor: float) -> None:
        """Apply perturbation to specific observation component."""
        if component_name == "gen":
            obs.gen_p[perturb_idx] *= perturb_factor
        elif component_name == "load":
            obs.load_p[perturb_idx] *= perturb_factor
        elif component_name == "rho":
            obs.rho[perturb_idx] *= perturb_factor

    def reset(self) -> None:
        """Reset agent state and save perturbation history."""
        self._save_perturbation_history()
        self.perturbation_history = {}
        self.current_perturbations = []
        super().reset()

    def _save_perturbation_history(self) -> None:
        """Save perturbation history to compressed file."""
        if not self.perturbation_history or not self.save_dir:
            return
        
        # Create save directory if needed
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find next available filename
        file_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")):
            file_idx += 1
        
        # Save compressed history
        save_path = os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")
        np.savez_compressed(save_path, **self.perturbation_history)

    def get_perturbation_stats(self) -> Dict:
        """Get current perturbation statistics."""
        stats = super().get_perturbation_info()
        stats.update({
            "active_perturbations": len(self.current_perturbations),
            "prob_perturb": self.prob_perturb,
            "mean_duration": self.mean_perturb_duration_steps,
            "component_sigmas": self.sigma_dict.copy()
        })
        return stats

    def set_perturbation_probability(self, prob: float) -> None:
        """Update perturbation probability."""
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {prob}")
        self.prob_perturb = prob

    def clear_persistent_perturbations(self) -> None:
        """Clear all currently active perturbations."""
        self.current_perturbations = []