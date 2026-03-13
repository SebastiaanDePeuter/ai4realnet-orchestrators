"""
Random Perturbation Agent for Grid2Op Robustness Testing

Implements stochastic perturbations to grid observations using configurable
probability distributions and temporal persistence. Simulates realistic
sensor noise, measurement errors, and communication failures.
"""

import os
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent


class RandomPerturbationAgent(BasePerturbationAgent):
    """
    Random perturbation agent with temporal persistence and configurable noise.
    
    Applies stochastic perturbations to different observation components
    (generation, load, line flows) with realistic temporal characteristics.
    Perturbations can persist across multiple time steps to simulate
    realistic measurement errors or communication delays.
    """

    def __init__(self,
                 obs_space: grid2op.Observation.ObservationSpace,
                 gen_idx_range: Tuple[int, int],
                 load_idx_range: Tuple[int, int], 
                 rho_idx_range: Tuple[int, int],
                 prob_perturb: float = 0.4,
                 sigma_gen: float = 0.15,
                 sigma_load: float = 0.15,
                 sigma_rho: float = 0.15,
                 mean_duration_steps: float = 6.0,
                 apply_persistent_perturb: bool = True,
                 seed: int = 0,
                 save_dir: Optional[str] = ""):
        """
        Initialize random perturbation agent.

        Args:
            obs_space: Grid2Op observation space
            gen_idx_range: (min, max) indices for generation data
            load_idx_range: (min, max) indices for load data
            rho_idx_range: (min, max) indices for line flow data
            prob_perturb: Probability of applying new perturbation each step
            sigma_gen: Log-normal sigma for generation perturbations
            sigma_load: Log-normal sigma for load perturbations  
            sigma_rho: Log-normal sigma for line flow perturbations
            mean_duration_steps: Mean duration of perturbation persistence
            apply_persistent_perturb: Whether perturbations persist across steps
            seed: Random seed for reproducibility
            save_dir: Directory to save perturbation history
        """
        super().__init__(obs_space, name="RandomPerturbationAgentNp")
        
        self.prob_perturb = prob_perturb
        self.save_dir = save_dir
        self.apply_persistent_perturb = apply_persistent_perturb
        self.mean_duration_steps = mean_duration_steps
        
        # Index ranges for different observation components
        self.idx_ranges = {
            "gen": gen_idx_range,
            "load": load_idx_range, 
            "rho": rho_idx_range
        }
        
        # Noise parameters for different components
        self.sigmas = {
            "gen": sigma_gen,
            "load": sigma_load,
            "rho": sigma_rho
        }
        
        # State tracking
        self.step_count = 0
        self.current_perturbations = []  # List of [component, idx, factor, remaining_steps]
        self.perturbation_history = {}
        
        # Random state for reproducibility
        self.random_state = np.random.RandomState(seed=seed)

    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """
        Apply random perturbations to observation.

        Args:
            obs: Original observation

        Returns:
            Perturbed observation with applied noise
        """
        obs_perturbed = obs.copy()
        obs_vector = obs_perturbed.to_vect()
        
        # Apply existing persistent perturbations
        obs_vector = self._apply_persistent_perturbations(obs_vector)
        
        # Decide whether to add new perturbations
        if self.random_state.random() <= self.prob_perturb:
            obs_vector = self._apply_new_perturbations(obs_vector)
        
        # Update observation vector
        obs_perturbed._vectorized = obs_vector
        
        # Store perturbation state for analysis
        self.perturbation_history[f"step_{self.step_count}"] = np.array(self.current_perturbations, dtype=object)
        self.step_count += 1
        
        return obs_perturbed

    def _apply_persistent_perturbations(self, obs_vector: np.ndarray) -> np.ndarray:
        """Apply currently active perturbations and update their duration."""
        if not self.current_perturbations:
            return obs_vector
        
        # Apply perturbations and update durations
        expired_indices = []
        for i, (component, idx, factor, remaining_steps) in enumerate(self.current_perturbations):
            if remaining_steps <= 0:
                expired_indices.insert(0, i)  # Insert at front for reverse iteration
                continue
                
            # Apply perturbation
            obs_vector[idx] *= factor
            
            # Decrease remaining duration
            self.current_perturbations[i][3] -= 1
        
        # Remove expired perturbations
        for idx in expired_indices:
            self.current_perturbations.pop(idx)
        
        return obs_vector

    def _apply_new_perturbations(self, obs_vector: np.ndarray) -> np.ndarray:
        """Generate and apply new random perturbations."""
        # Determine number of new perturbations
        if self.apply_persistent_perturb:
            num_perturbations = 1
        else:
            num_perturbations = self.random_state.randint(1, 11)  # 1-10 perturbations
        
        for _ in range(num_perturbations):
            # Randomly select component type
            component_prob = self.random_state.random()
            if component_prob < 0.25:
                component = "gen"
            elif component_prob < 0.5:
                component = "load"
            else:
                component = "rho"
            
            # Apply perturbation to selected component
            obs_vector = self._perturb_component(obs_vector, component)
        
        return obs_vector

    def _perturb_component(self, obs_vector: np.ndarray, component: str) -> np.ndarray:
        """
        Apply perturbation to specific observation component.

        Args:
            obs_vector: Observation vector to perturb
            component: Component type ("gen", "load", or "rho")

        Returns:
            Perturbed observation vector
        """
        # Select random index within component range
        min_idx, max_idx = self.idx_ranges[component]
        perturb_idx = self.random_state.randint(min_idx, max_idx + 1)
        
        # Generate perturbation factor
        if self.random_state.random() < 0.2:
            # 20% chance of complete data loss
            perturb_factor = 0.0
        else:
            # Log-normal multiplicative noise
            perturb_factor = self.random_state.lognormal(
                mean=0.0, 
                sigma=self.sigmas[component]
            )
        
        # Determine perturbation duration
        if self.apply_persistent_perturb:
            duration = self.random_state.geometric(1.0 / self.mean_duration_steps) - 1
        else:
            duration = 0
        
        # Apply perturbation
        obs_vector[perturb_idx] *= perturb_factor
        
        # Store perturbation for persistence tracking
        self.current_perturbations.append([component, perturb_idx, perturb_factor, duration])
        
        return obs_vector

    def reset(self) -> None:
        """Reset agent state and save perturbation history."""
        self._save_perturbation_history()
        self.step_count = 0
        self.perturbation_history = {}
        self.current_perturbations = []
        super().reset()

    def _save_perturbation_history(self) -> None:
        """Save perturbation history to compressed file."""
        if not self.perturbation_history or not self.save_dir:
            return
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find next available filename
        file_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")):
            file_idx += 1
        
        # Save history
        save_path = os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")
        np.savez_compressed(save_path, **self.perturbation_history)

    def get_perturbation_stats(self) -> Dict:
        """Get statistics about current perturbation state."""
        stats = super().get_perturbation_info()
        stats.update({
            "active_perturbations": len(self.current_perturbations),
            "step_count": self.step_count,
            "prob_perturb": self.prob_perturb,
            "mean_duration": self.mean_duration_steps,
            "persistent_mode": self.apply_persistent_perturb
        })
        return stats


class SimpleRandomPerturbationAgent(BasePerturbationAgent):
    """
    Simplified random perturbation agent for basic robustness testing.
    
    Applies simple multiplicative noise without temporal persistence.
    Useful for quick robustness assessments.
    """

    def __init__(self,
                 obs_space: grid2op.Observation.ObservationSpace,
                 prob_perturb: float = 0.3,
                 noise_std: float = 0.1,
                 seed: int = 0):
        """
        Initialize simple random perturbation agent.

        Args:
            obs_space: Grid2Op observation space
            prob_perturb: Probability of perturbing each observation element
            noise_std: Standard deviation of Gaussian noise
            seed: Random seed
        """
        super().__init__(obs_space, name="SimpleRandomAgent")
        
        self.prob_perturb = prob_perturb
        self.noise_std = noise_std
        self.random_state = np.random.RandomState(seed=seed)

    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """Apply simple random noise to observation."""
        obs_perturbed = obs.copy()
        obs_vector = obs_perturbed.to_vect()
        
        # Generate random mask for perturbation
        perturb_mask = self.random_state.random(len(obs_vector)) < self.prob_perturb
        
        # Apply Gaussian noise to selected elements
        noise = self.random_state.normal(1.0, self.noise_std, len(obs_vector))
        obs_vector[perturb_mask] *= noise[perturb_mask]
        
        # Update observation
        obs_perturbed._vectorized = obs_vector
        
        return obs_perturbed