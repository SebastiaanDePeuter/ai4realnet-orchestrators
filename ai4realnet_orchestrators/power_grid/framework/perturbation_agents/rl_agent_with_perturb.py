"""
RL Agent with Perturbation Wrapper for Robustness Testing

Wraps existing RL agents with perturbation capabilities to test robustness
by comparing actions taken on original vs perturbed observations. Tracks
when perturbations cause action changes for analysis.
"""

import os
from typing import List, Optional
import numpy as np
import grid2op
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from perturbation_agents.base_perturb_agent import BasePerturbationAgent


class RLAgentWithPerturbation(BaseAgent):
    """
    Wrapper that adds perturbation testing to existing RL agents.
    
    Tests agent robustness by applying perturbations to observations
    and tracking when perturbations cause different action selections.
    Useful for evaluating agent sensitivity to observation noise.
    """

    def __init__(self,
                 action_space: grid2op.Action.ActionSpace,
                 rl_agent: BaseAgent,
                 perturb_agent: BasePerturbationAgent,
                 seed_offset: int = 10,
                 save_dir: str = ""):
        """
        Initialize RL agent with perturbation wrapper.

        Args:
            action_space: Grid2Op action space
            rl_agent: Base RL agent to wrap
            perturb_agent: Perturbation agent for observation modification
            seed_offset: Offset for perturbation agent seed (to decorrelate randomness)
            save_dir: Directory to save action change history
        """
        super().__init__(action_space)
        
        self.rl_agent = rl_agent
        self.perturb_agent = perturb_agent
        self.seed_offset = seed_offset
        self.save_dir = save_dir
        
        # Track action changes for analysis
        self.action_history_unperturbed = []
        self.action_changes = []  # Boolean array indicating when perturbation changed action

    def act(self, 
            observation: BaseObservation, 
            reward: float, 
            done: bool = False) -> BaseAction:
        """
        Select action using perturbed observation and track changes.

        Args:
            observation: Current grid observation
            reward: Previous step reward
            done: Whether episode is complete

        Returns:
            Action selected on perturbed observation
        """
        # Get action on original observation for comparison
        action_unperturbed = self.rl_agent.act(observation, reward, done)
        
        # Get action on perturbed observation (actual return value)
        perturbed_observation = self.perturb_agent.perturb(observation)
        action_perturbed = self.rl_agent.act(perturbed_observation, reward, done)
        
        # Track action history and changes
        self.action_history_unperturbed.append(action_unperturbed.to_vect())
        self.action_changes.append(not self._actions_equal(action_unperturbed, action_perturbed))
        
        return action_perturbed

    def _actions_equal(self, action1: BaseAction, action2: BaseAction) -> bool:
        """
        Check if two actions are equivalent.

        Args:
            action1: First action to compare
            action2: Second action to compare

        Returns:
            True if actions are equivalent, False otherwise
        """
        try:
            # Compare action vectors
            return np.array_equal(action1.to_vect(), action2.to_vect())
        except Exception:
            # Fallback to string comparison if vectorization fails
            return str(action1) == str(action2)

    def reset(self, observation: BaseObservation) -> None:
        """
        Reset agent state and save action history.

        Args:
            observation: Initial observation of new episode
        """
        # Save history from previous episode
        self._save_action_history()
        
        # Reset state for new episode
        self.action_history_unperturbed = []
        self.action_changes = []
        
        # Reset wrapped components
        super().reset(observation)
        
        if hasattr(self.rl_agent, 'reset'):
            self.rl_agent.reset(observation)
        
        self.perturb_agent.reset()

    def seed(self, seed: int) -> None:
        """
        Set random seeds for both agents with offset.

        Args:
            seed: Base random seed
        """
        # Seed RL agent with base seed
        if hasattr(self.rl_agent, 'seed'):
            self.rl_agent.seed(seed)
        
        # Seed perturbation agent with offset to decorrelate randomness
        self.perturb_agent.set_seed(seed + self.seed_offset)

    def _save_action_history(self) -> None:
        """Save action history and change tracking to disk."""
        if not self.action_changes or not self.save_dir:
            return
        
        # Create save directory if needed
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Find next available filename
        file_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f"action_hist_{file_idx}.npz")):
            file_idx += 1
        
        # Save action history and change tracking
        save_path = os.path.join(self.save_dir, f"action_hist_{file_idx}.npz")
        np.savez_compressed(
            save_path,
            action_unperturbed=self.action_history_unperturbed,
            action_changed=self.action_changes
        )

    def get_robustness_stats(self) -> dict:
        """
        Get statistics about agent robustness to perturbations.

        Returns:
            Dictionary with robustness statistics
        """
        if not self.action_changes:
            return {
                "total_steps": 0,
                "actions_changed": 0,
                "change_rate": 0.0,
                "robustness_score": 1.0
            }
        
        total_steps = len(self.action_changes)
        actions_changed = sum(self.action_changes)
        change_rate = actions_changed / total_steps if total_steps > 0 else 0.0
        robustness_score = 1.0 - change_rate  # Higher score = more robust
        
        return {
            "total_steps": total_steps,
            "actions_changed": actions_changed,
            "change_rate": change_rate,
            "robustness_score": robustness_score,
            "perturbation_agent": self.perturb_agent.name,
            "rl_agent": type(self.rl_agent).__name__
        }

    def set_perturbation_agent(self, new_perturb_agent: BasePerturbationAgent) -> None:
        """
        Replace the current perturbation agent.

        Args:
            new_perturb_agent: New perturbation agent to use
        """
        self.perturb_agent = new_perturb_agent

    def disable_perturbations(self) -> None:
        """Temporarily disable perturbations (use original observations)."""
        from perturbation_agents.base_perturb_agent import NullPerturbationAgent
        self.perturb_agent = NullPerturbationAgent(self.perturb_agent.obs_space)

    def __str__(self) -> str:
        """String representation of the wrapped agent."""
        return (f"RLAgentWithPerturbation("
                f"rl_agent={type(self.rl_agent).__name__}, "
                f"perturb_agent={self.perturb_agent.name})")

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        stats = self.get_robustness_stats()
        return (f"RLAgentWithPerturbation("
                f"rl_agent={type(self.rl_agent).__name__}, "
                f"perturb_agent={self.perturb_agent.name}, "
                f"robustness_score={stats['robustness_score']:.3f})")