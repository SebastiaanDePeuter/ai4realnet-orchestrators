"""
Modified Lambda-PIR Perturbation Agent with Missing/Large/Adversarial Actions
Now uses the same action space as RL agent for fair comparison!
"""

import copy
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent

logger = logging.getLogger(__name__)


class LambdaPIRPerturbationAgent(BasePerturbationAgent):
    """
    Lambda-PIR Agent modified to use Missing/Large/Adversarial perturbations.
    
    Now directly comparable to RL agent by using the same action space:
    - Missing values (set to 0)
    - Large values (set to 999999)
    - Adversarial examples
    
    Lambda-PIR decides WHICH type and WHERE to apply based on:
    - Policy iteration: Quick decisions from learned patterns
    - Value iteration: Refined decisions through gradient search
    """
    
    def __init__(self,
             obs_space: grid2op.Observation.ObservationSpace,
             agent,
             policy_model: Optional[Any] = None,
             lambda_param: float = 0.9,
             initial_prob_policy: float = 0.8,
             epsilon: float = 0.1,
             gradient_step_size: float = 0.05,
             refinement_iterations: int = 5,
             decay_schedule: str = "linear",
             name: str = "LambdaPIRPerturbationAgent",
             save_dir: str = "",
             debug: bool = True,
             use_gpu: bool = True):
        """Initialize Lambda-PIR with discrete action space."""
        super().__init__(obs_space, name=name)
        
        self.agent = agent
        self.policy_model = policy_model
        
        # Lambda-PIR parameters
        self.lambda_param = lambda_param
        self.initial_prob_policy = initial_prob_policy
        self.current_prob_policy = initial_prob_policy
        self.epsilon = epsilon
        self.gradient_step_size = gradient_step_size
        self.refinement_iterations = refinement_iterations
        self.decay_schedule = decay_schedule
        self.debug = debug
        
        # Initialize with default action space
        self.possible_actions = [("do_nothing", 0)]  # Always have at least do_nothing
        self.missing_indices = []
        self.large_indices = []
        self.adv_indices = []
        self.attr_start_idx = {}
        
        # Tracking
        self.iteration_count = 0
        self.policy_updates = 0
        self.value_updates = 0
        self.action_history = []
        
        # Value estimates - initialize AFTER possible_actions is set
        self.action_values = np.zeros(len(self.possible_actions))
        
        logger.info(f"Initialized {name} with minimal action space (will build full space on first perturb)")

    def _build_action_space(self, sample_obs):
        """Build the discrete action space matching RL agent."""
        self.possible_actions = []

        # Action 0: Do nothing
        self.possible_actions.append(("do_nothing", 0))

        # Use passed sample_obs instead of calling get_obs
        obs_vector = sample_obs.to_vect()

        # Map attribute start indices
        self.attr_start_idx = {}
        current_idx = 0
        for attr in ["year", "month", "day", "hour_of_day", "minute_of_hour", 
                    "day_of_week", "gen_p", "gen_q", "gen_v", "load_p", 
                    "load_q", "load_v", "p_or", "q_or", "v_or", "a_or", 
                    "p_ex", "q_ex", "v_ex", "a_ex", "rho"]:
            if hasattr(sample_obs, attr):
                attr_array = getattr(sample_obs, attr)
                if isinstance(attr_array, np.ndarray):
                    self.attr_start_idx[attr] = current_idx
                    current_idx += len(attr_array)

        # Select key indices for perturbation (rho and power flows)
        rho_start = self.attr_start_idx.get("rho", 0)
        rho_end = rho_start + len(sample_obs.rho)
        p_or_start = self.attr_start_idx.get("p_or", 0)
        p_or_end = p_or_start + len(sample_obs.p_or)
        critical_indices = list(range(rho_start, rho_end)) + list(range(p_or_start, p_or_end))

        # Missing values
        self.missing_indices = []
        for idx in critical_indices:
            self.possible_actions.append(("missing", idx))
            self.missing_indices.append(len(self.possible_actions) - 1)

        # Large values
        self.large_indices = []
        for idx in critical_indices:
            self.possible_actions.append(("large", idx))
            self.large_indices.append(len(self.possible_actions) - 1)

        # Adversarial examples
        self.adv_indices = []
        if hasattr(self.agent, 'action_space'):
            n_agent_actions = min(20, self.agent.action_space.n)
            for target_action in range(n_agent_actions):
                self.possible_actions.append(("adv_exmpl", target_action))
                self.adv_indices.append(len(self.possible_actions) - 1)

        logger.info(f"Built action space with {len(self.possible_actions)} actions")

    def _ensure_action_space_built(self, obs):
        """Build action space on first actual observation if not already built."""
        if len(self.possible_actions) == 1:  # Only has do_nothing
            try:
                self._build_action_space(obs)
                # Resize action_values to match new action space
                self.action_values = np.zeros(len(self.possible_actions))
                logger.info(f"Built action space with {len(self.possible_actions)} actions")
            except Exception as e:
                logger.error(f"Failed to build action space: {e}")
                # Keep do_nothing as fallback
    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """Apply Lambda-PIR perturbation using missing/large/adversarial framework."""
        try:
            # Build action space on first perturb if needed
            self._ensure_action_space_built(obs)
            
            # Safety check
            if len(self.possible_actions) == 0:
                logger.error("No actions available, returning original observation")
                return obs.copy()
            
            obs_perturbed = copy.deepcopy(obs)
            
            # Get probability schedule
            prob_policy = self._get_probability_schedule(self.iteration_count)
            use_policy = self.space_prng.random() < prob_policy
            
            if use_policy:
                action_idx = self._policy_iteration_step(obs)
                self.policy_updates += 1
            else:
                action_idx = self._value_iteration_step(obs)
                self.value_updates += 1
            
            # Apply the selected action
            action_type, action_param = self.possible_actions[action_idx]
            obs_perturbed = self._apply_action(obs_perturbed, action_type, action_param)
            
            # Update statistics
            self._update_action_value(action_idx, obs, obs_perturbed)
            self.action_history.append(action_idx)
            
            if self.debug and self.iteration_count % 20 == 0:
                logger.debug(f"[ITER {self.iteration_count}] action={action_type}({action_param}) "
                        f"policy={use_policy} p_k={prob_policy:.3f}")
            
            self.iteration_count += 1
            self.perturbation_count += 1
            
            return obs_perturbed
            
        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            return obs.copy()

    def _policy_iteration_step(self, obs: grid2op.Observation.BaseObservation) -> int:
        """Policy iteration with safety checks."""
        # Safety check
        if len(self.possible_actions) == 0:
            logger.error("No actions available in policy iteration")
            return 0
        
        if self.space_prng.random() < 0.1:  # 10% exploration
            return self.space_prng.randint(0, len(self.possible_actions))
        else:
            scores = self.action_values.copy()
            
            # Add heuristic bonuses
            obs_vector = obs.to_vect()
            
            for i, (action_type, idx) in enumerate(self.possible_actions):
                if action_type == "large" and idx < len(obs_vector):
                    if "rho" in self.attr_start_idx:
                        rho_start = self.attr_start_idx["rho"]
                        if rho_start <= idx < rho_start + len(obs.rho):
                            rho_idx = idx - rho_start
                            current_load = obs.rho[rho_idx]
                            scores[i] += current_load * 10
            
            # Safety check before argmax
            if len(scores) == 0:
                logger.error("Empty scores array")
                return 0
            
            return np.argmax(scores)


    def _value_iteration_step(self, obs: grid2op.Observation.BaseObservation) -> int:
        """
        Value iteration: Refine action selection through lookahead.
        """
        # Start with policy selection
        best_action = self._policy_iteration_step(obs)
        best_value = self._evaluate_action(obs, best_action)
        
        # Refine through gradient search in action space
        for _ in range(self.refinement_iterations):
            # Sample nearby actions
            candidates = []
            
            # Try actions of the same type
            action_type, _ = self.possible_actions[best_action]
            if action_type == "missing":
                candidates = self.missing_indices[:5]  # Sample 5
            elif action_type == "large":
                candidates = self.large_indices[:5]
            elif action_type == "adv_exmpl":
                candidates = self.adv_indices[:5]
            
            # Evaluate candidates
            for candidate_idx in candidates:
                value = self._evaluate_action(obs, candidate_idx)
                if value > best_value:
                    best_value = value
                    best_action = candidate_idx
        
        return best_action

    def _evaluate_action(self, obs: grid2op.Observation.BaseObservation, action_idx: int) -> float:
        """
        Evaluate the effectiveness of an action.
        Higher score = more disruptive to defender.
        """
        try:
            # Apply action to get perturbed observation
            obs_test = copy.deepcopy(obs)
            action_type, action_param = self.possible_actions[action_idx]
            obs_test = self._apply_action(obs_test, action_type, action_param)
            
            # Measure disruption potential
            score = 0.0
            
            # Check if it triggers critical loads
            if hasattr(obs_test, 'rho'):
                max_rho = np.max(obs_test.rho)
                score += max_rho * 10  # Reward high line loads
                
                # Extra bonus for pushing lines over threshold
                critical_lines = np.sum(obs_test.rho > 0.95)
                score += critical_lines * 50
            
            # Penalty for do_nothing
            if action_type == "do_nothing":
                score -= 100
            
            return score
            
        except Exception as e:
            logger.debug(f"Action evaluation failed: {e}")
            return 0.0

    def _apply_action(self, obs: grid2op.Observation.BaseObservation, 
                     action_type: str, action_param: int) -> grid2op.Observation.BaseObservation:
        """Apply the selected action to the observation."""
        obs.to_vect()  # Ensure vectorized form is available
        
        if action_type == "do_nothing":
            return obs
            
        elif action_type == "missing":
            obs._vectorized[action_param] = 0
            # Update corresponding attribute
            self._update_obs_attribute(obs, action_param, 0)
            
        elif action_type == "large":
            obs._vectorized[action_param] = 999999
            # Update corresponding attribute
            self._update_obs_attribute(obs, action_param, 999999)
            
        elif action_type == "adv_exmpl":
            # Simple adversarial: Add noise to push towards misclassification
            obs_vector = obs.to_vect()
            noise = np.random.randn(*obs_vector.shape) * self.epsilon
            obs._vectorized = obs_vector + noise
            # Update rho to match perturbed values
            if hasattr(obs, 'rho'):
                rho_start = self.attr_start_idx.get("rho", 0)
                rho_end = rho_start + len(obs.rho)
                obs.rho = obs._vectorized[rho_start:rho_end]
        
        return obs

    def _update_obs_attribute(self, obs: grid2op.Observation.BaseObservation, 
                              idx: int, value: float):
        """Update the corresponding attribute after modifying vectorized form."""
        # Find which attribute this index belongs to
        for attr_name, start_idx in self.attr_start_idx.items():
            if hasattr(obs, attr_name):
                attr_array = getattr(obs, attr_name)
                end_idx = start_idx + len(attr_array)
                if start_idx <= idx < end_idx:
                    attr_idx = idx - start_idx
                    attr_array[attr_idx] = value
                    setattr(obs, attr_name, attr_array)
                    break

    def _update_action_value(self, action_idx: int, 
                            obs_before: grid2op.Observation.BaseObservation,
                            obs_after: grid2op.Observation.BaseObservation):
        """Update value estimates for actions based on outcomes."""
        # Simple learning: Track effectiveness
        reward = 0.0
        
        # Reward if we increased line loads
        if hasattr(obs_after, 'rho'):
            max_rho_after = np.max(obs_after.rho)
            max_rho_before = np.max(obs_before.rho)
            reward = (max_rho_after - max_rho_before) * 10
        
        # Update value with learning rate
        learning_rate = 0.1
        self.action_values[action_idx] = (
            (1 - learning_rate) * self.action_values[action_idx] + 
            learning_rate * reward
        )

    def _get_probability_schedule(self, iteration: int) -> float:
        """Get probability of using policy iteration."""
        if self.decay_schedule == "linear":
            decay_factor = 1.0 / (1.0 + 0.01 * iteration)
        elif self.decay_schedule == "exponential":
            decay_factor = np.exp(-0.01 * iteration)
        else:
            decay_factor = 1.0
        
        self.current_prob_policy = self.initial_prob_policy * decay_factor
        return np.clip(self.current_prob_policy, 0.1, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        if not self.action_history:
            return {}
        
        # Count action types
        action_counts = {"do_nothing": 0, "missing": 0, "large": 0, "adv_exmpl": 0}
        for action_idx in self.action_history[-100:]:
            action_type, _ = self.possible_actions[action_idx]
            action_counts[action_type] += 1
        
        return {
            "total_iterations": self.iteration_count,
            "policy_updates": self.policy_updates,
            "value_updates": self.value_updates,
            "current_prob_policy": self.current_prob_policy,
            "action_counts": action_counts,
            "top_action": max(action_counts, key=action_counts.get)
        }

    def reset(self):
        """Reset between episodes."""
        self.iteration_count = 0
        self.action_history = []
        super().reset()