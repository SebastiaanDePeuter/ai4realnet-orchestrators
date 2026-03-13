"""
Gradient Estimation Perturbation Agent for Adversarial Robustness Testing

Implements adversarial perturbations using gradient estimation to find minimal
modifications that can change agent behavior. Uses projected gradient descent
to iteratively craft perturbations within specified bounds.
"""

import os
from typing import Optional
import numpy as np
import tensorflow as tf
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from modified_curriculum_classes.my_agent import MyAgent


class GradientEstimationPerturbationAgent(BasePerturbationAgent):
    """
    Adversarial perturbation agent using gradient estimation and projected gradient descent.
    
    Crafts minimal perturbations to observations that can change agent decisions,
    useful for testing robustness against adversarial inputs.
    """

    def __init__(self, 
                 obs_space: grid2op.Observation.ObservationSpace,
                 agent: MyAgent,
                 epsilon: float = 0.1,
                 step_size: float = 0.05,
                 n_iter: int = 5,
                 save_dir: str = ""):
        """
        Initialize gradient estimation perturbation agent.

        Args:
            obs_space: Grid2Op observation space
            agent: Target agent to test
            epsilon: Maximum perturbation magnitude (L∞ bound)
            step_size: Step size for gradient descent
            n_iter: Number of optimization iterations
            save_dir: Directory to save perturbation history
        """
        super().__init__(obs_space, name="GradientEstimationAgent")
        
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_iter = n_iter
        self.save_dir = save_dir
        self.agent = agent
        
        # Pre-compute gradient estimation matrices for efficiency
        self._setup_gradient_helpers(obs_space.n)
        
        # Track perturbation history for analysis
        self.perturb_hist = []

    def _setup_gradient_helpers(self, obs_dim: int) -> None:
        """Setup matrices for finite difference gradient estimation."""
        delta = 0.01
        
        # Positive and negative perturbation matrices
        pos_perturb = np.zeros((obs_dim, obs_dim))
        neg_perturb = np.zeros((obs_dim, obs_dim))
        
        np.fill_diagonal(pos_perturb, delta)
        np.fill_diagonal(neg_perturb, -delta)
        
        # Combine for batch gradient computation
        self.grad_helper = np.concatenate([pos_perturb, neg_perturb])

    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """
        Apply gradient-based adversarial perturbation to observation.

        Args:
            obs: Original observation

        Returns:
            Adversarially perturbed observation
        """
        obs_perturbed = obs.copy()
        obs_vector = obs.to_vect().reshape((1, -1))
        
        # Get agent's preferred action for this observation
        _, target_action_idx = self.agent.act_with_id(obs)
        
    
        if target_action_idx < 0:
            # No specific action targeted, return unperturbed observation
            self.perturb_hist.append(np.zeros_like(obs_perturbed.to_vect()))
            return obs_perturbed
        
        # Compute adversarial perturbation
        adversarial_vector = self._projected_gradient_descent(obs_vector, target_action_idx)
        
        # Apply perturbation to observation
        obs_perturbed._vectorized = adversarial_vector[0]
        
        # Store perturbation for analysis
        perturbation = obs_perturbed.to_vect() - obs.to_vect()
        self.perturb_hist.append(perturbation)
        
        return obs_perturbed

    def _compute_gradient(self, obs_vector: np.ndarray, target_idx: int) -> np.ndarray:
        """
        Estimate gradient using finite differences.

        Args:
            obs_vector: Current observation vector
            target_idx: Target action index to perturb

        Returns:
            Estimated gradient for target action probability
        """
        # Create perturbed versions for gradient estimation
        perturbed_vectors = obs_vector[0] + self.grad_helper
        
        # Get model predictions - handle both Keras and SavedModel formats
        if isinstance(self.agent.model, tf.keras.models.Model):
            # Keras model - use batch prediction
            raw_predictions = self.agent.model.predict(perturbed_vectors, verbose=0)
            
            # Handle different model output formats
            if isinstance(raw_predictions, (list,tuple)):
                # Functional model returns (logits, values)
                raw_predictions = raw_predictions[0]

            raw_predictions = np.asarray(raw_predictions)


            # Convert to probabilities - handle both 2D and 3D outputs
            if raw_predictions.ndim == 2:
                # Shape: (batch, n_actions)
                probabilities = tf.nn.softmax(raw_predictions, axis=1).numpy()
            else:
                # Shape might be (batch, 1, n_actions) or similar
                raw_predictions = raw_predictions.reshape(
                    raw_predictions.shape[0], -1
                )
                probabilities = tf.nn.softmax(raw_predictions, axis=-1).numpy()
            
        else:
            
            signature_fn = self.agent.model.signatures["serving_default"]            
            
            output = signature_fn(
                observations=tf.convert_to_tensor(perturbed_vectors, dtype=tf.float32)
            )
            
            # Extract action logits - prioritize action_out
            if "action_out" in output:
                logits = output["action_out"]
            elif "action_dist_inputs" in output:
                logits = output["action_dist_inputs"]
            elif "output_0" in output:
                logits = output["output_0"]
            else:
                # Use first available output
                logits = output[list(output.keys())[0]]
            
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        
        # Compute finite difference gradient
        n_dims = obs_vector.shape[1]
        pos_probs = probabilities[:n_dims, target_idx]
        neg_probs = probabilities[n_dims:, target_idx]
        
        gradients = (pos_probs - neg_probs) / 0.02
        
        return gradients

    def _projected_gradient_descent(self, obs_vector: np.ndarray, target_idx: int) -> np.ndarray:
        """
        Perform projected gradient descent to find adversarial perturbation.

        Args:
            obs_vector: Original observation vector
            target_idx: Target action to perturb

        Returns:
            Adversarially perturbed observation vector
        """
        perturbed_vector = np.array(obs_vector)
        
        # Define perturbation bounds
        lower_bound = np.minimum(
            obs_vector * (1 - self.epsilon), 
            obs_vector - self.epsilon
        )
        upper_bound = np.maximum(
            obs_vector * (1 + self.epsilon), 
            obs_vector + self.epsilon
        )
        
        # Iterative gradient descent with projection
        for _ in range(self.n_iter):
            gradients = self._compute_gradient(perturbed_vector, target_idx)
            
            # Gradient step (minimize target action probability)
            perturbed_vector -= self.step_size * np.sign(gradients)
            
            # Project back to allowed region
            perturbed_vector = np.clip(perturbed_vector, lower_bound, upper_bound)
        
        return perturbed_vector

    def reset(self) -> None:
        """Reset agent state and save perturbation history."""
        self._save_perturbation_history()
        self.perturb_hist = []
        super().reset()

    def _save_perturbation_history(self) -> None:
        """Save perturbation history to disk for analysis."""
        if not self.save_dir or not self.perturb_hist:
            return
        
        # Find next available filename
        file_idx = 0
        while os.path.exists(os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")):
            file_idx += 1
        
        # Save compressed history
        save_path = os.path.join(self.save_dir, f"perturb_hist_{file_idx}.npz")
        np.savez_compressed(save_path, perturb_hist=self.perturb_hist)