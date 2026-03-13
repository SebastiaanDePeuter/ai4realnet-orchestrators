"""
Lambda-PIR Attacker Implementation with GPU Support

Shows proper integration with Grid2Op evaluation pipeline.
Follows the same interface pattern as other attack models.
Now includes GPU acceleration support!
"""

import copy
import traceback
import logging
from typing import Optional, Dict, Any

import numpy as np

# Conditional imports
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    PPO = None
    SAC = None

from perturbation_agents.lambda_pir_perturbation_agent import LambdaPIRPerturbationAgent
from attack_models.BaseAttackerClass import BaseAttackerClass

logger = logging.getLogger(__name__)


class LambdaPIRAttacker(BaseAttackerClass):
    """
    Lambda-PIR (Lambda-Policy Iteration with Randomization) Attacker.
    
    Hybrid adversarial perturbation agent combining:
    - Policy Iteration: Fast PPO/SAC predictions for quick perturbations
    - Value Iteration: Iterative gradient refinement for optimal perturbations
    - Adaptive Switching: Randomized schedule transitioning from policy to value
    - GPU Acceleration: 6-10x speedup with CUDA support (optional)
    
    This wrapper provides integration with Grid2Op evaluation pipelines.
    
    Args:
        model_path: Path to pre-trained PPO or SAC model
        env: Grid2Op environment instance (REQUIRED)
        agent: Target agent being evaluated (REQUIRED)
        lambda_param: λ ∈ [0,1) - Lookahead depth for refinement (default: 0.9)
        initial_prob_policy: Starting probability of policy iteration (default: 0.8)
        epsilon: Maximum perturbation magnitude (default: 0.1)
        gradient_step_size: Learning rate for value iteration (default: 0.05)
        refinement_iterations: Number of refinement steps (default: 5)
        decay_schedule: "linear", "exponential", or "constant" (default: "linear")
        name: Identifier for this attacker (default: "LambdaPIRAttacker")
        save_dir: Directory for saving perturbation history (default: "")
        use_gpu: Enable GPU acceleration if available (default: True)
    
    Example Usage:
        # Basic usage with default parameters
        attacker = LambdaPIRAttacker(
            model_path="trained_models/PPO",
            env=env,
            agent=defender_agent
        )
        
        # With custom λ-PIR parameters and GPU acceleration
        attacker = LambdaPIRAttacker(
            model_path="trained_models/SAC",
            env=env,
            agent=defender_agent,
            lambda_param=0.95,              # Very deep lookahead
            initial_prob_policy=0.7,        # 70% policy, 30% refinement from start
            epsilon=0.15,                   # Larger perturbations allowed
            refinement_iterations=10,       # More refinement steps
            decay_schedule="exponential",   # Smooth decay schedule
            use_gpu=True                    # GPU acceleration enabled
        )
    """
    
    def __init__(self,
                 model_path: str,
                 env,
                 agent,
                 lambda_param: float = 0.9,
                 initial_prob_policy: float = 0.8,
                 epsilon: float = 0.1,
                 gradient_step_size: float = 0.05,
                 refinement_iterations: int = 5,
                 decay_schedule: str = "linear",
                 name: str = "LambdaPIRAttacker",
                 save_dir: str = "",
                 use_gpu: bool = True):
        """
        Initialize Lambda-PIR Attacker with GPU support.
        
        Parameters MUST include:
        - model_path: Path to trained model
        - env: Grid2Op environment
        - agent: Defender agent to attack
        
        Optional parameters:
        - use_gpu: Enable GPU acceleration (default: True)
        """
        try:
            logger.info(f"[DEBUG] Initializing {name}")
            
            # REQUIRED parameters
            self.model_path = model_path
            self.env = env
            self.agent = agent
            
            # Attacker identifiers
            self.model_name = name
            self.pickle_file = "lambda_pir_attacker.pkl"
            
            # λ-PIR configuration
            self.lambda_param = lambda_param
            self.initial_prob_policy = initial_prob_policy
            self.epsilon = epsilon
            self.gradient_step_size = gradient_step_size
            self.refinement_iterations = refinement_iterations
            self.decay_schedule = decay_schedule
            self.save_dir = save_dir
            self.use_gpu = use_gpu
            
            # Load and initialize perturbation agent
            self.perturbation_agent = self._load_lambda_pir_model()
            
            # Statistics tracking
            self.stats = {
                "total_perturbations": 0,
                "policy_iterations": 0,
                "value_iterations": 0,
                "avg_perturbation_magnitude": 0.0,
            }
            
            logger.info(f"[DEBUG] {name} initialized successfully")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Configuration:")
            logger.info(f"    λ = {lambda_param}")
            logger.info(f"    p_k = {initial_prob_policy}")
            logger.info(f"    ε = {epsilon}")
            logger.info(f"    decay_schedule = {decay_schedule}")
            logger.info(f"    GPU = {use_gpu}")
            
        except Exception as e:
            logger.error(f"[ERROR] {name} init failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def perturb(self, obs):
        """
        Apply Lambda-PIR hybrid perturbation to observation.
        
        Args:
            obs: Original Grid2Op observation
            
        Returns:
            Perturbed observation
        """
        try:
            obs_perturbed = copy.deepcopy(obs)
            obs_perturbed = self.perturbation_agent.perturb(obs_perturbed)
            
            # Update statistics
            self.stats["total_perturbations"] += 1
            agent_stats = self.perturbation_agent.get_stats()
            if agent_stats:
                self.stats["policy_iterations"] = agent_stats.get("policy_updates", 0)
                self.stats["value_iterations"] = agent_stats.get("value_updates", 0)
                self.stats["avg_perturbation_magnitude"] = agent_stats.get(
                    "avg_perturbation_magnitude", 0.0
                )
            
            return obs_perturbed
            
        except Exception as e:
            logger.error(f"[ERROR] Perturbation failed: {e}")
            logger.error(traceback.format_exc())
            return obs.copy()

    def _load_lambda_pir_model(self) -> LambdaPIRPerturbationAgent:
        """
        Load and configure Lambda-PIR perturbation agent with GPU support.
        
        Returns:
            Configured LambdaPIRPerturbationAgent instance
        """
        try:
            # Load policy model (PPO or SAC)
            policy_model = self._load_policy_model(self.model_path)
            
            if policy_model is None:
                logger.warning("No policy model loaded. Agent will use zero perturbations.")
            else:
                policy_type = self._identify_policy_type(policy_model)
                logger.info(f"Loaded {policy_type.upper()} policy model from {self.model_path}")
            
            # Create Lambda-PIR perturbation agent with GPU support
            perturbation_agent = LambdaPIRPerturbationAgent(
                obs_space=self.env.observation_space,
                agent=self.agent,  
                policy_model=policy_model,
                lambda_param=self.lambda_param,
                initial_prob_policy=self.initial_prob_policy,
                epsilon=self.epsilon,
                gradient_step_size=self.gradient_step_size,
                refinement_iterations=self.refinement_iterations,
                decay_schedule=self.decay_schedule,
                name=self.model_name,
                save_dir=self.save_dir,
                use_gpu=self.use_gpu  # ← Pass GPU flag to perturbation agent
            )
            
            logger.info(f"LambdaPIRPerturbationAgent created successfully")
            return perturbation_agent
            
        except Exception as e:
            logger.error(f"Failed to load Lambda-PIR model: {e}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def _load_policy_model(model_path: str) -> Optional[Any]:
        """
        Load PPO or SAC model from file.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model or None if not found/failed
        """
        if not model_path:
            logger.warning("No model path provided")
            return None
        
        try:
            # Try PPO first
            try:
                model = PPO.load(model_path, device="cpu")
                logger.info(f"Loaded PPO model from {model_path}")
                return model
            except Exception as ppo_error:
                logger.debug(f"PPO load failed: {ppo_error}")
            
            # Try SAC
            try:
                model = SAC.load(model_path, device="cpu")
                logger.info(f"Loaded SAC model from {model_path}")
                return model
            except Exception as sac_error:
                logger.debug(f"SAC load failed: {sac_error}")
            
            logger.error(f"Could not load PPO or SAC model from {model_path}")
            return None
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None

    @staticmethod
    def _identify_policy_type(model: Any) -> str:
        """
        Identify policy model type.
        
        Args:
            model: Loaded policy model
            
        Returns:
            "ppo", "sac", or "unknown"
        """
        model_class = model.__class__.__name__
        
        if "PPO" in model_class:
            return "ppo"
        elif "SAC" in model_class:
            return "sac"
        else:
            return "unknown"

    def reset(self) -> None:
        """Reset attacker state between episodes."""
        if self.perturbation_agent:
            self.perturbation_agent.reset()
        logger.debug(f"{self.model_name}: Reset complete")

    def cleanup_gpu(self) -> None:
        """
        Cleanup GPU resources (call at END of all evaluations).
        
        This should be called ONCE after all episodes complete.
        Do NOT call between episodes!
        """
        if self.perturbation_agent and hasattr(self.perturbation_agent, 'cleanup_gpu_final'):
            self.perturbation_agent.cleanup_gpu_final()
            logger.info(f"{self.model_name}: GPU cleanup complete")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get attacker statistics.
        
        Returns:
            Dictionary of statistics including GPU info
        """
        agent_stats = {}
        if self.perturbation_agent:
            agent_stats = self.perturbation_agent.get_stats()
        
        return {
            "model_name": self.model_name,
            "total_perturbations": self.stats["total_perturbations"],
            "policy_iterations": self.stats["policy_iterations"],
            "value_iterations": self.stats["value_iterations"],
            "avg_perturbation_magnitude": self.stats["avg_perturbation_magnitude"],
            **agent_stats,  # Include agent-level stats including GPU info
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        gpu_info = " (GPU)" if stats.get('gpu_enabled', False) else " (CPU)"
        return (f"{self.model_name}{gpu_info}("
               f"λ={self.lambda_param}, "
               f"p_k={stats.get('current_probability_policy', 0):.3f}, "
               f"perturbs={self.stats['total_perturbations']}, "
               f"policy={self.stats['policy_iterations']}, "
               f"value={self.stats['value_iterations']})")

    def __repr__(self) -> str:
        """Detailed representation."""
        gpu_info = "GPU" if self.use_gpu else "CPU"
        return (f"{self.__class__.__name__}("
               f"model_path='{self.model_path}', "
               f"lambda={self.lambda_param}, "
               f"initial_prob_policy={self.initial_prob_policy}, "
               f"epsilon={self.epsilon}, "
               f"use_gpu={self.use_gpu})")