"""
Production-Ready Senior Student Agent for Grid2Op Reinforcement Learning

This module implements the Senior component of the curriculum learning framework,
which uses Proximal Policy Optimization (PPO) to refine agent policies through
reinforcement learning. The Senior builds upon Junior models trained through
supervised learning and optimizes them for dynamic grid management scenarios.

Key Features:
- PPO-based reinforcement learning with Ray RLLib
- Support for custom model architectures and configurations
- Configurable perturbation agents for robustness training
- Distributed training with multiple workers
- Comprehensive model validation and testing
- Checkpoint management and model export capabilities

Original work based on: https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
Licensed under Mozilla Public License 2.0
"""

import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

import ray
import tensorflow as tf
from ray._raylet import ObjectRef
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env
from sklearn.base import BaseEstimator

# Updated imports for production folder structure
from modified_curriculum_classes.ppo import PPOConfig
from modified_curriculum_classes.senior_env_rllib import SeniorEnvRllib
from curriculumagent.senior.rllib_execution.senior_model_rllib import Grid2OpCustomModel
from modified_curriculum_classes.my_agent import MyAgent
from perturbation_agents.base_perturb_agent import BasePerturbationAgent

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Senior:
    """
    Senior Agent: Advanced Reinforcement Learning Component for Grid2Op Curriculum Learning.
    
    The Senior agent represents the final stage of the curriculum learning pipeline,
    using PPO (Proximal Policy Optimization) to refine policies learned by Junior agents
    through direct interaction with the Grid2Op environment.
    
    Architecture:
    - Built on Ray RLLib framework for distributed training
    - Supports custom model architectures and configurations
    - Integrates with perturbation agents for robustness training
    - Provides comprehensive checkpointing and model export
    
    Training Process:
    1. Initializes with Junior model weights as starting point
    2. Uses PPO to optimize policy through environment interaction
    3. Supports adversarial training with perturbation agents
    4. Exports final model compatible with MyAgent for deployment
    
    Example:
        >>> senior = Senior(
        ...     env_path="l2rpn_case14_sandbox",
        ...     action_space_path="actions/actions.npy",
        ...     model_path="junior_model/",
        ...     ckpt_save_path="checkpoints/"
        ... )
        >>> senior.train(iterations=100)
        >>> agent = senior.get_my_agent("final_model/")
    """

    def __init__(self,
                 env_path: Union[str, Path],
                 action_space_path: Union[Path, List[Path]],
                 model_path: Union[Path, str],
                 ckpt_save_path: Optional[Union[Path, str]] = None,
                 scaler: Optional[Union[ObjectRef, BaseEstimator, str]] = None,
                 custom_junior_config: Optional[Union[dict, str]] = None,
                 num_workers: Optional[int] = None,
                 subset: Optional[bool] = False,
                 train_batch_size: int = 4000,
                 perturb_agent: Optional[BasePerturbationAgent] = None,
                 env_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the Senior reinforcement learning agent.

        Args:
            env_path: Path to Grid2Op environment (must be available to all workers)
            action_space_path: Path(s) to action space files (.npy format)
                              Can be single file or list of files to concatenate
            model_path: Path to Junior model directory (TensorFlow SavedModel format)
            ckpt_save_path: Directory for saving PPO training checkpoints
                           If None, uses Ray's default checkpoint directory
            scaler: Optional observation scaler for preprocessing:
                   - sklearn BaseEstimator: Direct scaler object
                   - ObjectRef: Ray object reference to shared scaler
                   - str/Path: Path to pickled scaler file
            custom_junior_config: Advanced Junior model configuration:
                                 - dict: Configuration parameters
                                 - str/Path: Path to JSON configuration file
            num_workers: Number of parallel rollout workers for training
                        If None, uses half of available CPU cores
            subset: Whether to use observation subset for efficiency
                   If True, filters observations to essential grid state information
            train_batch_size: PPO training batch size (number of environment steps)
            perturb_agent: Optional perturbation agent for adversarial training
            env_kwargs: Additional Grid2Op environment initialization parameters

        Raises:
            AssertionError: If Ray is not initialized
            ValueError: If model or environment configuration is invalid
            FileNotFoundError: If required files are missing
        """
        # Validate Ray initialization
        if not ray.is_initialized():
            raise AssertionError(
                "Ray must be initialized before creating Senior agent. "
                "Please call ray.init() first."
            )

        # Initialize random state for reproducibility
        self.random = random.Random()
        self.random.seed(0)

        # Store configuration
        self.ckpt_save_path = Path(ckpt_save_path) if ckpt_save_path else None
        self.env_kwargs = env_kwargs or {}

        # Process scaler configuration
        self.scaler = self._load_scaler(scaler)

        # Build environment configuration
        self.env_config = {
            "action_space_path": action_space_path,
            "env_path": str(env_path),
            "action_threshold": 0.95,
            "subset": subset,
            "scaler": self.scaler,
            "topo": True,
            "env_kwargs": self.env_kwargs
        }

        # Process model configuration
        self.model_config, self._is_advanced_model = self._build_model_config(
            model_path, custom_junior_config
        )

        # Register custom model with RLLib
        ModelCatalog.register_custom_model('Senior', Grid2OpCustomModel)

        # Initialize and validate environment
        self.rllib_env: Optional[SeniorEnvRllib] = None
        self._validate_environment_and_model()

        # Configure PPO trainer
        self.ppo_config, self.ppo = self._initialize_ppo_trainer(
            num_workers, train_batch_size, perturb_agent
        )

        logger.info("Senior agent initialized successfully")

    def train(self, 
              iterations: int = 100,
              perturb_agent: Optional[BasePerturbationAgent] = None,
              perturb_start: Optional[Union[int, float]] = None) -> List[Dict[str, Any]]:
        """
        Train the Senior agent using PPO reinforcement learning.

        This method executes the main training loop, optionally incorporating
        adversarial training with perturbation agents for robustness.

        Args:
            iterations: Number of PPO training iterations to execute
            perturb_agent: Optional perturbation agent for adversarial training
                          Overrides agent specified in initialization
            perturb_start: When to start adversarial training:
                          - int: Iteration number to start perturbations
                          - float: Fraction of total iterations (0.0-1.0)
                          - None: Use perturbations from start if agent provided

        Returns:
            List of training statistics dictionaries from each iteration
            Each dict contains metrics like episode_reward_mean, policy_loss, etc.

        Raises:
            RuntimeError: If training encounters unrecoverable errors
        """
        logger.info(f"Starting Senior training for {iterations} iterations")
        
        training_results = []
        perturb_threshold = self._calculate_perturbation_threshold(perturb_start, iterations)
        
        try:
            for iteration in range(iterations):
                # Handle perturbation agent activation
                if perturb_agent and iteration >= perturb_threshold:
                    if iteration == perturb_threshold:
                        logger.info(f"Activating perturbation agent at iteration {iteration}")
                    perturb_agent.reset()

                # Execute training iteration
                iteration_result = self.ppo.train()
                training_results.append(iteration_result)
                
                # Log progress
                episode_reward = iteration_result['sampler_results']['episode_reward_mean']
                logger.info(f"Iteration {iteration}: Episode reward mean = {episode_reward:.3f}")

                # Save checkpoint periodically
                if iteration % 5 == 0 and self.ckpt_save_path:
                    checkpoint_path = self.ppo.save(checkpoint_dir=str(self.ckpt_save_path))
                    logger.debug(f"Checkpoint saved: {checkpoint_path}")

            # Save final checkpoint
            if self.ckpt_save_path:
                final_checkpoint = self.ppo.save(checkpoint_dir=str(self.ckpt_save_path))
                logger.info(f"Final checkpoint saved: {final_checkpoint}")

        except Exception as e:
            logger.error(f"Training failed at iteration {iteration}: {e}")
            raise RuntimeError(f"Senior training failed: {e}") from e

        logger.info("Senior training completed successfully")
        return training_results

    def restore(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Restore Senior agent from a saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or file

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist
            ValueError: If checkpoint is incompatible with current configuration
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            self.ppo.restore(str(checkpoint_path))
            logger.info(f"Successfully restored checkpoint: {checkpoint_path}")
        except Exception as e:
            raise ValueError(f"Failed to restore checkpoint {checkpoint_path}: {e}") from e

    def export_model(self, export_path: Union[str, Path] = "./exported_model") -> Path:
        """
        Export the trained policy model for deployment.

        This method exports the current policy as a TensorFlow SavedModel
        that can be loaded by MyAgent for production use.

        Args:
            export_path: Directory where to save the exported model

        Returns:
            Path to the exported model directory

        Raises:
            RuntimeError: If model export fails
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        try:
            self.ppo.export_policy_model(str(export_path))
            logger.info(f"Model exported to: {export_path}")
            return export_path
        except Exception as e:
            raise RuntimeError(f"Failed to export model to {export_path}: {e}") from e

    def get_my_agent(self, model_save_path: Union[str, Path] = "./final_model") -> MyAgent:
        """
        Create a deployment-ready MyAgent from the trained Senior model.

        This method exports the current policy and wraps it in a MyAgent
        instance configured with the same parameters used during training.

        Args:
            model_save_path: Directory where to save the final model

        Returns:
            Configured MyAgent ready for deployment

        Raises:
            RuntimeError: If agent creation fails
        """
        model_save_path = Path(model_save_path)
        
        try:
            # Export the trained model
            self.export_model(model_save_path)

            # Create MyAgent with matching configuration
            agent = MyAgent(
                action_space=self.rllib_env.single_env.action_space,
                model_path=model_save_path,
                action_space_path=self.env_config["action_space_path"],
                scaler=self.env_config["scaler"],
                best_action_threshold=self.env_config["action_threshold"],
                topo=self.env_config["topo"],
                subset=self.env_config["subset"]
            )

            logger.info(f"Created MyAgent from Senior model at {model_save_path}")
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to create MyAgent: {e}") from e

    # Private helper methods

    def _load_scaler(self, scaler: Optional[Union[ObjectRef, BaseEstimator, str]]) -> Optional[BaseEstimator]:
        """Load and validate observation scaler."""
        if scaler is None:
            return None

        if isinstance(scaler, BaseEstimator):
            return scaler

        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:
                    loaded_scaler = pickle.load(fp)
                logger.info(f"Loaded scaler from {scaler}")
                return loaded_scaler
            except Exception as e:
                logger.warning(f"Failed to load scaler from {scaler}: {e}. Using None.")
                return None

        # Handle Ray ObjectRef (advanced use case)
        try:
            return ray.get(scaler)
        except Exception as e:
            logger.warning(f"Failed to get scaler from Ray ObjectRef: {e}. Using None.")
            return None

    def _build_model_config(self, 
                           model_path: Union[Path, str], 
                           custom_config: Optional[Union[dict, str]]) -> tuple:
        """Build model configuration for RLLib."""
        model_path = Path(model_path)
        
        if custom_config is None:
            return {"model_path": str(model_path)}, False

        # Handle custom configuration
        if isinstance(custom_config, (str, Path)):
            try:
                with open(custom_config) as json_file:
                    custom_config = json.load(json_file)
                logger.info(f"Loaded custom config from {custom_config}")
            except Exception as e:
                logger.error(f"Failed to load custom config from {custom_config}: {e}")
                raise ValueError(f"Invalid custom config file: {custom_config}") from e

        if not isinstance(custom_config, dict):
            raise ValueError(f"Custom config must be dict or path to JSON file, got {type(custom_config)}")

        model_config = {
            "model_path": str(model_path),
            "custom_config": custom_config
        }
        
        return model_config, True

    def _validate_environment_and_model(self) -> None:
        """Validate environment and model configuration through testing."""
        logger.info("Validating environment and model configuration")

        try:
            # Test environment initialization
            self.rllib_env = SeniorEnvRllib(self.env_config)
            logger.info("SeniorEnvRllib initialized successfully")

            # Test environment execution
            obs, _ = self.rllib_env.reset()
            terminated, truncated = False, False
            step_count = 0
            max_steps = 100  # Prevent infinite loops

            while not (terminated or truncated) and step_count < max_steps:
                action = self.random.randrange(self.rllib_env.action_space.n)
                _, _, terminated, truncated, _ = self.rllib_env.step(action)
                step_count += 1

            logger.info(f"Environment test completed in {step_count} steps")

            # Reset for Ray environment check
            obs, _ = self.rllib_env.reset()
            check_env(self.rllib_env)
            logger.info("Ray environment validation passed")

            # Test model loading
            self._validate_tensorflow_model()
            self._validate_rllib_model(obs)

        except Exception as e:
            logger.error(f"Environment/model validation failed: {e}")
            raise RuntimeError(f"Senior initialization failed validation: {e}") from e

    def _validate_tensorflow_model(self) -> None:
        """Validate TensorFlow model loading."""
        try:
            model = tf.keras.models.load_model(self.model_config["model_path"])
            model.compile()
            logger.info("TensorFlow model validation passed")
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow model: {e}") from e

    def _validate_rllib_model(self, obs: Any) -> None:
        """Validate RLLib model initialization and forward pass."""
        try:
            if self._is_advanced_model:
                model = Grid2OpCustomModel(
                    obs_space=self.rllib_env.observation_space,
                    action_space=self.rllib_env.action_space,
                    num_outputs=self.rllib_env.action_space.n,
                    model_config={},
                    model_path=self.model_config["model_path"],
                    custom_config=self.model_config["custom_config"],
                    name="Senior"
                )
            else:
                model = Grid2OpCustomModel(
                    obs_space=self.rllib_env.observation_space,
                    action_space=self.rllib_env.action_space,
                    num_outputs=self.rllib_env.action_space.n,
                    model_config={},
                    model_path=self.model_config["model_path"],
                    name="Senior"
                )

            # Test forward pass
            obs_dict = {"obs": obs.reshape(1, -1)}
            output = model.forward(input_dict=obs_dict, state=[], seq_lens=None)
            
            if output is None:
                raise ValueError("Model forward pass returned None")
                
            logger.info("RLLib model validation passed")

        except Exception as e:
            raise ValueError(f"RLLib model validation failed: {e}") from e

    def _initialize_ppo_trainer(self, 
                               num_workers: Optional[int],
                               train_batch_size: int,
                               perturb_agent: Optional[BasePerturbationAgent]) -> tuple:
        """Initialize PPO trainer with configuration."""
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, os.cpu_count() // 2)

        logger.info(f"Configuring PPO with {num_workers} workers")

        # Build PPO configuration
        ppo_config = (
            PPOConfig()
            .environment(env=SeniorEnvRllib, env_config=self.env_config)
            .rollouts(num_rollout_workers=num_workers)
            .framework("tf2")
            .training(
                train_batch_size=train_batch_size,
                model={
                    "custom_model": "Senior",
                    "custom_model_config": self.model_config
                }
            )
            .evaluation(evaluation_num_workers=1)
        )

        # Add additional configuration
        ppo_config.update_from_dict({
            'seed': 0,
            "perturb_agent": perturb_agent
        })

        # Build trainer
        ppo_trainer = ppo_config.build()
        
        return ppo_config, ppo_trainer

    def _calculate_perturbation_threshold(self, 
                                        perturb_start: Optional[Union[int, float]], 
                                        total_iterations: int) -> int:
        """Calculate when to start perturbation-based training."""
        if perturb_start is None:
            return 0

        if isinstance(perturb_start, float):
            if not 0.0 <= perturb_start <= 1.0:
                raise ValueError(f"perturb_start fraction must be in [0.0, 1.0], got {perturb_start}")
            return int(perturb_start * total_iterations)

        if isinstance(perturb_start, int):
            if not 0 <= perturb_start <= total_iterations:
                raise ValueError(f"perturb_start must be in [0, {total_iterations}], got {perturb_start}")
            return perturb_start

        raise ValueError(f"perturb_start must be int, float, or None, got {type(perturb_start)}")