"""
Grid2Op Agent Implementation

This module provides a sophisticated Grid2Op agent capable of handling various action spaces,
different model types (Keras and RLib), observation scaling, and advanced action simulation.
The agent is designed for robustness testing and can handle both single and multi-step actions.

Key Features:
- Support for multiple model architectures (Keras Sequential, Functional, RLib AutoTrackable)
- Advanced action simulation and validation
- Configurable observation preprocessing and scaling
- Multi-step action sequences (tuples and triples)
- Topology recovery and line reconnection logic
- Comprehensive error handling and logging

Original work based on: https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
Licensed under Mozilla Public License 2.0
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, List, Tuple
import traceback
import grid2op
import numpy as np
import tensorflow as tf
from grid2op.Agent import BaseAgent
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model
from tensorflow.python.training.tracking.tracking import AutoTrackable

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import from production folder structure
    from curriculumagent.common.obs_converter import obs_to_vect
    from curriculumagent.common.utilities import (
        find_best_line_to_reconnect,
        is_legal,
        split_action_and_return,
        simulate_action,
        revert_topo,
    )
except ImportError:
    try:
        # Fallback for deployment package structure
        from .obs_converter import obs_to_vect
        from .utilities import (
            find_best_line_to_reconnect,
            is_legal,
            split_action_and_return,
            simulate_action,
            revert_topo,
        )
    except ImportError:
        # Final fallback for local development
        from common.obs_converter import obs_to_vect
        from common.utilities import (
            find_best_line_to_reconnect,
            is_legal,
            split_action_and_return,
            simulate_action,
            revert_topo,
        )


class MyAgent(BaseAgent):
    """
    Advanced Grid2Op agent with support for multiple model types and action strategies.
    
    This agent implements sophisticated action selection logic including:
    - Multi-step action sequences (single, tuple, triple actions)
    - Action simulation and validation before execution
    - Configurable observation preprocessing and scaling
    - Topology recovery mechanisms
    - Overload detection and mitigation strategies
    
    The agent can work with different model architectures:
    - Keras Sequential models (Junior models)
    - Keras Functional models (Senior models)
    - RLib AutoTrackable models (Legacy Senior models)
    
    Example:
        >>> agent = MyAgent(
        ...     action_space=env.action_space,
        ...     model_path="path/to/model",
        ...     action_space_path="path/to/actions",
        ...     subset=True,
        ...     topo=True
        ... )
        >>> action = agent.act(observation, reward, done)
    """

    def __init__(
            self,
            action_space: grid2op.Action.ActionSpace,
            model_path: Union[Path, str],
            action_space_path: Optional[Union[Path, List[Path]]] = None,
            this_directory_path: Optional[str] = "./",
            subset: Optional[Union[bool, List[int]]] = False,
            scaler: Optional[BaseEstimator] = None,
            best_action_threshold: float = 0.95,
            topo: Optional[bool] = False,
            check_overload: Optional[bool] = False,
            max_action_sim: Optional[int] = 50,
            action_space_file: Optional[str] = None
    ):
        """
        Initialize the advanced Grid2Op agent.

        This agent supports multiple model types and advanced action selection strategies
        including tuple and triple actions, observation preprocessing, and topology management.

        Args:
            action_space: Grid2Op action space defining possible actions
            model_path: Path to the trained model (Keras or RLib format)
            action_space_path: Path to action space files (.npy format)
                              Can be a directory or specific file path
            this_directory_path: Working directory for relative path resolution
            subset: Observation filtering configuration:
                   - False: Use full observation vector
                   - True: Use predefined subset for efficiency
                   - List[int]: Use specific observation indices
            scaler: Optional sklearn scaler for observation preprocessing
            best_action_threshold: Line load threshold for action selection (0.0-1.0)
                                  Actions are only considered if they improve load below this value
            topo: Whether to attempt topology restoration when conditions are safe
            check_overload: Whether to simulate stress testing during action evaluation
            max_action_sim: Maximum number of actions to simulate per decision
                           Lower values increase speed but may reduce optimality
            action_space_file: Alternative to action_space_path for direct file specification

        Raises:
            FileNotFoundError: If model or action files cannot be found
            ValueError: If action space configuration is invalid
        """
        super().__init__(action_space=action_space)

        # Resolve action space path
        if isinstance(action_space_file, str):
            action_space_path = Path(action_space_file)

        # Load and validate action space
        self.actions = self._load_action_space(this_directory_path, action_space_path)
        logger.info(f"Loaded {self.actions.shape[0]} actions from action space")

        # Store configuration parameters
        self.subset = subset
        self.check_overload = check_overload
        self.best_action_threshold = best_action_threshold
        self.max_action_sim = max_action_sim
        self.topo = bool(topo)

        # Load and compile model
        self.model = self._load_model(model_path)
        self.scaler = scaler

        # Initialize agent state
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None

        logger.info(f"MyAgent initialized with {type(self.model).__name__} model")

    def act_with_id(
            self, 
            observation: grid2op.Observation.BaseObservation, 
            simulated_act: bool = False
    ) -> Tuple[grid2op.Action.BaseAction, int]:
        """
        Select an action and return both the action and its ID for tracking.

        This method implements the core action selection logic including:
        - Multi-step action sequence management
        - Load threshold-based decision making
        - Action simulation and validation
        - Topology recovery when safe

        Args:
            observation: Current grid state observation
            simulated_act: Whether this is a simulated action (affects state tracking)

        Returns:
            Tuple of (selected_action, action_id) where action_id is -1 for default actions
        """
        
        # Handle continuing multi-step action sequences
        if self.next_actions is not None:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        # Track overload conditions
        if not simulated_act:
            if observation.rho.max() >= 1.0:
                self.overflow_steps += 1
            else:
                self.overflow_steps = 0

        # Case 1: Safe conditions - use minimal intervention
        if observation.rho.max() < self.best_action_threshold:
            if self.topo:
                # Attempt topology restoration
                action_array = revert_topo(self.action_space, observation)
                default_action = self.action_space.from_vect(action_array)
            else:
                # Do nothing action
                default_action = self.action_space({})
            
            default_action = find_best_line_to_reconnect(obs=observation, original_action=default_action)
            return default_action, -1

        # Case 2: Dangerous conditions - active intervention required
        current_max_rho = observation.rho.max()
        logger.info(
            f"Time {observation.get_time_stamp()}: Critical load detected - "
            f"Line {observation.rho.argmax()} at {observation.rho.max():.3f}"
        )

        best_action_id = None
        min_rho_achieved = current_max_rho

        # Get model-recommended actions
        sorted_actions = self._get_model_actions(observation)[:self.max_action_sim]
        
        # Simulate and evaluate actions
        for simulation_count, action_id in enumerate(sorted_actions):
            action_vector = self.actions[action_id, :]
            
            # Simulate action to predict its effect
            predicted_rho, is_valid = simulate_action(
                action_space=self.action_space,
                obs=observation,
                action_vect=action_vector,
                check_overload=self.check_overload
            )
            
            
            if not is_valid:
                continue

            # Check if action meets safety threshold
            if predicted_rho <= self.best_action_threshold:
                logger.info(
                    f"Optimal action found: ID={action_id}, predicted_rho={predicted_rho:.3f}, "
                    f"simulations={simulation_count + 1}"
                )
                best_action_id = action_id
                break

            # Track best improvement even if not optimal
            if predicted_rho < min_rho_achieved:
                min_rho_achieved = predicted_rho
                best_action_id = action_id

        # Execute selected action
        if best_action_id is not None:
            if simulated_act:
                # For simulation, return single action from sequence
                next_action = next(split_action_and_return(
                    observation, self.action_space, self.actions[best_action_id, :]
                ))
            else:
                # For real execution, setup action sequence
                self.next_actions = split_action_and_return(
                    observation, self.action_space, self.actions[best_action_id, :]
                )
                next_action = next(self.next_actions)
            
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
        else:
            # Fallback to do-nothing with line reconnection
            logger.warning("No suitable action found, using do-nothing with line reconnection")
            next_action = find_best_line_to_reconnect(
                obs=observation, 
                original_action=self.action_space({})
            )
            best_action_id = -1
        
        return next_action, best_action_id

    def act(
            self, 
            observation: grid2op.Observation.BaseObservation, 
            reward: float, 
            done: bool, 
            simulated_act: bool = False
    ) -> grid2op.Action.BaseAction:
        """
        Main action selection interface for Grid2Op environments.

        Args:
            observation: Current grid state
            reward: Previous step reward (unused)
            done: Whether episode is complete (unused)
            simulated_act: Whether this is a simulated action

        Returns:
            Grid2Op action to execute
        """
        action, _ = self.act_with_id(observation, simulated_act)
        return action

    def reset(self, observation: grid2op.Observation.BaseObservation) -> None:
        """
        Reset agent state for new episode.

        Args:
            observation: Initial observation of new episode
        """
        self.next_actions = None
        self.overflow_steps = 0
        self.recovery_stack.clear()
        logger.debug("Agent state reset for new episode")

    def _load_action_space(
            self, 
            directory_path: str, 
            action_space_path: Union[Path, List[Path]]
    ) -> np.ndarray:
        """
        Load and validate action space from file(s).

        Args:
            directory_path: Base directory for relative paths
            action_space_path: Path(s) to action space files

        Returns:
            Numpy array containing all available actions

        Raises:
            FileNotFoundError: If action files cannot be found
            ValueError: If action space configuration is invalid
        """
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logger.info(f"Loading actions from file: {action_space_path}")
                return np.load(str(action_space_path))
            
            elif action_space_path.is_dir():
                logger.info(f"Loading actions from directory: {action_space_path}")
                action_files = [
                    f for f in os.listdir(action_space_path) 
                    if "actions" in f and f.endswith(".npy")
                ]
                
                if not action_files:
                    raise FileNotFoundError(f"No action files found in {action_space_path}")

                loaded_actions = []
                for filename in action_files:
                    file_path = action_space_path / filename
                    loaded_actions.append(np.load(file_path))
                    logger.debug(f"Loaded actions from {filename}")

                return np.concatenate(loaded_actions, axis=0)

        elif isinstance(action_space_path, list):
            logger.info(f"Loading actions from {len(action_space_path)} files")
            loaded_actions = []
            
            for path in action_space_path:
                path = Path(path)
                if not path.is_file():
                    raise FileNotFoundError(f"Action file not found: {path}")
                loaded_actions.append(np.load(str(path)))
            
            return np.concatenate(loaded_actions, axis=0)

        else:
            raise ValueError(
                f"Invalid action_space_path type: {type(action_space_path)}. "
                f"Expected Path, List[Path], or valid path string."
            )

    def _load_model(self, model_path: Union[Path, str]) -> Union[Model, AutoTrackable]:
        """
        Load and initialize the neural network model.

        Args:
            model_path: Path to the trained model

        Returns:
            Loaded TensorFlow model

        Raises:
            FileNotFoundError: If model files cannot be found
        """
        model_path = Path(model_path)
        
        try:
            # Try loading as Keras model first (newer format)
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile()
            logger.info(f"Loaded Keras model from {model_path}")
            return model
            
        except (OSError, IOError, ValueError):
            try:
                # Fallback to SavedModel format (older RLib models)
                model = tf.saved_model.load(str(model_path))
                logger.info(f"Loaded SavedModel from {model_path}")
                return model
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load model from {model_path}. "
                    f"Ensure the path contains valid TensorFlow model files. Error: {e}"
                )

    def _get_model_actions(self, observation: grid2op.Observation.BaseObservation) -> np.ndarray:
        """
        Get action probabilities from the neural network model.

        Args:
            observation: Current grid observation

        Returns:
            Array of action indices sorted by predicted value (best first)
        """
        if isinstance(self.model, Model):
            return self._get_keras_model_actions(observation)
        else:
            return self._get_saved_model_actions(observation)

    def _get_keras_model_actions(self, observation: grid2op.Observation.BaseObservation) -> np.ndarray:
        """
        Get actions from Keras model (Junior or Senior).

        Args:
            observation: Current grid observation

        Returns:
            Sorted action indices
        """
        # Prepare model input based on subset configuration
        if isinstance(self.subset, list):
            model_input = observation.to_vect()[self.subset]
        elif self.subset:
            model_input = obs_to_vect(observation, False)
        else:
            model_input = observation.to_vect()

        # Apply scaling if configured
        if self.scaler is not None:
            model_input = self.scaler.transform(model_input.reshape(1, -1))

        model_input = model_input.reshape((1, -1))

        # Get predictions based on model type
        if isinstance(self.model, tf.keras.models.Sequential):
            # Junior model - Sequential architecture
            action_probabilities = self.model.predict(model_input, verbose=0)
        else:
            # Senior model - Functional architecture
            logits, _ = self.model(model_input, training=False)
            action_probabilities = tf.nn.softmax(logits).numpy().reshape(-1)

        # Return actions sorted by probability (highest first)
        return action_probabilities.argsort()[::-1]

    def _get_saved_model_actions(self, observation: grid2op.Observation.BaseObservation) -> np.ndarray:
        """Get actions from SavedModel format (legacy RLib models or SAC)."""
        
        # Prepare input
        if isinstance(self.subset, list):
            model_input = observation.to_vect()[self.subset]
        elif self.subset:
            model_input = obs_to_vect(observation, False)
        else:
            model_input = observation.to_vect()
        
        if self.scaler is not None:
            model_input = self.scaler.transform(model_input.reshape(1, -1)).reshape(-1)
        
        # CRITICAL: Convert to float32 (models expect float32, not float64)
        model_input = model_input.astype(np.float32)
        
        # Call SavedModel signature
        signature_fn = self.model.signatures["serving_default"]
        
        # Detect signature and pass appropriate arguments
        try:
            # Try RLib-style call first (with timestep and is_training)
            output = signature_fn(
                observations=tf.convert_to_tensor(model_input.reshape(1, -1), dtype=tf.float32),  # ← Add dtype
                timestep=tf.convert_to_tensor(0, dtype=tf.int64),
                is_training=tf.convert_to_tensor(False),
            )
        except (TypeError, ValueError) as e:
            if "timestep" in str(e) or "unexpected keyword argument" in str(e):
                # SAC-style or simpler model (observations only)
                logger.debug("Model doesn't accept timestep parameter, using simple signature")
                output = signature_fn(
                    observations=tf.convert_to_tensor(model_input.reshape(1, -1), dtype=tf.float32)  # ← Add dtype
                )
            else:
                raise
        
        # Extract action logits - handle different output formats
        output_keys = list(output.keys())
        
        if "action_dist_inputs" in output:
            action_logits = output["action_dist_inputs"]
        elif "action_out" in output:
            action_logits = output["action_out"]
        elif "logits" in output:
            action_logits = output["logits"]
        elif "output_0" in output:
            action_logits = output["output_0"]
        elif len(output_keys) == 1:
            action_logits = output[output_keys[0]]
        else:
            action_keys = [k for k in output_keys if 'action' in k.lower()]
            if action_keys:
                action_logits = output[action_keys[0]]
            else:
                logger.error(f"Unknown output format. Available keys: {output_keys}")
                raise KeyError(f"Cannot find action logits in model output. Available keys: {output_keys}")
        
        # Convert to probabilities
        action_probabilities = tf.nn.softmax(action_logits).numpy().reshape(-1)
        
        return action_probabilities.argsort()[::-1]


def make_agent(env: grid2op.Environment.BaseEnv, this_directory_path: Union[str, Path]) -> MyAgent:
    """
    Factory function to create a standard agent configuration.
    
    This function provides a default configuration suitable for most deployment scenarios.
    It assumes the standard directory structure with 'model' and 'actions' subdirectories.

    Args:
        env: Grid2Op environment instance
        this_directory_path: Path to directory containing model and actions

    Returns:
        Configured MyAgent instance

    Example:
        >>> env = grid2op.make("l2rpn_case14_sandbox")
        >>> agent = make_agent(env, "./trained_agent")
        >>> action = agent.act(env.reset(), 0, False)
    """
    this_directory_path = Path(this_directory_path)
    
    agent = MyAgent(
        action_space=env.action_space,
        model_path=this_directory_path / "model",
        action_space_path=this_directory_path / "actions",
        this_directory_path=str(this_directory_path),
        subset=True,  # Use observation subset for efficiency
    )
    
    logger.info(f"Created standard agent from {this_directory_path}")
    return agent