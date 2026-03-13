# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

"""
Grid2Op Curriculum Learning Agent for Robustness Testing

This module provides a production-ready implementation of the Curriculum Learning Agent
originally based on the submission from https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution.

The agent implements a multi-stage curriculum learning approach with:
- Teacher: Action space exploration and reduction
- Tutor: Experience generation with reduced action space  
- Junior: Supervised learning from tutor experience
- Senior: Reinforcement learning refinement

This version is optimized for robustness and resilience testing scenarios.
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple

import grid2op
import numpy as np
import ray
from grid2op.Agent import BaseAgent
from lightsim2grid import LightSimBackend

# Updated imports for production folder structure

from modified_curriculum_classes.my_agent import MyAgent
#from curriculumagent.senior.senior_student import Senior
#from curriculumagent.teacher.collect_teacher_experience import make_unitary_actionspace
#from curriculumagent.teacher.teacher import general_teacher
#from curriculumagent.tutor.collect_tutor_experience import generate_tutor_experience, prepare_dataset
# Training dependencies - NOT needed for inference/FAB deployment
Senior = None
make_unitary_actionspace = None
general_teacher = None
generate_tutor_experience = None
prepare_dataset = None
# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumAgent(BaseAgent):
    """
    Production-ready Curriculum Learning Agent for Grid2Op environments.
    
    This agent implements a comprehensive curriculum learning pipeline designed for
    robustness testing in power grid management scenarios. It combines multiple
    learning stages to create a robust agent capable of handling various perturbations
    and adversarial conditions.
    
    The agent architecture consists of four main components:
    1. Teacher: Explores and identifies effective actions through greedy search
    2. Tutor: Generates training experience using the reduced action space
    3. Junior: Learns action selection through supervised learning
    4. Senior: Refines policy using reinforcement learning
    
    Key Features:
    - Modular design allowing individual component training
    - Support for action space reduction for computational efficiency
    - Compatible with various Grid2Op environments
    - Optimized for robustness testing scenarios
    
    Example:
        >>> agent = CurriculumAgent(env.action_space, env.observation_space, "test_agent")
        >>> agent.load("path/to/model", "path/to/actions")
        >>> action = agent.act(observation, reward, done)
    """

    def __init__(self,
                 action_space: grid2op.Action.ActionSpace,
                 observation_space: grid2op.Observation.BaseObservation,
                 name: str,
                 **kwargs):
        """
        Initialize the Curriculum Learning Agent.
        
        Args:
            action_space: Grid2Op action space defining possible actions
            observation_space: Grid2Op observation space defining state representation
            name: Unique identifier for this agent instance
            **kwargs: Additional configuration parameters for the underlying MyAgent
                     - subset (bool): Whether to use observation subset for efficiency
                     - topo (bool): Whether to include topology actions
                     - device (str): Computing device ('cpu' or 'cuda')
        """
        super().__init__(action_space)
        
        self.name = name
        self.observation_space = observation_space
        self.agent: Optional[MyAgent] = None
        self.senior: Optional[Senior] = None
        self.ckpt_path: Optional[Path] = None
        
        # Initialize default "do nothing" action
        self.do_nothing = self.action_space({})
        
        # Store additional configuration parameters
        self._config_kwargs = kwargs.copy() if kwargs else {}
        
        logger.info(f"Initialized CurriculumAgent '{name}' with config: {self._config_kwargs}")

    def act(self, 
            observation: grid2op.Observation.BaseObservation,
            reward: float = 0.0, 
            done: bool = False, 
            simulated_act: bool = False) -> grid2op.Action.BaseAction:
        """
        Select an action based on the current observation.
        
        This method serves as the main interface for action selection during
        both training and evaluation phases.
        
        Args:
            observation: Current grid state observation
            reward: Reward received from previous action (unused in inference)
            done: Whether the episode has terminated (unused in inference)
            simulated_act: Whether this is a simulated action for planning
            
        Returns:
            Grid2Op action to be executed in the environment
            
        Raises:
            Warning: If agent hasn't been loaded, returns do-nothing action
        """
        if self.agent is None:
            warnings.warn(
                "Agent not loaded yet. Returning do-nothing action. "
                "Please call load() method to initialize the agent."
            )
            return self.do_nothing
        
        try:
            action = self.agent.act(
                observation=observation, 
                reward=reward, 
                done=done, 
                simulated_act=simulated_act
            )
            return action
        except Exception as e:
            logger.error(f"Error during action selection: {e}")
            logger.warning("Falling back to do-nothing action")
            return self.do_nothing

    def reset(self, observation: grid2op.Observation.BaseObservation) -> None:
        """
        Reset agent state at the beginning of a new episode.
        
        This method is called by the Grid2Op environment when starting
        a new episode to ensure clean agent state initialization.
        
        Args:
            observation: Initial observation of the new episode
        """
        if self.agent is not None:
            self.agent.reset(observation)
        else:
            logger.warning("Attempted to reset unloaded agent")

    def load(self, 
             model_path: Union[str, Path],
             actions_path: Optional[Union[str, Path, List[Path]]] = None,
             **kwargs) -> None:
        """
        Load a trained curriculum agent from saved files.
        
        This method initializes the agent with pre-trained models and action spaces.
        It supports loading from various file structures and configurations.
        
        Args:
            model_path: Directory containing the trained model files
                       Should contain 'saved_model.pb' and related TensorFlow files
            actions_path: Path to action space file(s) (.npy format)
                         If None, looks for 'actions' subdirectory in model_path
            **kwargs: Additional configuration parameters:
                     - subset (bool): Use observation subset for efficiency
                     - topo (bool): Include topology actions
                     - device (str): Computing device specification
                     
        Raises:
            FileNotFoundError: If required model or action files are missing
            ValueError: If file structure is invalid
        """
        # Validate and resolve file paths
        actions_path, model_path = self._validate_paths(model_path, actions_path)
        
        # Merge configuration parameters
        config = self._config_kwargs.copy()
        config.update(kwargs)
        
        logger.info(f"Loading agent from model: {model_path}, actions: {actions_path}")
        
        try:
            # Initialize the underlying MyAgent with validated paths
            self.agent = MyAgent(
                action_space=self.action_space,
                model_path=model_path,
                action_space_path=actions_path,
                **config
            )
            logger.info(f"Successfully loaded CurriculumAgent '{self.name}'")
            
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            raise

    def save(self, save_path: Union[str, Path]) -> None:
        """
        Save the current agent model and action space to disk.
        
        Creates a standardized directory structure with model and action files
        suitable for later loading or deployment.
        
        Args:
            save_path: Directory where to save the agent files
                      Creates 'model/' and 'actions/' subdirectories
                      
        Raises:
            ValueError: If agent hasn't been initialized
            OSError: If directory creation or file writing fails
        """
        if self.agent is None:
            raise ValueError(
                "Cannot save uninitialized agent. Please load or train the agent first."
            )

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create and save model files
        model_path = save_path / "model"
        model_path.mkdir(exist_ok=True)
        
        try:
            self.agent.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        # Create and save action space files
        actions_path = save_path / "actions"
        actions_path.mkdir(exist_ok=True)
        
        try:
            actions_file = actions_path / "actions.npy"
            np.save(actions_file, self.agent.actions)
            logger.info(f"Action space saved to {actions_file}")
        except Exception as e:
            logger.error(f"Failed to save actions: {e}")
            raise

    def train(self, 
              env: Union[grid2op.Environment.BaseEnv, str] = "l2rpn_case14_sandbox",
              iterations: int = 100,
              save_path: Union[Path, str] = None,
              **kwargs_senior) -> None:
        """
        Train the Senior component using reinforcement learning.
        
        This method performs incremental training of the Senior (RL) component
        while keeping the existing action space and Junior model weights.
        It requires the agent to be pre-loaded with a Junior model.
        
        Args:
            env: Grid2Op environment name or instance for training
            iterations: Number of RL training iterations to perform
            save_path: Directory to save training checkpoints and final model
                      Defaults to current directory + 'training_output'
            **kwargs_senior: Additional parameters for Senior training:
                           - num_workers: Number of parallel workers
                           - learning_rate: RL learning rate
                           - batch_size: Training batch size
                           
        Raises:
            AssertionError: If agent hasn't been loaded
            RuntimeError: If Ray initialization or training fails
        """
        # Validate inputs and setup paths
        if save_path is None:
            save_path = Path.cwd() / "training_output"
        save_path = Path(save_path)
        
        if isinstance(env, grid2op.Environment.BaseEnv):
            env_path = env.get_path_env()
        else:
            env_path = env

        assert self.agent is not None, (
            "Agent must be loaded before training. Please call load() first."
        )
        
        # Save current state to ensure consistency
        logger.info("Saving current agent state before training")
        self.save(save_path)
        
        # Validate saved files
        actions_path, model_path = self._validate_paths(save_path)

        # Setup training directory structure
        ckpt_path = save_path / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray for distributed training
        if not ray.is_initialized():
            logger.info("Initializing Ray for distributed training")
            ray.init()

        try:
            # Initialize and configure Senior trainer
            logger.info(f"Initializing Senior trainer with {iterations} iterations")
            self.senior = Senior(
                env_path=env_path,
                action_space_path=actions_path,
                model_path=model_path,
                ckpt_save_path=ckpt_path,
                **kwargs_senior
            )

            # Execute training
            logger.info("Starting Senior RL training")
            self.senior.train(iterations=iterations)
            
            # Update agent with trained model
            logger.info("Training completed. Updating agent with new model")
            self.agent = self.senior.get_my_agent(path=model_path)
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Ensure Ray cleanup
            if ray.is_initialized():
                ray.shutdown()

    def train_full_pipeline(
            self,
            env: Union[str, grid2op.Environment.BaseEnv] = "l2rpn_case14_sandbox",
            iterations: int = 100,
            save_path: Optional[Path] = None,
            **kwargs,
    ) -> MyAgent:
        """
        Execute the complete curriculum learning pipeline from scratch.
        
        This method implements the full Teacher-Tutor-Junior-Senior training sequence:
        1. Teacher: Discovers effective actions through exploration
        2. Tutor: Generates training experience using discovered actions
        3. Junior: Learns action mapping through supervised learning
        4. Senior: Refines policy using reinforcement learning
        
        Args:
            env: Grid2Op environment for training (name or instance)
            iterations: Number of training episodes/iterations per stage
            save_path: Directory for saving all training artifacts
                      Defaults to current directory + 'full_pipeline_output'
            **kwargs: Configuration parameters:
                     - seed (int): Random seed for reproducibility (default: 42)
                     - jobs (int): Number of parallel jobs (default: CPU count)
                     - max_actionspace_size (int): Maximum actions to keep (default: 250)
                     - epochs (int): Junior training epochs (default: 10 + iterations)
                     
        Returns:
            Trained MyAgent instance ready for deployment
            
        Raises:
            ValueError: If environment cannot be initialized
            FileNotFoundError: If required components are missing
        """
        # Setup configuration and paths
        if save_path is None:
            save_path = Path.cwd() / "full_pipeline_output"
        save_path = Path(save_path)
        
        # Extract configuration parameters
        config = self._extract_pipeline_config(kwargs)
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        
        # Setup environment and validation
        env_path = self._setup_environment(env)
        
        # Create directory structure
        self._create_pipeline_directories(save_path)
        
        logger.info(f"Starting full pipeline training with {iterations} iterations")
        logger.info(f"Output directory: {save_path}")
        logger.info(f"Configuration: {config}")
        
        try:
            # Stage 1: Teacher - Action Discovery
            teacher_path = self._train_teacher_stage(
                save_path, env_path, iterations, config['jobs']
            )
            
            # Stage 2: Action Space Reduction
            actions_path = self._create_action_space(
                save_path, teacher_path, env_path, config['max_actionspace_size']
            )
            
            # Stage 3: Tutor - Experience Generation
            tutor_path = self._train_tutor_stage(
                save_path, env_path, actions_path, iterations, config
            )
            
            # Stage 4: Junior - Supervised Learning
            junior_path = self._train_junior_stage(
                save_path, tutor_path, actions_path, config, iterations
            )
            
            # Stage 5: Senior - Reinforcement Learning
            agent = self._train_senior_stage(
                save_path, env_path, actions_path, junior_path, iterations
            )
            
            # Stage 6: Final Model Preparation
            self._finalize_model(save_path, agent)
            
            logger.info("Full pipeline training completed successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Pipeline training failed: {e}")
            raise
        finally:
            # Cleanup Ray if initialized
            if ray.is_initialized():
                ray.shutdown()

    def create_deployment_package(self, package_path: Union[str, Path] = "./deployment") -> None:
        """
        Create a deployment-ready package with all necessary files.
        
        This method packages the trained agent with all dependencies
        into a standardized structure suitable for production deployment.
        
        Args:
            package_path: Directory where to create the deployment package
                         
        Raises:
            ValueError: If agent hasn't been trained/loaded
            OSError: If file operations fail
        """
        if self.agent is None:
            raise ValueError(
                "Cannot create deployment package for uninitialized agent. "
                "Please load or train the agent first."
            )

        package_path = Path(package_path)
        package_path.mkdir(parents=True, exist_ok=True)

        # Save the core agent
        self.save(package_path)

        # Create required directory structure
        common_path = package_path / "common"
        common_path.mkdir(exist_ok=True)

        # Copy utility modules
        source_path = Path(__file__).parent.parent
        
        try:
            # Copy common utilities
            shutil.copy(source_path / "curriculum_agent" / "common" / "__init__.py", common_path)
            shutil.copy(source_path / "curriculum_agent" / "common" / "obs_converter.py", common_path)
            shutil.copy(source_path / "curriculum_agent" / "common" / "utilities.py", common_path)

            # Copy agent implementation
            shutil.copy(source_path / "curriculum_agent" / "submission" / "my_agent.py", package_path)
            shutil.copy(source_path / "curriculum_agent" / "submission" / "__init__.py", package_path)

            logger.info(f"Deployment package created at {package_path}")
            logger.info("Note: Review my_agent.py for any custom configuration requirements")
            
        except Exception as e:
            logger.error(f"Failed to create deployment package: {e}")
            raise

    # Private helper methods
    
    def _validate_paths(self, 
                       model_path: Union[str, Path],
                       actions_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Validate and resolve model and action file paths."""
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Resolve actions path
        if actions_path is None:
            actions_path = model_path / "actions"

        if not isinstance(actions_path, list):
            actions_path = Path(actions_path)
            if not actions_path.is_dir():
                logger.warning(f"Actions directory not found: {actions_path}. Trying parent directory.")
                actions_path = model_path

            if not any(".npy" in f for f in os.listdir(actions_path)):
                raise FileNotFoundError(f"No .npy action files found in {actions_path}")
        else:
            if not any(".npy" in str(p) for p in actions_path):
                raise FileNotFoundError(f"No .npy files in action path list: {actions_path}")

        # Resolve model path
        final_model_path = model_path / "model"
        if not final_model_path.is_dir():
            logger.warning(f"Model subdirectory not found: {final_model_path}. Using parent directory.")
            final_model_path = model_path

        if not any("saved_model.pb" in f for f in os.listdir(final_model_path)):
            raise FileNotFoundError(f"No TensorFlow model found in {final_model_path}")

        return actions_path, final_model_path

    def _extract_pipeline_config(self, kwargs: dict) -> dict:
        """Extract and validate configuration parameters for pipeline training."""
        return {
            'seed': kwargs.get("seed", 42),
            'jobs': kwargs.get("jobs", os.cpu_count()),
            'max_actionspace_size': kwargs.get("max_actionspace_size", 250),
            'epochs': kwargs.get("epochs", None)  # Will be set later if None
        }

    def _setup_environment(self, env: Union[str, grid2op.Environment.BaseEnv]) -> str:
        """Setup and validate the Grid2Op environment."""
        if isinstance(env, grid2op.Environment.BaseEnv):
            env_path = env.get_path_env()
        else:
            env_path = env

        try:
            # Test environment initialization
            test_env = grid2op.make(env_path, backend=LightSimBackend())
            test_env.close()
            logger.info(f"Environment validation successful: {env_path}")
            return env_path
        except Exception as e:
            raise ValueError(f"Failed to initialize Grid2Op environment '{env_path}': {e}")

    def _create_pipeline_directories(self, save_path: Path) -> None:
        """Create the directory structure for pipeline training."""
        subdirs = ["teacher", "tutor", "junior", "senior", "model", "actions"]
        for subdir in subdirs:
            (save_path / subdir).mkdir(exist_ok=True, parents=True)
        logger.info(f"Created pipeline directories in {save_path}")

    def _train_teacher_stage(self, save_path: Path, env_path: str, iterations: int, jobs: int) -> Path:
        """Execute the Teacher training stage."""
        teacher_experience_path = save_path / "teacher" / "general_teacher_experience.csv"
        
        if not teacher_experience_path.exists():
            logger.info("Stage 1: Training Teacher (action discovery)")
            general_teacher(
                save_path=teacher_experience_path,
                env_name_path=env_path,
                n_episodes=iterations,
                jobs=jobs,
            )
        else:
            logger.info(f"Stage 1: Skipping Teacher (experience exists at {teacher_experience_path})")
        
        return teacher_experience_path

    def _create_action_space(self, save_path: Path, teacher_path: Path, env_path: str, max_size: int) -> Path:
        """Create reduced action space from teacher experience."""
        actions_path = save_path / "actions" / "actions.npy"
        
        if not actions_path.exists():
            logger.info("Stage 2: Creating reduced action space")
            make_unitary_actionspace(
                actions_path, [teacher_path], env_path, best_n=max_size
            )
        else:
            logger.info(f"Stage 2: Skipping action space creation (exists at {actions_path})")
        
        return actions_path

    def _train_tutor_stage(self, save_path: Path, env_path: str, actions_path: Path, iterations: int, config: dict) -> Path:
        """Execute the Tutor training stage."""
        tutor_experience_path = save_path / "tutor" / "tutor_experience.npy"
        
        if not tutor_experience_path.exists():
            logger.info("Stage 3: Training Tutor (experience generation)")
            generate_tutor_experience(
                env_path,
                tutor_experience_path,
                actions_path,
                num_chronics=iterations,
                jobs=config['jobs'],
                seed=config['seed'],
            )
        else:
            logger.info(f"Stage 3: Skipping Tutor (experience exists at {tutor_experience_path})")
        
        return tutor_experience_path

    def _train_junior_stage(self, save_path: Path, tutor_path: Path, actions_path: Path, config: dict, iterations: int) -> Path:
        """Execute the Junior training stage."""
        junior_data_path = save_path / "tutor" / "junior_data"
        junior_results_path = save_path / "junior"
        junior_data_path.mkdir(exist_ok=True, parents=True)
        
        # Prepare dataset
        prepare_dataset(
            traindata_path=tutor_path.parent,
            target_path=junior_data_path,
            dataset_name="training",
        )

        if not (junior_results_path / "saved_model.pb").exists():
            logger.info("Stage 4: Training Junior (supervised learning)")
            epochs = config['epochs'] if config['epochs'] is not None else 10 + iterations
            
            # Import train_junior function (assuming it exists in the framework)
            from curriculumagent.junior.junior_student import train_junior
            
            train_junior(
                run_name="junior",
                dataset_path=junior_data_path,
                target_model_path=junior_results_path,
                action_space_file=actions_path,
                dataset_name="training",
                epochs=epochs,
                seed=config['seed'],
            )
        else:
            logger.info(f"Stage 4: Skipping Junior (model exists at {junior_results_path})")
        
        return junior_results_path

    def _train_senior_stage(self, save_path: Path, env_path: str, actions_path: Path, junior_path: Path, iterations: int) -> MyAgent:
        """Execute the Senior training stage."""
        senior_results_path = save_path / "senior"
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()

        # Calculate resources
        resources = ray.nodes()
        num_workers = int(resources[0]["Resources"]["CPU"] // 2)

        # Initialize Senior
        senior = Senior(
            env_path=env_path,
            action_space_path=actions_path,
            model_path=junior_path,
            ckpt_save_path=senior_results_path,
            num_workers=num_workers,
            subset=False
        )

        if not (senior_results_path / "sandbox").exists():
            logger.info("Stage 5: Training Senior (reinforcement learning)")
            senior.train(iterations=iterations)
        else:
            logger.info(f"Stage 5: Loading existing Senior from {senior_results_path}")
            senior.restore(senior_results_path)
            logger.info(f"Senior trained for {senior.ppo.iteration} iterations")

        return senior

    def _finalize_model(self, save_path: Path, senior: Senior) -> None:
        """Finalize and save the complete trained model."""
        agent_path = save_path / "model"
        logger.info("Stage 6: Finalizing trained agent")
        
        self.agent = senior.get_my_agent(path=agent_path)
        self.save(save_path)
        
        logger.info("Full pipeline training completed successfully")