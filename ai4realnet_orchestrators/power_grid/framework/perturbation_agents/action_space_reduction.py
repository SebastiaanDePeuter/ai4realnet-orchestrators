"""
Teacher Experience Collection Module for Grid2Op Curriculum Learning

This module implements the Teacher component of the curriculum learning framework,
which explores and identifies effective actions through systematic search and simulation.
The Teacher generates training experience by evaluating perturbation strategies and
their impact on grid stability and agent performance.

Key Features:
- Systematic exploration of perturbation effects on grid observations
- Multi-process experience collection for efficiency
- Action ranking based on reward improvement
- Support for missing value and large value perturbations
- Comprehensive logging and progress tracking
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import datetime as dt

import grid2op.Environment
import grid2op.Observation
from modified_curriculum_classes.baseline import CurriculumAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeacherExperienceCollector:
    """
    Collects teacher experience by evaluating perturbation effects on agent performance.
    
    The Teacher systematically applies perturbations to grid observations and evaluates
    how different perturbation strategies affect agent decision-making and grid stability.
    This experience is used to train more robust agents in subsequent curriculum stages.
    """

    def __init__(self, 
                 rho_threshold: float = 0.95,
                 top_k_actions: int = 125,
                 max_substations_changed: int = 99,
                 max_lines_changed: int = 99):
        """
        Initialize the teacher experience collector.

        Args:
            rho_threshold: Line load threshold above which perturbations are applied
            top_k_actions: Number of best actions to save per step
            max_substations_changed: Maximum number of substations that can be modified
            max_lines_changed: Maximum number of lines that can be modified
        """
        self.rho_threshold = rho_threshold
        self.top_k_actions = top_k_actions
        self.max_substations_changed = max_substations_changed
        self.max_lines_changed = max_lines_changed

    def collect_experience(self,
                          env: grid2op.Environment.Environment,
                          agent: CurriculumAgent,
                          perturbation_actions: List[Tuple[str, List[int]]],
                          save_path: Path,
                          observation_attributes: Tuple[List[str], Dict[str, int]],
                          seed: int = 42,
                          chronic_limit: Optional[int] = None) -> None:
        """
        Collect teacher experience across multiple chronics.

        Args:
            env: Grid2Op environment for simulation
            agent: Trained curriculum agent for action selection
            perturbation_actions: List of (perturbation_type, indices) tuples
            save_path: Path to save collected experience
            observation_attributes: Tuple of (attribute_list, attribute_start_indices)
            seed: Random seed for reproducibility
            chronic_limit: Maximum number of chronics to process (None for all)
        """
        attr_list, attr_start_idx = observation_attributes
        
        # Configure environment parameters
        self._configure_environment(env)
        
        # Setup chronics with deterministic shuffling
        self._setup_chronics(env, seed)
        
        num_chronics = self._get_chronic_count(env, chronic_limit)
        logger.info(f"Starting teacher experience collection for {num_chronics} chronics")

        start_time = time.time()
        
        for chronic_idx in range(num_chronics):
            self._process_chronic(
                chronic_idx=chronic_idx,
                env=env,
                agent=agent,
                perturbation_actions=perturbation_actions,
                attr_list=attr_list,
                attr_start_idx=attr_start_idx,
                save_path=save_path,
                start_time=start_time
            )

        total_time = (time.time() - start_time) / 60
        logger.info(f"Teacher experience collection completed in {total_time:.2f} minutes")

    def _configure_environment(self, env: grid2op.Environment.Environment) -> None:
        """Configure environment parameters for teacher experience collection."""
        params = env.parameters
        params.MAX_SUB_CHANGED = self.max_substations_changed
        params.MAX_LINE_STATUS_CHANGED = self.max_lines_changed
        env.change_parameters(params)
        logger.debug("Environment parameters configured")

    def _setup_chronics(self, env: grid2op.Environment.Environment, seed: int) -> None:
        """Setup chronics with deterministic shuffling."""
        np.random.seed(seed)
        env.chronics_handler.shuffle(
            shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)]
        )
        logger.debug(f"Chronics shuffled with seed {seed}")

    def _get_chronic_count(self, 
                          env: grid2op.Environment.Environment, 
                          chronic_limit: Optional[int]) -> int:
        """Determine number of chronics to process."""
        chronics_path = os.path.join(env.get_path_env(), env._chronics_folder_name())
        total_chronics = len(os.listdir(chronics_path))
        
        if chronic_limit is not None:
            return min(total_chronics, chronic_limit)
        return total_chronics

    def _process_chronic(self,
                        chronic_idx: int,
                        env: grid2op.Environment.Environment,
                        agent: CurriculumAgent,
                        perturbation_actions: List[Tuple[str, List[int]]],
                        attr_list: List[str],
                        attr_start_idx: Dict[str, int],
                        save_path: Path,
                        start_time: float) -> None:
        """Process a single chronic and collect experience."""
        # Initialize agent if needed
        if agent is None:
            agent = self._create_default_agent(env)

        elapsed_minutes = (time.time() - start_time) / 60
        logger.info(
            f"Processing chronic {chronic_idx} at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"(elapsed: {elapsed_minutes:.2f} min, PID: {os.getpid()})"
        )

        # Reset environment and start chronic
        env.reset()
        dst_step = 0
        scenario_name = env.chronics_handler.get_name()
        logger.debug(f"Starting scenario: {scenario_name}")
        
        env.fast_forward_chronics(dst_step)
        observation = env.get_obs()
        done = False

        while not done:
            # Search for best perturbation actions
            best_actions = self._search_perturbation_actions(
                observation=observation,
                agent=agent,
                perturbation_actions=perturbation_actions,
                attr_list=attr_list,
                attr_start_idx=attr_start_idx
            )

            if best_actions:
                # Use best perturbation action
                best_action = best_actions[0][1]
                perturbed_obs = self._apply_perturbation(
                    observation, best_action[0], best_action[1], attr_list, attr_start_idx
                )
                agent_action = agent.act(perturbed_obs, 0, False)
                observation, _, done, _ = env.step(agent_action)
                
                # Save experience
                self._save_experience(save_path, best_actions)
            else:
                # No suitable perturbation found, use normal action
                agent_action = agent.act(observation, 0, False)
                observation, _, done, _ = env.step(agent_action)

            dst_step += 1

    def _create_default_agent(self, env: grid2op.Environment.Environment) -> CurriculumAgent:
        """Create default curriculum agent if none provided."""
        agent = CurriculumAgent(env.action_space, env.observation_space, "teacher_agent")
        current_path = Path(os.getcwd())
        agent.load(current_path / "baseline_model", current_path / "action_definitions")
        return agent

    def _search_perturbation_actions(self,
                                   observation: grid2op.Observation.BaseObservation,
                                   agent: CurriculumAgent,
                                   perturbation_actions: List[Tuple[str, List[int]]],
                                   attr_list: List[str],
                                   attr_start_idx: Dict[str, int]) -> List[Tuple[float, Tuple[str, List[int]]]]:
        """
        Search and rank perturbation actions by their effect on agent performance.

        Args:
            observation: Current grid observation
            agent: Agent to evaluate
            perturbation_actions: Available perturbation actions
            attr_list: List of observation attributes
            attr_start_idx: Starting indices for each attribute

        Returns:
            List of (reward_improvement, action) tuples sorted by improvement
        """
        # Get baseline performance without perturbation
        try:
            _, baseline_reward, done, _ = observation.simulate(
                agent.act(observation, 0, False, simulated_act=True)
            )
            if done:
                return []
        except Exception as e:
            logger.warning(f"Baseline simulation failed: {e}")
            return []

        baseline_score = -baseline_reward
        performed_actions = []
        
        # Penalty for episode termination
        steps_remaining = observation.max_step - observation.current_step
        termination_penalty = 100000 * (steps_remaining / observation.max_step)

        for perturbation_type, perturbation_indices in perturbation_actions:
            try:
                # Apply perturbation
                perturbed_obs = self._apply_perturbation(
                    observation, perturbation_type, perturbation_indices, attr_list, attr_start_idx
                )
                
                # Skip if observation is not in critical state
                if perturbed_obs.rho.max() < self.rho_threshold:
                    continue

                # Evaluate agent performance on perturbed observation
                agent_action = agent.act(perturbed_obs, 0, False, simulated_act=True)
                _, reward, done, _ = observation.simulate(agent_action)
                
                # Calculate score with termination penalty
                score = -reward + termination_penalty * done
                reward_improvement = score - baseline_score
                
                performed_actions.append((reward_improvement, (perturbation_type, perturbation_indices)))
                
            except Exception as e:
                logger.debug(f"Perturbation evaluation failed for {perturbation_type}: {e}")
                continue
            finally:
                # Cleanup perturbed observation
                if 'perturbed_obs' in locals():
                    del perturbed_obs

        # Sort by reward improvement (best first) and return top k
        best_actions = sorted(performed_actions, key=lambda x: x[0], reverse=True)[:self.top_k_actions]
        
        return best_actions

    def _apply_perturbation(self,
                          observation: grid2op.Observation.BaseObservation,
                          perturbation_type: str,
                          perturbation_indices: List[int],
                          attr_list: List[str],
                          attr_start_idx: Dict[str, int]) -> grid2op.Observation.BaseObservation:
        """
        Apply perturbation to observation.

        Args:
            observation: Original observation
            perturbation_type: Type of perturbation ('missing' or 'large')
            perturbation_indices: Indices to perturb
            attr_list: List of observation attributes
            attr_start_idx: Starting indices for each attribute

        Returns:
            Perturbed observation copy
        """
        perturbed_obs = observation.copy()
        perturbed_obs.to_vect()

        if perturbation_type == "missing":
            perturbation_value = 0
        elif perturbation_type == "large":
            perturbation_value = 999999
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        # Apply perturbation to vectorized observation
        perturbed_obs._vectorized[perturbation_indices] = perturbation_value

        # Update specific attributes (particularly rho which is critical)
        for idx in perturbation_indices:
            if idx < len(attr_list):
                attr_name = attr_list[idx]
                
                if attr_name == "rho":  # Critical for line load monitoring
                    current_attr = getattr(perturbed_obs, attr_name)
                    
                    if isinstance(current_attr, np.ndarray):
                        attr_copy = current_attr.copy()
                        local_idx = idx - attr_start_idx[attr_name]
                        if 0 <= local_idx < len(attr_copy):
                            attr_copy[local_idx] = perturbation_value
                    else:
                        attr_copy = perturbation_value
                    
                    setattr(perturbed_obs, attr_name, attr_copy)

        return perturbed_obs

    def _save_experience(self,
                        save_path: Path,
                        best_actions: List[Tuple[float, Tuple[str, List[int]]]]) -> None:
        """
        Save teacher experience to CSV file.

        Args:
            save_path: Path to save file
            best_actions: List of best actions with their improvements
        """
        if not best_actions:
            return

        # Prepare data for saving
        data_rows = []
        for reward_improvement, (perturb_type, perturb_indices) in best_actions:
            row = [reward_improvement, perturb_type] + perturb_indices
            data_rows.append(row)

        # Determine maximum number of perturbation indices
        max_indices = max(len(row) - 2 for row in data_rows)
        
        # Create DataFrame with consistent columns
        column_names = ["reward_improvement", "perturb_type"] + [f"perturb_idx_{i}" for i in range(max_indices)]
        
        # Pad rows to have consistent length
        padded_rows = []
        for row in data_rows:
            while len(row) < len(column_names):
                row.append(np.nan)
            padded_rows.append(row)

        df = pd.DataFrame(padded_rows, columns=column_names)

        # Save to file
        self._save_dataframe_to_csv(save_path, df[:self.top_k_actions])

    def _save_dataframe_to_csv(self, save_path: Path, df: pd.DataFrame) -> None:
        """Save DataFrame to CSV with appropriate headers."""
        # Ensure save_path is a file path
        if save_path.is_dir():
            save_path = save_path / "teacher_experience.csv"

        try:
            if not save_path.exists():
                # Write with header for new file
                df.to_csv(save_path, index=False, header=True, mode="w")
                logger.debug(f"Created new experience file: {save_path}")
            else:
                # Append without header for existing file
                df.to_csv(save_path, index=False, header=False, mode="a")
                logger.debug(f"Appended to experience file: {save_path}")
                
        except Exception as e:
            logger.error(f"Failed to save experience to {save_path}: {e}")
            raise


# Convenience functions for backward compatibility and ease of use

def collect_teacher_experience(env: grid2op.Environment.Environment,
                             agent: CurriculumAgent,
                             perturbation_actions: List[Tuple[str, List[int]]],
                             save_path: Path,
                             observation_attributes: Tuple[List[str], Dict[str, int]],
                             seed: int = 42,
                             top_k: int = 125,
                             chronic_limit: Optional[int] = None,
                             rho_threshold: float = 0.95) -> None:
    """
    Convenience function to collect teacher experience.

    Args:
        env: Grid2Op environment
        agent: Curriculum agent
        perturbation_actions: List of perturbation actions to evaluate
        save_path: Path to save experience
        observation_attributes: Tuple of (attr_list, attr_start_idx)
        seed: Random seed
        top_k: Number of top actions to save
        chronic_limit: Maximum chronics to process
        rho_threshold: Line load threshold for perturbation application
    """
    collector = TeacherExperienceCollector(
        rho_threshold=rho_threshold,
        top_k_actions=top_k
    )
    
    collector.collect_experience(
        env=env,
        agent=agent,
        perturbation_actions=perturbation_actions,
        save_path=save_path,
        observation_attributes=observation_attributes,
        seed=seed,
        chronic_limit=chronic_limit
    )


def setup_observation_attributes(observation: grid2op.Observation.BaseObservation) -> Tuple[List[str], Dict[str, int]]:
    """
    Extract observation attribute mapping for perturbation operations.

    Args:
        observation: Grid2Op observation to analyze

    Returns:
        Tuple of (attribute_list, attribute_start_indices)
    """
    attr_list = []
    for attr in observation.attr_list_vect:
        val = getattr(observation, attr)
        attr_list += [attr] * (val.size if hasattr(val, 'size') else 1)
    
    attr_start_idx = pd.DataFrame(attr_list, columns=["attr"])\
                      .reset_index()\
                      .groupby("attr")["index"].min().to_dict()
    
    return attr_list, attr_start_idx


def generate_perturbation_actions(observation: grid2op.Observation.BaseObservation,
                                 max_perturbations: int = 1000,
                                 perturbation_types: List[str] = None) -> List[Tuple[str, List[int]]]:
    """
    Generate a set of perturbation actions for teacher experience collection.

    Args:
        observation: Sample observation to determine valid indices
        max_perturbations: Maximum number of perturbation actions to generate
        perturbation_types: Types of perturbations to include ['missing', 'large']

    Returns:
        List of (perturbation_type, indices) tuples
    """
    if perturbation_types is None:
        perturbation_types = ['missing', 'large']

    obs_vector = observation.to_vect()
    valid_indices = list(range(len(obs_vector)))
    
    perturbation_actions = []
    
    # Generate single-index perturbations
    for perturb_type in perturbation_types:
        for idx in valid_indices[:max_perturbations // len(perturbation_types)]:
            perturbation_actions.append((perturb_type, [idx]))
    
    # Generate multi-index perturbations (pairs)
    import itertools
    for perturb_type in perturbation_types:
        for idx_pair in itertools.combinations(valid_indices[:50], 2):
            if len(perturbation_actions) < max_perturbations:
                perturbation_actions.append((perturb_type, list(idx_pair)))
    
    return perturbation_actions[:max_perturbations]