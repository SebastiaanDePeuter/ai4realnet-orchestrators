"""
Base Perturbation Agent for Grid2Op Robustness Testing

This module provides the abstract base class for perturbation agents used in
robustness and resilience testing of Grid2Op agents. Perturbation agents
systematically modify observations to test agent behavior under various
adversarial or noisy conditions.

The base class defines the interface that all perturbation agents must implement
and provides common functionality for observation manipulation and state management.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import logging

import grid2op
from grid2op.Space import RandomObject

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasePerturbationAgent(RandomObject, ABC):
    """
    Abstract base class for all perturbation agents in robustness testing.
    
    This class defines the interface for agents that systematically modify
    Grid2Op observations to test the robustness and resilience of trained
    agents. Perturbation agents can implement various strategies including:
    
    - Adversarial perturbations designed to degrade performance
    - Noise injection to simulate sensor errors or communication issues
    - Missing data scenarios to test handling of incomplete information
    - Large value injections to simulate sensor malfunctions
    
    All perturbation agents inherit random state management from RandomObject
    to ensure reproducible testing scenarios.
    
    Example:
        >>> class MyPerturbAgent(BasePerturbationAgent):
        ...     def perturb(self, obs):
        ...         perturbed_obs = obs.copy()
        ...         # Apply specific perturbation logic
        ...         return perturbed_obs
        >>> 
        >>> agent = MyPerturbAgent(env.observation_space)
        >>> perturbed_obs = agent.perturb(original_obs)
    """

    def __init__(self, 
                 obs_space: grid2op.Observation.ObservationSpace,
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the base perturbation agent.

        Args:
            obs_space: Grid2Op observation space defining the structure
                      of observations that will be perturbed
            name: Optional name identifier for this perturbation agent
            **kwargs: Additional parameters for specific perturbation strategies
        """
        # Initialize random state management
        RandomObject.__init__(self)
        
        # Store observation space for validation and perturbation logic
        self.obs_space = obs_space
        
        # Store agent identification
        self.name = name or self.__class__.__name__
        
        # Initialize perturbation statistics
        self.perturbation_count = 0
        self.reset_count = 0
        
        # Store additional configuration
        self.config = kwargs
        
        logger.debug(f"Initialized {self.name} with observation space: {obs_space}")

    @abstractmethod
    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """
        Apply perturbation to the given observation.
        
        This is the core method that must be implemented by all subclasses.
        It should apply the specific perturbation strategy while preserving
        the basic structure and validity of the observation.

        Args:
            obs: Original Grid2Op observation to be perturbed

        Returns:
            Perturbed copy of the observation
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If observation is incompatible with this agent
            
        Note:
            Implementations should:
            1. Create a copy of the input observation
            2. Apply perturbations to the copy
            3. Ensure the result is still a valid observation
            4. Update internal statistics if needed
        """
        # Default implementation creates a copy - subclasses should override
        obs_perturbed = obs.copy()
        
        # Update statistics
        self.perturbation_count += 1
        
        # Log perturbation activity
        logger.debug(f"{self.name}: Applied perturbation #{self.perturbation_count}")
        
        return obs_perturbed

    def reset(self) -> None:
        """
        Reset the perturbation agent state.
        
        This method is called at the beginning of new episodes or test scenarios
        to ensure clean state initialization. Subclasses can override this to
        implement specific reset behavior for their perturbation strategies.
        """
        self.reset_count += 1
        logger.debug(f"{self.name}: Reset #{self.reset_count}")

    def validate_observation(self, obs: grid2op.Observation.BaseObservation) -> bool:
        """
        Validate that an observation is compatible with this perturbation agent.

        Args:
            obs: Observation to validate

        Returns:
            True if observation is compatible, False otherwise
        """
        try:
            # Check basic compatibility with observation space
            if obs.action_space != self.obs_space.action_space:
                logger.warning(f"{self.name}: Action space mismatch")
                return False
                
            # Check that observation can be vectorized (required for most perturbations)
            obs_vector = obs.to_vect()
            if obs_vector is None or len(obs_vector) == 0:
                logger.warning(f"{self.name}: Cannot vectorize observation")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Observation validation failed: {e}")
            return False

    def get_perturbation_info(self) -> Dict[str, Any]:
        """
        Get information about the current state of the perturbation agent.

        Returns:
            Dictionary containing agent state and statistics
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "perturbation_count": self.perturbation_count,
            "reset_count": self.reset_count,
            "config": self.config,
            "seed": self.space_prng.get_state()[1][0] if hasattr(self.space_prng, 'get_state') else None
        }

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducible perturbations.

        Args:
            seed: Random seed value
        """
        self.seed(seed)
        logger.debug(f"{self.name}: Set seed to {seed}")

    def copy(self) -> 'BasePerturbationAgent':
        """
        Create a copy of this perturbation agent.
        
        Note:
            Subclasses should override this method to ensure proper copying
            of their specific state and configuration.

        Returns:
            Copy of this perturbation agent
        """
        # Basic copy - subclasses should override for complete copying
        copied_agent = self.__class__(
            obs_space=self.obs_space,
            name=f"{self.name}_copy",
            **self.config
        )
        
        # Copy random state if possible
        if hasattr(self.space_prng, 'get_state'):
            copied_agent.space_prng.set_state(self.space_prng.get_state())
            
        return copied_agent

    def __str__(self) -> str:
        """String representation of the perturbation agent."""
        return f"{self.name}(perturbations={self.perturbation_count}, resets={self.reset_count})"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"perturbations={self.perturbation_count}, "
                f"resets={self.reset_count})")


class NullPerturbationAgent(BasePerturbationAgent):
    """
    A no-op perturbation agent that returns observations unchanged.
    
    This agent is useful for baseline testing and as a reference implementation.
    It demonstrates the minimal implementation required for a perturbation agent.
    """

    def perturb(self, obs: grid2op.Observation.BaseObservation) -> grid2op.Observation.BaseObservation:
        """
        Return observation unchanged (no perturbation applied).

        Args:
            obs: Original observation

        Returns:
            Unmodified copy of the observation
        """
        # Call parent method to handle statistics and logging
        obs_perturbed = super().perturb(obs)
        
        # No additional perturbation logic needed for null agent
        return obs_perturbed


# Utility functions for perturbation agent management

def create_perturbation_suite(obs_space: grid2op.Observation.ObservationSpace,
                            agent_configs: list) -> list:
    """
    Create a suite of perturbation agents for comprehensive testing.

    Args:
        obs_space: Grid2Op observation space
        agent_configs: List of (agent_class, config_dict) tuples

    Returns:
        List of initialized perturbation agents
    """
    agents = []
    
    for agent_class, config in agent_configs:
        try:
            agent = agent_class(obs_space, **config)
            agents.append(agent)
            logger.info(f"Created perturbation agent: {agent.name}")
        except Exception as e:
            logger.error(f"Failed to create {agent_class.__name__}: {e}")
    
    return agents


def validate_perturbation_agent(agent: BasePerturbationAgent,
                              test_obs: grid2op.Observation.BaseObservation) -> bool:
    """
    Validate that a perturbation agent works correctly with test observations.

    Args:
        agent: Perturbation agent to test
        test_obs: Sample observation for testing

    Returns:
        True if agent passes validation, False otherwise
    """
    try:
        # Test basic functionality
        if not agent.validate_observation(test_obs):
            logger.error(f"Agent {agent.name} failed observation validation")
            return False
        
        # Test perturbation
        perturbed_obs = agent.perturb(test_obs)
        if perturbed_obs is None:
            logger.error(f"Agent {agent.name} returned None from perturb()")
            return False
        
        # Test reset
        agent.reset()
        
        logger.info(f"Agent {agent.name} passed validation")
        return True
        
    except Exception as e:
        logger.error(f"Agent {agent.name} failed validation: {e}")
        return False