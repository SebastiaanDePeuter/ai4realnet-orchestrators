import numpy as np
import pandas as pd
from utility.UtilityHelper import UtilityHelper


class Grid2OpUtils:
    """
    Utility class for Grid2Op environment.
    Provides functionality to compute similarity between actions and to index key observation attributes.
    """

    def __init__(self, env, agent):
        """
        Initialize the utility class.

        Args:
            env: Grid2Op environment
            agent: Agent instance.
        """
        self.env = env

        # Initialize similarity and caches
        (self.similarity_acts, self.similarity_acts_array, self.n_subs_impact, self.n_grid_elem_impact) = UtilityHelper.init_similarity_score_dict(env, agent)

        # Dynamic cache
        self.similarity_cache = {}

        # Initialize observations attributes
        obs = env.reset()
        (self.attr_list, 
        self.attr_start_idx,
        self.idx_min_max) = UtilityHelper.build_attr_list_and_index(obs, ["gen_p", "load_p", "rho"])

    def init_attr_info(self, env):
        """
        Index key observation attributes.
        Builds a flat attribute list for all observation features and stores the index ranges.

        Args:
            env: Grid2Op environment reference 
        """
        obs = env.reset()

        self.attr_list, self.attr_start_idx, self.idx_min_max = UtilityHelper.build_attr_list_and_index(
            obs,
            attrs_to_index=["gen_p", "load_p", "rho"] 
        )

    def calculate_similarity_score(self, act1, act2):
        """
        Calculate a similarity score between two actions.
        Uses substation impacts and grid topology changes.

        Args:
            act1: Vectorized representation of the first action
            act2: Vectorized representation of the second action
        
        Returns:
            Similarity score between 0 (not similar) and 1 (fully similar).
        """
        # Convert actions to action objects
        # Convert actions to action objects
        act1_obj = self.env.action_space.from_vect(act1)
        act2_obj = self.env.action_space.from_vect(act2)

        # Preprocess actions
        pre1 = UtilityHelper.preprocess_actions(act1_obj, self.n_subs_impact, self.n_grid_elem_impact)
        pre2 = UtilityHelper.preprocess_actions(act2_obj, self.n_subs_impact, self.n_grid_elem_impact)

        return UtilityHelper.calculate_similarity_score_preprocessed(pre1, pre2, self.n_subs_impact, self.n_grid_elem_impact)

    def get_similarity_score(self, act1, act2):
        """
        Get stored similarity score or compute it.
        
        Args:
            act1: Vectorized representation of action 1.
            act2: Vectorized representation of action 2.
        
        Returns:
            Similarity score between 0 (not similar) and 1 (fully similar)
        """
        act1_idx = act1.astype(np.float32).tobytes()
        act2_idx = act2.astype(np.float32).tobytes()

        # Tries to use global cache
        if act1_idx in self.similarity_acts and act2_idx in self.similarity_acts[act1_idx]:
            return self.similarity_acts[act1_idx][act2_idx]
        
        # Compute the score
        similarity_score = self.calculate_similarity_score(act1, act2)

        if act1_idx not in self.similarity_acts:
            self.similarity_acts[act1_idx] = {}
        if act2_idx not in self.similarity_acts:
            self.similarity_acts[act2_idx] = {}     
        
        self.similarity_acts[act1_idx][act2_idx] = similarity_score
        self.similarity_acts[act2_idx][act1_idx] = similarity_score

        return similarity_score