import pandas as pd
import numpy as np
import copy
from utility.UtilityHelper import UtilityHelper

class BaseAttackerClass:
    def _attribute_setup(self, obs):
        """
        Build attribute mapping from observation.
        Creates a flat list of attributes from the observation and generates and index mapping for the first occurrence of each attribute.

        Args:
            obs: Observation object

        Returns:
            A tuple containing:
                - Flattened list of attributes repeated according to their size
                - Dictionary mapping each attribute name to the index of the firs occurrence.
        """
        # Call the static method from UtilityHelper
        attr_list, attr_start_idx, _ = UtilityHelper.build_attr_list_and_index(obs)
        
        return attr_list, attr_start_idx
    

    def __getattr__(self, name):
        return getattr(self.env, name)