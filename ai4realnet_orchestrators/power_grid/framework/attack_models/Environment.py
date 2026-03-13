import numpy as np
import pandas as pd
from utility.UtilityHelper import UtilityHelper

class Environment:
    """
    Wrapper environment that integrates a defender agent and (optionally) an attacker into a Grid2Op environment.

    Args:
        env: The Grid2Op environment
        defender: Agent controlling the environment under normal conditions
        attacker (optional): Attacker agent capable of perturbing observations
    """
    def __init__(self, env, defender, attacker=None):
        self.env = env
        self.defender = defender
        self.attacker = attacker

        # Initialize similarity and caches
        (self.similarity_acts, self.similarity_acts_array, self.n_subs_impact, self.n_grid_elem_impact) = UtilityHelper.init_similarity_score_dict(env, defender)

        # Dynamic cache
        self.similarity_cache = {}

        # Initialize observations attributes
        obs = env.reset()
        (self.attr_list, 
        self.attr_start_idx,
        self.idx_min_max) = UtilityHelper.build_attr_list_and_index(obs, ["gen_p", "load_p", "rho"])

    
    def reset(self, ep):
        """
        Reset the environment for a specific time series episode.

        Args:
            ep: Episode index from the environment's time series dataset.
        
        Returns:
            Flattened observations vector
        """
        self.obs = self.env.reset(options={"time serie id": ep})
        return self.obs.to_vect() 
    
    def step(self):
        """
        Execute one environment step under attacker-defender dynamics.
        If an attacker is present:
            - Perturb the current observation
            - Compute perturbation delta
            - Compute defender's action on perturbed and unperturbed observations.
        Else:
            - Use unperturbed observation for the defender's action

        Returns:
            A tuple with:
                - Flattened new observations vector
                - Perturbation vector applied to observations
                - Action taken on perturbed observation
                - Action taken on original observation
                - Reward from the environment
                - Boolean indicating whether the episode is finished.
        """
        if self.attacker == None:
            perturbed_obs = self.obs
            perturbation = np.zeros_like(perturbed_obs.to_vect())
        else:
            perturbed_obs = self.attacker.perturb(self.obs)
            perturbation = perturbed_obs.to_vect() - self.obs.to_vect()

            # Defender action on clean observation for comparison
            act_unperturbed = self.defender.act(self.obs, 0, False, simulated_act=True)

        # Defender action on observation
        act = self.defender.act(perturbed_obs, 0, False)
        
        if self.attacker == None:
            act_unperturbed = act

        # Step the environment
        self.obs, reward, done, _ = self.env.step(act)
        
        return self.obs.to_vect(), perturbation, act.to_vect(), act_unperturbed.to_vect(), reward, done

    def do_nothing_action(self):
        """
        Generate a vector for the environment when nothing is done.
        """
        return self.env.action_space({}).to_vect()
    

    def init_similarity_score_dict(self, env, agent):
        """
        Initialize similarity matrices and impact dictionaries for actions

        Args:
            - env: The environment
            - agent: Agent whose actions are used for similarity init.
        """
        acts = [env.action_space.from_vect(a) for a in agent.agent.actions]
        acts_asbytes = [a.astype(np.float32).tobytes() for a in agent.agent.actions]

        n_acts = len(acts)
        self.similarity_acts = dict()
        self.similarity_acts_array = np.zeros((n_acts, n_acts))

        # Dictionaries to cache action impacts
        self.n_subs_impact = {} # Action impact on subsystem
        self.n_grid_elem_impact = {} # Action impact on grid element
        for i in range(n_acts):
            i_b = acts_asbytes[i]

            self.similarity_acts[i_b] = dict()
            self.n_subs_impact[i_b] = acts[i].get_topological_impact()[1].sum()
            self.n_grid_elem_impact[i_b] = (acts[i]._set_topo_vect > 0).sum()
            self.similarity_acts_array[i, i] = 1 # Action is fully similar to itself
            
            # Iterate over the upper triangle part
            for j in range(i + 1, n_acts):
                j_b = acts_asbytes[j]

                similarity_score = self.calculate_similarity_score(acts[i].to_vect(), acts[j].to_vect())
                self.similarity_acts[i_b][j_b] = similarity_score
                self.similarity_acts[j_b][i_b] = similarity_score

                self.similarity_acts_array[i, j] = similarity_score
                self.similarity_acts_array[j, i] = similarity_score
   
    def calculate_similarity_score(self, act1, act2):
        """
        Calculate the similarity score between two actions based on substations they affect.
        
        Args:
            act1: Vectorized representation of action 1.
            act2: Vectorized representation of action 2.
        
        Returns:
            Similarity score between 0 (not similar) and 1 (fully similar)
        """
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