import numpy as np
import pandas as pd
from pandas import isna

from scipy.integrate import trapezoid
from scipy.signal import find_peaks


class metrics:
    """
        Class for computing robustness and resilience metrics for reinforcement learning agents under perturbations.
        This class evaluates the performance of an agent by comparing its behavior and rewards in perturbed and unperturbed environments.         
        Attributes:
            similarity_score_fn (Callable): Function to compute similarity scores between two actions passed as np.ndarray.
            model_name (str): Name of the model being evaluated.
            
            rewards_unperturbed (list): Rewards from the environment without a perturbation agent.
            rewards_perturbed (list): Rewards from the environment with a perturbation agent.
            cos_similarity_all (list): Cosine similarity between the observation in the environment with and without perturbation agent in each step.
            euclidean_dist_all (list): Euclidean distances between the observation in the environment with and without perturbation agent in each step.

            metrics_robustness (pd.DataFrame): DataFrame containing raw robustness metrics for each episode.
            metrics_resilience (pd.DataFrame): DataFrame containing raw resilience metrics.
            metrics_resilience_obs_sim (pd.DataFrame): DataFrame containing resilience metrics calculated using observation similarity measure.
            perturb_vulnerability (np.ndarray): Proportion of significant perturbations resulting in changed actions.
        Methods:
            compute_metrics(...): Computes robustness and resilience metrics for all episodes and gets the average.
            get_robustness_metrics_single_ep(...): Computes robustness metrics for a single episode.
            get_resilience_metrics_single_ep(...): Computes resilience metrics for a single episode.
            aggregate_metrics_resilience(...): Aggregates resilience metrics across episodes.
            get_perturb_vulnerability(...): Computes the vulnerability of each data point in the observation to perturbations.
            compute_perturb_prop_single_ep(...): Computes the proportion of significant perturbations that result in changed actions for a single episode.
    """

    def __init__(self, data_dict_perturbed, data_dict_unperturbed, do_nothing_action, similarity_score_fn, model_name=""):
        """
            Initializes the metrics instance and computes robustness and resilience metrics for the provided episodes.
            Args:
                data_dict_perturbed (dict): Dictionary containing data from the environment with the perturbation agent to evaluate, including observations, actions, perturbations, and rewards.
                data_dict_unperturbed (dict): Dictionary containing data from the environment without a perturbation agent, including observations and rewards.
                do_nothing_action (np.ndarray): The action representing a 'do nothing' or baseline action in the environment.
                similarity_score_fn (Callable): Function to compute similarity scores between two actions passed as np.ndarrays.
                model_name (str, optional): Name of the model being evaluated. Defaults to an empty string.
        """

        obs_unperturb = data_dict_unperturbed["observations"]
        obs_perturb = data_dict_perturbed["observations"]
        perturbations = data_dict_perturbed["perturbations"]
        actions_unperturbed = data_dict_perturbed["actions_unperturbed"]
        actions_perturbed = data_dict_perturbed["actions"]
        rewards_unperturbed = data_dict_unperturbed["rewards"]
        rewards_perturbed = data_dict_perturbed["rewards"]

        self.similarity_score_fn = similarity_score_fn
        self.model_name = model_name

        self.rewards_unperturbed = rewards_unperturbed
        self.rewards_perturbed = rewards_perturbed

        self.metrics_robustness, self.metrics_resilience, self.metrics_resilience_obs_sim = None, None, None

        self.compute_metrics(obs_unperturb, obs_perturb, perturbations, actions_unperturbed, actions_perturbed, 
                            rewards_unperturbed, rewards_perturbed, do_nothing_action)


    def compute_metrics(self, obs_unperturb, obs_perturb, perturbations, actions_unperturbed, actions_perturbed, rewards_unperturbed, 
            rewards_perturbed, do_nothing_action):
        # initialize lists to store metrics
        metrics_robustness, metrics_resilience, metrics_resilience_obs_sim = [[] for _ in range(3)]

        # initialize list of columns for metrics 
        cols_robustness = ["episode", "n_steps_with_act", "n_actions_changed", "similarity_score", "total_reward", "n_steps", "ave_reward_per_step"]
        cols_resilience = ["episode", "degradation_time", "restoration_time", "min_reward", "max_reward", "n_steps", "area", "n_degr_states"]
            
        self.cos_similarity_all = []
        self.euclidean_dist_all = []
        
        for ep in range(len(obs_perturb)):
            ep_rewards_unpert = rewards_unperturbed[ep] if ep < len(rewards_unperturbed) else []
            ep_rewards_pert = rewards_perturbed[ep] if ep < len(rewards_perturbed) else []
            
            if len(ep_rewards_unpert) == 0 or len(ep_rewards_pert) == 0:
                print(f"Warning: Empty rewards for episode {ep}, using default metrics")
                results_resilience = {'degradation_time': 0, 'restoration_time': 0, 'area': 0, 'area_per_1000_steps': 0}
            else:
                results_resilience = self.get_resilience_metrics_single_ep(
                    ep_rewards_unpert, ep_rewards_pert,
                    distance_peaks=500, distance_valleys=500
                )
            # compute robustness metrics and similarity in observation
            cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions = self.get_robustness_metrics_single_ep(obs_unperturb[ep], obs_perturb[ep], actions_unperturbed[ep], actions_perturbed[ep], do_nothing_action)  
            r = [x for x in rewards_perturbed[ep] if not isna(x)]
            metrics_robustness_ep = [ep, n_actions, actions_changed, similarity_score, sum(r), len(r), sum(r) / len(r)]
            metrics_robustness.append(metrics_robustness_ep)

            # compute resilience metrics
            results_resilience = self.get_resilience_metrics_single_ep(rewards_unperturbed[ep], rewards_perturbed[ep], distance_peaks=500, distance_valleys=500)
            results_resilience = [ep] + [np.mean(r) if len(r) > 0 else 0 for r in results_resilience] + [len(results_resilience[0])]
            metrics_resilience.append(results_resilience)

            # compute resilience metrics for cosine similarity
            results_resilience_similarity = self.get_resilience_metrics_single_ep(np.ones_like(cos_similarity), cos_similarity, distance_peaks=100, distance_valleys=100)
            results_resilience_similarity = [ep] + [np.mean(r) if len(r) > 0 else 0 for r in results_resilience_similarity] + [len(results_resilience_similarity[0])]
            metrics_resilience_obs_sim.append(results_resilience_similarity)

            self.cos_similarity_all.append(cos_similarity)
            self.euclidean_dist_all.append(euclidean_dist)

        # combine robustness metrics into one dataframe
        metrics_robustness = pd.DataFrame(metrics_robustness, columns=cols_robustness)
        metrics_robustness[metrics_robustness.columns[2:]] = metrics_robustness[metrics_robustness.columns[2:]].astype(float)
        self.metrics_robustness = metrics_robustness

        # combine resilience metrics and get the mean for each perturbation agent
        metrics_resilience = pd.DataFrame(metrics_resilience, columns=cols_resilience)
        metrics_resilience = pd.DataFrame(self.aggregate_metrics_resilience(metrics_resilience)).T
        self.metrics_resilience = metrics_resilience

        # combine resilience metrics and get the mean for each perturbation agent
        metrics_resilience_obs_sim = pd.DataFrame(metrics_resilience_obs_sim, columns=cols_resilience)
        metrics_resilience_obs_sim = pd.DataFrame(self.aggregate_metrics_resilience(metrics_resilience_obs_sim)).T
        self.metrics_resilience_obs_sim = metrics_resilience_obs_sim

        np.seterr(divide = 'ignore', invalid='ignore') 
        perturb_vulnerability = self.get_perturb_vulnerability(perturbations, actions_perturbed, actions_unperturbed)
        self.perturb_vulnerability = perturb_vulnerability
        np.seterr(divide = 'warn', invalid='warn')

        return metrics_robustness, metrics_resilience, metrics_resilience_obs_sim, perturb_vulnerability

    def get_robustness_metrics_single_ep(self, obs_unperturb, obs_perturb, actions_unperturbed, actions_perturbed, do_nothing_action):
        """
        Computes robustness metrics for a single episode.

        Args:
            obs_unperturb (np.ndarray): Observations from the unperturbed environment.
            obs_perturb (np.ndarray): Observations from the perturbed environment.
            actions_unperturbed (np.ndarray): Actions taken in the unperturbed environment.
            actions_perturbed (np.ndarray): Actions taken in the perturbed environment.
            do_nothing_action (np.ndarray): The baseline 'do nothing' action.

        Returns:
            tuple: (cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions)
                - cos_similarity (list): Cosine similarity between observations at each step.
                - euclidean_dist (list): Euclidean distance between observations at each step.
                - actions_changed (int): Number of steps where the action changed due to perturbation.
                - similarity_score (float): Sum of similarity scores for changed actions.
                - n_actions (int): Number of steps where a non-baseline action was performed.
        """
        cos_similarity = []
        euclidean_dist = []
        actions_changed = 0
        similarity_score = 0
        n_actions = 0
        for step, act in enumerate(actions_perturbed):
            # Check if a non-baseline action was performed
            act_performed = (act != do_nothing_action).any()
            if act_performed:
                n_actions += 1

            # Compute observation similarity metrics
            if step < len(obs_unperturb):
                obs_orig = obs_unperturb[step]
                obs_ = obs_perturb[step]
                cos_similarity.append(np.dot(obs_orig, obs_) / (np.linalg.norm(obs_orig) * np.linalg.norm(obs_)))
                euclidean_dist.append(np.linalg.norm(obs_orig - obs_))

            # Count changed actions and accumulate similarity score
            if (act != actions_unperturbed[step]).any():
                actions_changed += 1
                act2 = actions_unperturbed[step]
                if act_performed and (act2 != do_nothing_action).any():
                    similarity_score += self.similarity_score_fn(act, act2)

        return cos_similarity, euclidean_dist, actions_changed, similarity_score, n_actions

    def get_resilience_metrics_single_ep(self, data_unperturbed, data_perturbed, distance_peaks=None, distance_valleys=None):
        """
        Computes resilience metrics for a single episode by analyzing the difference between unperturbed and perturbed data.

        Args:
            data_unperturbed (list or np.ndarray): Data from the unperturbed environment (e.g., rewards).
            data_perturbed (list or np.ndarray): Data from the perturbed environment (e.g., rewards).
            distance_peaks (int, optional): Minimum distance between peaks for peak detection.
            distance_valleys (int, optional): Minimum distance between valleys for valley detection.

        Returns:
            tuple: (degradation_times, restoration_times, min_rewards, max_rewards, [n_steps_perturbed], [area_unpert_rpa])
        """
        # ========== SAFETY CHECKS ==========
        # Handle None inputs
        if data_unperturbed is None or data_perturbed is None:
            return ([0], [0], [0], [0], [0], [0])
        
        # Convert to lists if needed and filter NaN/zero
        data_unpert_clean = [x for x in data_unperturbed if not isna(x) and x != 0]
        data_pert_clean = [x for x in data_perturbed if not isna(x) and x != 0]
        
        # Handle empty arrays
        if len(data_unpert_clean) == 0 or len(data_pert_clean) == 0:
            return ([0], [0], [0], [0], [0], [0])
        
        n_steps_perturbed = len(data_pert_clean)
        min_len = min(len(data_unpert_clean), n_steps_perturbed)
        
        if min_len == 0:
            return ([0], [0], [0], [0], [0], [0])
        
        data_ = np.array([data_unpert_clean[:min_len], data_pert_clean[:min_len]])
        
        # Compute percentage difference between unperturbed and perturbed data
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            diff = (100 * (data_[0] - data_[1]) / data_[0])[:-1]
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If no degradation or empty diff, return zeros
        if len(diff) == 0 or diff.max() <= 0:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])
        
        # ========== END OF SAFETY CHECKS ==========

        # Find peaks (degradation) and valleys (restoration)
        peaks = list(find_peaks(diff, distance=distance_peaks)[0])
        
        positive_indices = np.where(diff > 0)[0]
        if len(positive_indices) == 0:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])
        
        valleys = [positive_indices[0]] + list(find_peaks(-diff, distance=distance_valleys)[0])

        if len(peaks) == 0 or len(valleys) == 1:
            return ([0], [0], [0], [0], [n_steps_perturbed], [0])

        prev_peak = 0
        peak = peaks.pop(0)
        prev_valley = 0
        valley = valleys.pop(0)

        degradation_times = []
        restoration_times = []
        min_rewards = []
        max_rewards = []
        
        while len(peaks) + len(valleys) > 0:
            if peak < valley:
                diff_reward = diff[peak]
                if prev_peak > prev_valley:
                    if diff[prev_peak] >= diff_reward:
                        if len(peaks) > 0:
                            peak = peaks.pop(0)
                        else:
                            peak = len(diff)
                        continue
                    else:
                        degradation_times[-1] += peak - prev_peak
                        min_rewards[-1] = diff_reward
                else:
                    degradation_times.append(peak - prev_valley)
                    min_rewards.append(diff_reward)

                prev_peak = peak
                if len(peaks) > 0:
                    peak = peaks.pop(0)
                else:
                    peak = len(diff)
            else:
                diff_reward = diff[valley]
                if prev_valley > prev_peak:
                    if diff[prev_valley] <= diff_reward or diff_reward < 0:
                        if len(valleys) > 0:
                            valley = valleys.pop(0)
                        else:
                            valley = len(diff)
                        continue
                    elif prev_peak > 0:
                        restoration_times[-1] += valley - prev_valley
                        max_rewards[-1] = diff_reward
                elif prev_peak > 0:
                    restoration_times.append(valley - prev_peak)
                    max_rewards.append(diff_reward)

                prev_valley = valley
                if len(valleys) > 0:
                    valley = valleys.pop(0)
                else:
                    valley = len(diff)

        # Handle unmatched degradation/restoration
        if len(degradation_times) > len(restoration_times):
            restoration_times.append(len(diff) - prev_peak)
            max_rewards.append(min(diff[prev_peak:]))
        
        # Handle empty lists
        if len(degradation_times) == 0:
            degradation_times = [0]
        if len(restoration_times) == 0:
            restoration_times = [0]
        if len(min_rewards) == 0:
            min_rewards = [0]
        if len(max_rewards) == 0:
            max_rewards = [0]

        # Compute area between unperturbed and perturbed curves
        area_unpert_rpa = trapezoid(data_[0]) - trapezoid(data_[1])

        return (degradation_times, restoration_times, min_rewards, max_rewards, [n_steps_perturbed], [area_unpert_rpa])
    def aggregate_metrics_resilience(self, metrics_resilience_raw):
        """
        Aggregates resilience metrics across episodes, weighted by the number of degraded states.

        Args:
            metrics_resilience_raw (pd.DataFrame): DataFrame containing raw resilience metrics for each episode.

        Returns:
            pd.DataFrame: Aggregated resilience metrics as a DataFrame.
        """
        metrics_resilience_temp = metrics_resilience_raw.drop(columns=["episode"])
        if metrics_resilience_temp["n_degr_states"].sum() == 0:
            return pd.Series([0] * metrics_resilience_temp.shape[1], index=metrics_resilience_temp.columns)
        # Weighted average for each metric, weighted by number of degraded states
        metrics_resilience = np.array([
            np.average(
                metrics_resilience_temp[metrics_resilience_temp["n_degr_states"] > 0][col].values,
                weights=metrics_resilience_temp[metrics_resilience_temp["n_degr_states"] > 0]["n_degr_states"]
            )
            for col in metrics_resilience_temp.columns
        ])
        # Compute mean steps, area per 1000 steps, and degraded states per 1000 steps
        metrics_resilience[-3] = metrics_resilience_temp["n_steps"].mean()
        metrics_resilience[-2] = metrics_resilience_temp["area"].mean() / metrics_resilience[-3] * 1000
        metrics_resilience[-1] = metrics_resilience_temp["n_degr_states"].mean() / metrics_resilience[-3] * 1000
        metrics_resilience = pd.DataFrame(
            metrics_resilience,
            index=list(metrics_resilience_temp.columns[:-2]) + ["area_per_1000_steps", "n_degr_states_per_1000_steps"]
        )
        return metrics_resilience

    def get_perturb_vulnerability(self, perturbations, actions, actions_unperturbed):
        """
        Computes the vulnerability of each observation feature to perturbations.

        Args:
            perturbations (list of np.ndarray): Perturbations applied in each episode.
            actions (list of np.ndarray): Actions taken in each episode (perturbed).
            actions_unperturbed (list of np.ndarray): Actions taken in each episode (unperturbed).

        Returns:
            np.ndarray: Proportion of significant perturbations resulting in changed actions, averaged over episodes.
        """
        # Number of steps per episode
        n_steps_perturb = [len(perturbations[i]) for i in range(len(perturbations))]
        # Compute vulnerability for each episode
        succesful_perturb_prop = [
            self.compute_perturb_prop_single_ep(
                perturbations[i],
                np.not_equal(actions[i], actions_unperturbed[i]).any(axis=1)
            )
            for i in range(len(perturbations))
        ]
        # Weighted average across episodes
        succesful_perturb_prop = np.average(succesful_perturb_prop, weights=n_steps_perturb, axis=0)
        return succesful_perturb_prop

    def compute_perturb_prop_single_ep(self, perturbation, acts_changed):
        """
        Computes the proportion of times a significant perturbation of each feature resulted in a changed action for a single episode.

        Args:
            perturbation (np.ndarray): Perturbation values for each step and feature.
            acts_changed (np.ndarray): Boolean array indicating if the action changed at each step.

        Returns:
            np.ndarray: Proportion of significant perturbations resulting in changed actions for each feature.
        """
        # Mask out zero perturbations
        masked_data = np.ma.masked_equal(perturbation, 0)
        mean_perturb = masked_data.mean(axis=0).data
        std_perturb = masked_data.std(axis=0).data

        # Define significant perturbation thresholds
        lo = mean_perturb - std_perturb
        hi = mean_perturb + std_perturb
        # Identify significant perturbations
        perturbed_signif = (perturbation < lo) + (perturbation > hi)
        # Compute proportion for each feature
        perturb_prop = perturbed_signif[acts_changed].sum(axis=0) / perturbed_signif.sum(axis=0)
        perturb_prop[np.isnan(perturb_prop)] = 0

        return perturb_prop
