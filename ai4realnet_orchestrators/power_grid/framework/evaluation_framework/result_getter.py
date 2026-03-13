import pickle
from evaluation_framework.metrics import metrics
import os
import evaluation_framework.plots_and_tables as plots
import pandas as pd
import time
from utility.UtilityHelper import FilenameHelper

class result_getter:
    """Abstract class for running episodes in an environment and collecting results to evaluate metrics in a pickle file."""
    def __init__(self, env, defender, n_episodes, save_folder, attackers):
        """
        Initialize the result_getter.

        Args:
            env: The environment to run episodes in.
            defender: The defender agent.
            n_episodes: Number of episodes to run.
            save_folder: Folder to save results.
            attackers: List of attacker agents.
        """
        self.env = env
        self.defender = defender
        self.attackers = attackers
        self.n_episodes = n_episodes
        self.save_folder = save_folder

    def run_episodes(self, filename, attacker):
        """
        Run episodes in the environment with a given attacker and save results.

        Args:
            filename: Path to save the pickle file.
            attacker: The attacker agent (or None for unperturbed).

        Returns:
            data_dict: Dictionary containing episode data.
        """
        observations = []
        perturbations = []
        actions = []
        actions_unperturbed = []
        rewards = []

        self.env.attacker = attacker
        
        # ====================================================================
        # TIMING: Track overall episode timing
        # ====================================================================
        episode_start_time = time.time()
        attacker_name = attacker.model_name if attacker else "Unperturbed"
        print(f"\n[TIMER] Starting episodes for: {attacker_name}")
        
        episode_times = []  # Track each episode time
        
        for ep in range(self.n_episodes):
            obs = self.env.reset(ep)
            done = False

            # Initialize episode data lists
            observations.append([obs])
            perturbations.append([])
            actions.append([])
            actions_unperturbed.append([])
            rewards.append([])
            steps = 0
            
            # ================================================================
            # TIMING: Track individual episode timing
            # ================================================================
            ep_start_time = time.time()
            step_times = []

            while not done:
                # ============================================================
                # TIMING: Measure each step
                # ============================================================
                step_start = time.time()
                
                # Step through the environment
                obs, perturbation, act, act_unperturbed, reward, done = self.env.step()
                observations[ep].append(obs)
                perturbations[ep].append(perturbation)
                actions[ep].append(act)
                actions_unperturbed[ep].append(act_unperturbed)
                rewards[ep].append(reward)

                # Record step time
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                steps += 1
                
                # ============================================================
                # TIMING: Print progress every 10 steps
                # ============================================================
                if False and steps % 500 == 0:
                    # Get last 10 steps for average
                    last_10_steps = step_times[-10:] if len(step_times) >= 10 else step_times
                    avg_step_time = sum(last_10_steps) / len(last_10_steps)
                    
                    elapsed = time.time() - ep_start_time
                    
                    # Estimate remaining time based on average
                    print(f"  [EP{ep+1}] Step {steps}, Avg: {avg_step_time:.3f}s/step, "
                          f"Elapsed: {elapsed:.1f}s")

            # ================================================================
            # TIMING: Episode summary
            # ================================================================
            ep_total_time = time.time() - ep_start_time
            avg_step_time = ep_total_time / steps if steps > 0 else 0
            episode_times.append(ep_total_time)
            
            print(f"[TIMER] Episode {ep+1}/{self.n_episodes} completed: "
                  f"{steps} steps in {ep_total_time:.1f}s ({avg_step_time:.3f}s/step)")
        
        # ====================================================================
        # TIMING: Total summary for this attacker
        # ====================================================================
        total_time = time.time() - episode_start_time
        total_steps = sum(len(ep_rewards) for ep_rewards in rewards)
        avg_time_per_step = total_time / total_steps if total_steps > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: {attacker_name}")
        print(f"{'='*80}")
        print(f"Episodes:              {self.n_episodes}")
        print(f"Total steps:           {total_steps}")
        print(f"Total time:            {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average time/step:     {avg_time_per_step:.3f}s")
        if episode_times:
            print(f"Min episode time:      {min(episode_times):.1f}s")
            print(f"Max episode time:      {max(episode_times):.1f}s")
            print(f"Avg episode time:      {sum(episode_times)/len(episode_times):.1f}s")
        print(f"{'='*80}\n")

        # Save episode data to a dictionary
        data_dict = {
            "observations": observations,
            "perturbations": perturbations,
            "actions": actions,
            "actions_unperturbed": actions_unperturbed,
            "rewards": rewards
        }

        # Save data to pickle file
        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return data_dict

    def calculate_metrics(self):
        """
        Calculate and plot metrics for all attackers and the unperturbed case.
        Saves results and plots to the specified folder.
        """
        unperturbed_pickle = f"{self.save_folder}/unperturbed.pkl"
        # Load or generate unperturbed data
        if os.path.exists(unperturbed_pickle):
            with open(unperturbed_pickle, "rb") as handle:
                print(f"[SKIPPED] Pickle file {unperturbed_pickle} already exists, loading data...")
                unperturbed_data_dict = pickle.load(handle)
        else:
            print(f"[RUNNING] Running episodes for unperturbed case...")
            unperturbed_data_dict = self.run_episodes(unperturbed_pickle, attacker=None)
        unperturbed_metrics = metrics(
            unperturbed_data_dict,
            unperturbed_data_dict,
            self.env.do_nothing_action(),
            self.env.get_similarity_score,
            model_name="Unperturbed"
        )

        metrics_dicts = []

        # Process each attacker
        for attacker in self.attackers:
            save_pickle = f"{self.save_folder}/{attacker.pickle_file}"
            if os.path.exists(save_pickle):
                with open(save_pickle, "rb") as handle:
                    print(f"[SKIPPED] Pickle file {save_pickle} already exists, loading data...")
                    data_dict = pickle.load(handle)
            else:
                print(f"[RUNNING] Running episodes for {attacker.model_name}...")
                data_dict = self.run_episodes(save_pickle, attacker)

            # Calculate metrics for the current attacker
            metrics_dict = metrics(
                data_dict,
                unperturbed_data_dict,
                self.env.do_nothing_action(),
                self.env.get_similarity_score,
                model_name=attacker.model_name
            )
            metrics_dicts.append(metrics_dict)

            # Generate and save plots for each episode
            # ================================================================
            # FIX: Sanitize attacker.model_name to remove invalid characters
            # ================================================================
            sanitized_model_name = FilenameHelper.sanitize_filename(attacker.model_name)
            
            for episode in range(self.n_episodes):
                episode_folder = f"{self.save_folder}/Episode{episode}/"
                if not os.path.exists(episode_folder):
                    os.makedirs(episode_folder)
                
                # Use sanitized name in filenames
                plots.plot_reward_curve_comparison(
                    metrics_dict, episode, model_name=attacker.model_name,
                    filename=episode_folder + f"reward_curve_{sanitized_model_name}.svg"
                )
                plots.plot_cos_similarity_curve_comparison(
                    metrics_dict, episode, model_name=attacker.model_name,
                    filename=episode_folder + f"cos_sim_curve_{sanitized_model_name}.svg"
                )

        # Calculate and plot robustness and resilience metrics
        robustness_table = plots.get_metrics_robustness_not_compared_to_unperturbed(
            metrics_dicts, print_as_latex=False
        )
        reward_table, observation_table = plots.get_metrics_resilience(
            metrics_dicts, print_as_latex=False
        )

        plots.plot_metrics_robustness_compared_to_unperturbed(
            metrics_dicts, unperturbed_metrics, filename=f"{self.save_folder}/robustness.svg"
        )

        # Save tables as LaTeX files
        robustness_table.to_latex(f"{self.save_folder}/robustness_table.tex")
        reward_table.to_latex(f"{self.save_folder}/reward_table.tex")
        observation_table.to_latex(f"{self.save_folder}/observation_table.tex")