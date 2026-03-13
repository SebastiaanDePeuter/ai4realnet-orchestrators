import numpy as np
import pandas as pd
import grid2op
from grid2op.Reward import L2RPNReward
from scipy.stats import bootstrap
import torch
import re
import os
class FilenameHelper:
    """Helper class for sanitizing filenames to work on all platforms"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Remove invalid Windows filename characters.
        
        Invalid chars: < > : " / \ | ? *
        Replaces them with underscores to ensure cross-platform compatibility.
        
        Args:
            filename: Original filename potentially containing invalid characters
            
        Returns:
            Sanitized filename safe for Windows/Linux/Mac
        """
        # Replace invalid Windows characters with underscores
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        # Remove leading/trailing spaces and dots (also Windows-reserved)
        sanitized = sanitized.strip(' .')
        
        return sanitized


# ============================================================================
class UtilityHelper:

    # --- Principal Methods ---
    @staticmethod
    def get_backend():
        """
        Select and return the Grid2Op backend.
        Tries to use a faster backend (LightSim2Grid) if available, otherwise uses and returns PandaPower.

        Returns:
            Backend class to initialize Grid2Op environment.
        """
        try:
            from lightsim2grid import LightSimBackend
            return LightSimBackend
        except ImportError:
            from grid2op.Backend import PandaPowerBackend
            return PandaPowerBackend
        
    @staticmethod
    def build_attr_list_and_index(obs, attrs_to_index=None):
        """
        Build flattened attribute list and mapping for first occurrence.
        Optionally computes min/max index ranges for selected attributes.

        Args:
            obs: Observation object
            attrs_to_index: Optional list of attribute names to index

        Returns:
            attr_list: flat list of attribute names
            attr_start_idx: dict mapping attribute to its first index
            idx_min_max: list of (min, max) index tuples for selected attributes
        """
        attr_list = []
        for attr in obs.attr_list_vect:
            val = getattr(obs, attr)
            attr_list += [attr] * (val.size if hasattr(val, "size") else 1)

        df = pd.DataFrame(attr_list, columns=["attr"]).reset_index()
        attr_start_idx = df.groupby("attr")["index"].min().to_dict()

        idx_min_max = []
        if attrs_to_index:
            idx_min_max = [
                (df[df["attr"] == a].index.min(), df[df["attr"] == a].index.max())
                for a in attrs_to_index
            ]
        return attr_list, attr_start_idx, idx_min_max
    
    @staticmethod
    def preprocess_actions(act, n_subs_cache=None, n_grid_elem_cache = None):

        """Extract useful info from an action object once, to avoid recomputation."""

        topo_vect = act._set_topo_vect
        subs_impact = act.get_topological_impact()[1]
        n_subs = subs_impact.sum()
        n_grid_elem = (topo_vect > 0).sum()

        # Utilizing Cache
        act_bytes = act.to_vect().astype(np.float32).tobytes()
        if n_subs_cache:
            n_subs_cache[act_bytes] = n_subs
        if n_grid_elem_cache:
            n_grid_elem_cache[act_bytes] = n_grid_elem

        return{
            "topo_vect": topo_vect,
            "subs_impact": subs_impact,
            "n_subs": n_subs,
            "n_grid_elem": n_grid_elem

        }

    @staticmethod
    def calculate_similarity_score(env, act1,act2):
        """
        Calculate the similarity score between two actions based on substations they affect.
        
        Args:
            env: Grid2Op environment. 
            act1: Vectorized representation of action 1.
            act2: Vectorized representation of action 2.
        
        Returns:
            Similarity score between 0 (not similar) and 1 (fully similar)
        """

        # Convert actions to action objects
        act1_obj = env.action_space.from_vect(act1)
        act2_obj = env.action_space.from_vect(act2)

        pre1 = UtilityHelper.preprocess_actions(act1_obj)
        pre2 = UtilityHelper.preprocess_actions(act2_obj)

    
        return UtilityHelper.calculate_similarity_score_preprocessed(pre1,pre2)
    
    @staticmethod
    def calculate_similarity_score_preprocessed(pre1,pre2, n_subs_cache = None, n_grid_elem_cache = None):

        # Actualize caches if given
        if n_subs_cache:
            n_subs_cache[pre1["topo_vect"].tobytes()] = pre1["n_subs"]
            n_subs_cache[pre2["topo_vect"].tobytes()] = pre2["n_subs"]
        if n_grid_elem_cache:
            n_grid_elem_cache[pre1["topo_vect"].tobytes()] = pre1["n_grid_elem"]
            n_grid_elem_cache[pre2["topo_vect"].tobytes()] = pre2["n_grid_elem"]

        
        # Overlap in substations
        n_same_sub = np.logical_and(pre1["subs_impact"], pre2["subs_impact"]).sum()
        
        # Exact same grid element chnages
        n_same_change = np.logical_and(pre1["topo_vect"] > 0, pre1["topo_vect"] == pre2["topo_vect"]).sum()
        
        # Different changes on same elements
        n_almost_same_change = np.logical_and(
            pre1["topo_vect"] > 0,
            np.logical_and(pre2["topo_vect"] >0, pre1["topo_vect"] != pre2["topo_vect"])
        ).sum()

        # Handle edge cases
        if (pre1["n_subs"] == 0) ^ (pre2["n_subs"] == 0):
            return 0.0
        elif (pre1["n_subs"] == 0) and (pre2["n_subs"] == 0):
            return 1.0
        else:
            similarity_subs = ((n_same_sub/pre1["n_subs"]) + (n_same_sub / pre2["n_subs"])) / 2
            similarity_changes = (
                ((n_same_change + (n_almost_same_change /2)) / pre1["n_grid_elem"]) +
                ((n_same_change + (n_almost_same_change /2)) / pre2["n_grid_elem"])
            ) / 2
        return (similarity_subs + similarity_changes) / 2

    @staticmethod
    def init_similarity_score_dict(env, agent):
        """
        Initialize similarity matrices and impact dictionaries for actions

        Args:
            - env: The environment
            - agent: Agent whose actions are used for similarity init.
        """
        acts = [env.action_space.from_vect(a) for a in agent.agent.actions]
        acts_asbytes = [a.astype(np.float32).tobytes() for a in agent.agent.actions]

        # Preprocess the actions
        preprocessed = [UtilityHelper.preprocess_actions(a) for a in acts]
        
        n_acts = len(acts)
        similarity_acts = dict()
        similarity_acts_array = np.zeros((n_acts, n_acts))
        n_subs_impact = {}
        n_grid_elem_impact = {}
        for i in range(n_acts):
            i_b = acts_asbytes[i]
            similarity_acts[i_b] = {}
            n_subs_impact[i_b] = preprocessed[i]["n_subs"]
            n_grid_elem_impact[i_b] = preprocessed[i]["n_grid_elem"]
            similarity_acts_array[i, i] = 1 # Action is fully similar to itself
            
            # Iterate over the upper triangle part
            for j in range(i + 1, n_acts):
                j_b = acts_asbytes[j]

                similarity_score = UtilityHelper.calculate_similarity_score_preprocessed(
                    preprocessed[i],
                    preprocessed[j]
                )
                if j_b not in similarity_acts:
                    similarity_acts[j_b] = {}
                similarity_acts[i_b][j_b] = similarity_score
                similarity_acts[j_b][i_b] = similarity_score
                similarity_acts_array[i, j] = similarity_score
                similarity_acts_array[j, i] = similarity_score

        return similarity_acts, similarity_acts_array, n_subs_impact, n_grid_elem_impact

    @staticmethod
    def select_representative_actions(similarity_matrix, similarity_threshold=0.9, max_subset=20):
        """
        Compute similarity among actions using topological impacts and group actions that affect the grid in similar ways.

        Args:
            similarity_matrix: Array with shape (n_actions, n_actions)
            similarity_threshold (optional): Minimum similarity to consider (default = 0.9)
            max_subset (optional): Maximum number of actions

        Returns:
            A list with subset of action indices of representative actions
        """

        similar_act_groups = (pd.DataFrame(np.where(similarity_matrix > similarity_threshold))
                            .T.groupby(0)
                            .agg({1: list})
                            .drop_duplicates(1)
                            .reset_index(drop=True))
        subset_actions = list({group[0] for group in similar_act_groups[1].values})[:max_subset]
        return subset_actions
    
    @staticmethod
    def get_best_combinations(raw_data_path = "attack_models/teacher_experience_triples_all.csv", reward_improv_threshold = 0.1, std_factor = 1.5 ):
        """Load teacher experience data and compute the best combinations.
        A combination is considered 'best' if it appears more often than the average among cases with reward improvement above a threshold
        
        Args:
            raw_data_path: Path to the teacher experience logs.
            reward_improv_threshold: Minimum reward value to filter cases (default = 0.1)
            std_factor: Factor multiplying with standard deviation to control how far from the average the combinations are (default = 1.5)


        """
        try:
            # Load raw teacher experience logs
            raw_teach_pairs = pd.read_csv(raw_data_path)

            # Keep only cases where reward improvement is greater than 0.1
            teach_pairs = raw_teach_pairs[raw_teach_pairs["reward_improvement"] > reward_improv_threshold]

            # Count occurrences of each perturbation triple
            teach_pairs_count = teach_pairs.groupby(
                ["perturb_idx_0", "perturb_idx_1", "perturb_idx_2"],
                as_index=False, dropna=False
            )["reward_improvement"].count()

            # Define a statistical threshold
            threshold = (teach_pairs_count["reward_improvement"].mean() + 
                        std_factor * teach_pairs_count["reward_improvement"].std())
            
            # Select combinations that exceed the threshold
            best_combinations = teach_pairs_count[
                teach_pairs_count["reward_improvement"] > threshold
            ].drop(columns=["reward_improvement"]).values

            # Clean NaN values
            best_combinations = [list(combination[~np.isnan(combination)].astype(int)) 
                          for combination in best_combinations]
            return best_combinations
        except FileNotFoundError:
            # Return an empty list if the dataset is missing
            print(f"Warning: {raw_data_path} not found. Some tests may fail.")
            return []


    # --- General Utility ---  
    @staticmethod
    def extract_list_from_file(file_path, is_float = False):
        result = []
        with open(file_path, newline='\n') as file:
            for line in file.readlines():
                line = line.strip()
                for i in line.split(","):
                    i = i.strip()
                    if not i:
                        continue
                    if is_float:
                        result.append(float(i))
                    else:
                        result.append(int(i))
        return result
    

    # --- Statistics ---

    @staticmethod
    def bootstrap_ci(data, func=np.mean, confidence_level=0.95, n_resamples=1000, method="percentile"):
    
        """
        Compute bootstrap confidence interval (ci),

        Args:
            data: Input data
            func (optional): Function to apply (default = np.mean)
            confidence_level (optional): Desired confidence level (default = 0.95)
            n_resamples (optional): Number of bootstrap resamples (default = 1000)
            method (optional): Bootstrap method to use from scipy options (default = 'percentile')

        Returns:
            tuple [float,float]: Lower and upper bounds of ci.
        """
        data = np.array(data)
        res = bootstrap(
            (data,), func,
            confidence_level=confidence_level,
            method=method,
            vectorized=False,
            n_resamples=n_resamples,
        )
        return res.confidence_interval.low, res.confidence_interval.high
    
    # --- Load Models ---

    @staticmethod
    def safe_load_models(filename:str, device: torch.device, classes:list):
        """
        Loads a PyTorch checkpoint safelly, allowing custom classes to be used.

        Args:
            filename: Path to the checkpoint file.
            device: Pytorch device to map the model to.
            classes: List of custom classes that may appear in the checkpoint
        """

        allowlist = list(classes) if classes is not None else []
        def _do_load(load_kwargs):
            return torch.load(filename, map_location=device, **load_kwargs)
        try:
            if hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals(allowlist):
                    return _do_load({})
            else:
                if hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals(allowlist)
                    return _do_load({})
                else:
                    return _do_load({"weights_only": False})
        except Exception as e:
            msg = str(e)
            m = re.search(r"Unsupported global: GLOBAL (.+?) was not an allowed global", msg)
            if m:
                missing = m.group(1)
                print(f"[safe_load_models] Missing allowlist entry detected: {missing}")
            try:
                print("[safe_load_models] Retrying torch.load with weights_only=False")
                return _do_load({"weights_only": False})
            except Exception as e2:
                print("[safe_load_models] Second load attempt failed:", e2)
                raise


    # --- PLOT RESULTS ---
    @staticmethod
    def latex_to_html_table(latex_file):


        with open(latex_file) as f:
            content = f.read()

        # Remove LaTeX commands 
        content = re.sub(r"\\begin{tabular}{.*?}", "", content)
        content = re.sub(r"\\end{tabular}", "", content)
        for cmd in ["\\toprule", "\\midrule", "\\bottomrule"]:
            content = content.replace(cmd, "")

        # Split into rows
        rows = [r.strip() for r in content.split("\\\\") if r.strip()]

        if rows and rows[0].startswith("{") and rows[0].endswith("}"):
            rows = rows[1:]

        # Split each row into columns
        data = [[c.strip() for c in row.split("&")] for row in rows]

        # Use first row as header
        df = pd.DataFrame(data[1:], columns=data[0])

        # Return HTML table
        return df.to_html(index=False, escape=False)


    @staticmethod
    def create_html_report(save_folder="test_results", output_file="report.html"):
        html_file = os.path.join(save_folder, output_file)
        html_content = "<html><head><title>Robustness Report</title></head><body>\n"
        html_content += "<h1>Robustness & Resilience Report</h1>\n"

        # Add tables
        for name, filename in [("Robustness", "robustness_table.tex"),
                            ("Resilience (Reward)", "reward_table.tex"),
                            ("Resilience (Observation Similarity)", "observation_table.tex")]:
            path = os.path.join(save_folder, filename)
            html_content += f"<h2>{name}</h2>\n"
            if os.path.exists(path):
                html_content += UtilityHelper.latex_to_html_table(path)
            else:
                html_content += f"<p>Missing table: {filename}</p>\n"

        # Add plots
        html_content += "<h2>Plots</h2>\n"

        # Overall plots (SVGs directly under save_folder)
        overall_svgs = sorted([f for f in os.listdir(save_folder) if f.endswith(".svg")])
        if overall_svgs:
            html_content += "<h3>Overall Plots</h3>\n"
            for file in overall_svgs:
                rel_path = os.path.join(save_folder, file)
                rel_path = os.path.relpath(rel_path, start=save_folder).replace("\\", "/").replace(" ", "%20")
                html_content += f'<img src="{file}" alt="{file}" style="max-width:800px;"><br>\n'

        # Episode plots
        episode_dirs = sorted([d for d in os.listdir(save_folder) if d.lower().startswith("episode")])
        for ep_dir in episode_dirs:
            ep_path = os.path.join(save_folder, ep_dir)
            if os.path.isdir(ep_path):
                html_content += f"<h3>{ep_dir}</h3>\n"
                for file in sorted(os.listdir(ep_path)):
                    if file.endswith(".svg"):
                        abs_path = os.path.join(ep_path, file)
                        rel_path = os.path.relpath(abs_path, start=save_folder).replace("\\", "/").replace(" ", "%20")
                        html_content += f'<img src="{rel_path}" alt="{file}" style="max-width:800px;"><br>\n'

        html_content += "</body></html>"

        with open(html_file, "w") as f:
            f.write(html_content)
    
        print(f"[INFO] HTML report saved to {html_file}")
    @staticmethod
    def extract_grid_graph_structure(env):
        """Build adjacency matrix and node feature shape from Grid2Op environment."""
        n_lines = len(env.name_line)
        n_subs = len(env.name_sub)
        adj = np.zeros((n_subs, n_subs))
        # unwrap if it's a wrapper
        raw_env = getattr(env, "env", None) or getattr(env, "_env", None) or env
        grid_obj = getattr(raw_env, "_grid", None) or getattr(raw_env, "grid", None)
        if grid_obj is None:
            raise AttributeError(f"No grid object found in environment {env}")
        for line in env._grid.lines_or_to_subid:
            i, j = line
            adj[i, j] = 1
            adj[j, i] = 1
        feat_dim = 8  # can be tuned or derived dynamically
        return {"adj": adj, "feat_dim": feat_dim}
