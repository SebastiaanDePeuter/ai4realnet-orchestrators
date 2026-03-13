import pandas as pd
from modified_curriculum_classes.baseline import CurriculumAgent
import grid2op
from grid2op.Reward import L2RPNReward
import os
from pathlib import Path
import numpy as np
from perturbation_agents.rl_perturb_agent import RLPerturbationAgent
import sys
from perturbation_agents.action_space_reduction import collect_teacher_experience
from utility.UtilityHelper import UtilityHelper

def train_rl_perturb_agent():
    """
    Trains an RL perturbation agent using a curriculum agent's actions and teacher experience triples.
    Saves the trained model and training progress to disk.
    """
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    curr_path = Path(os.getcwd())
    
    bk_cls = UtilityHelper.get_backend()

    path_agents = "curr_agent_perturb_res"
    scoring_function = L2RPNReward
    max_iter = 30

    if not os.path.exists(path_agents):
        os.mkdir(path_agents)

    # Create grid2op environment
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name, reward_class=L2RPNReward, backend=bk_cls())
    attr_list = []
    obs = env.reset()

    # Collect attribute names for observation vector
    attr_list, attr_start_idx , _ = UtilityHelper.build_attr_list_and_index(obs=obs)


    # Load curriculum agent and its actions
    agent = CurriculumAgent(env.action_space, env.observation_space, "curriculum_agent_test")
    agent.load(curr_path / "model", curr_path / "actions")

    similarity_acts, similarity_acts_array, _,_ = UtilityHelper.init_similarity_score_dict(env, agent)

    subset_acts = UtilityHelper.select_representative_actions(similarity_matrix=similarity_acts_array)
    best_combis = UtilityHelper.get_best_combinations()
    # Reload curriculum agent for RL perturbation agent
    agent = CurriculumAgent(env.action_space, env.observation_space, "curriculum_agent_test")
    agent.load(curr_path / "model", curr_path / "actions")

    # Initialize RL perturbation agent
    rl_perturb_agent = RLPerturbationAgent(env.observation_space, agent.agent, 0.1, attr_list, attr_start_idx, subset=best_combis, subset_acts=subset_acts)
    print(f"started training with {rl_perturb_agent.n_actions} actions")
    try:
        result = rl_perturb_agent.train(env, num_episodes=n_episodes)
    except KeyboardInterrupt:
        result = rl_perturb_agent.train(env, num_episodes=n_episodes)
    except Exception:
        result = rl_perturb_agent.train(env, num_episodes=n_episodes)
    finally:
        # Save trained model and training progress
        x = 0
        while os.path.exists(os.path.join(curr_path, f"trained_rlpa_{x}.pth")):
            x += 1
        rl_perturb_agent.save_model(f"trained_rlpa_{x}.pth", f"trained_rlpa_target_net_{x}.pth")

        df = pd.DataFrame(result[1:], columns=result[0])
        df.to_csv(f"training_progress_rlpa_{x}.csv", index=False)
        print(f"saved to trained_rlpa_{x}.pth and training_progress_rlpa_{x}.csv")

def run_action_reduction_heuristic():
    """
    Runs a heuristic to reduce the action space by selecting vulnerable indices and best pairs/triples.
    Collects multi-experience data and saves it to disk.
    """
    bk_cls = UtilityHelper.get_backend()
    path_agents = "curr_agent_perturb_res"
    scoring_function = L2RPNReward
    max_iter = 30

    if not os.path.exists(path_agents):
        os.mkdir(path_agents)

    # Create grid2op environment
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name, reward_class=L2RPNReward, backend=bk_cls())
    attr_list = []
    attr_ = []
    obs = env.reset()

    attr_list, attr_start_idx, idx_min_max = UtilityHelper.build_attr_list_and_index(obs, attrs_to_index=["gen_p", "load_p", "rho"])

    # Vulnerable indices for perturbation
    vulnerable_vals = [ 35,  48,  63,  73,  84,  87, 115, 116, 125, 130, 143, 147, 169,
       179, 219, 240, 257, 279, 310, 341, 347, 360, 386, 395, 403, 417,
       449, 451, 455]
    # Load teacher experience pairs and select best pairs
    teach_pairs = pd.read_csv("teacher_experience_pairs.csv")
    teach_pairs = teach_pairs[teach_pairs["reward_improvement"] > 0.1]
    teach_pairs_count = teach_pairs.groupby(["perturb_idx_0", "perturb_idx_1"], as_index=False, dropna=False)["reward_improvement"].sum()
    teach_pairs_count = teach_pairs_count.sort_values("reward_improvement", ascending=False)

    best_pairs = teach_pairs_count[teach_pairs_count["reward_improvement"] > 12.5].dropna()
    best_pairs = best_pairs.sort_values(["perturb_idx_0", "perturb_idx_1"])
    best_pairs["perturb_idx_1"] = best_pairs["perturb_idx_1"].astype(int)
    best_pairs = [list(t) for t in best_pairs[["perturb_idx_0", "perturb_idx_1"]].values]

    # Generate triples from best pairs and vulnerable values
    triples = pd.DataFrame([{val1, val2, val3} for val1, val2 in best_pairs for val3 in vulnerable_vals if val3 != val1 and val3 != val2])
    triples = [list(t) for t in triples.drop_duplicates().values]
    possible_actions = [[val] for val in vulnerable_vals] + best_pairs + triples
    perturb_types_obs = [
                        "missing", 
                        "large", 
                    ]

    # Create all possible actions with perturbation types
    possible_actions = [(perturb_type, idx) for perturb_type in perturb_types_obs for idx in possible_actions]

    curr_path = Path(os.getcwd())

    # Load curriculum agent
    agent = CurriculumAgent(env.action_space, env.observation_space, "curriculum_agent_test")
    agent.load(curr_path / "model", curr_path / "actions")

    # Save multi-experience data to disk
    x = 0
    while os.path.exists(os.path.join(curr_path, f"teacher_experience_triples_{x}.csv")):
        x += 1
    collect_teacher_experience(env, agent, possible_actions, curr_path / f"teacher_experience_triples_{x}.csv", 2, (attr_list, attr_start_idx), chronic_limit=50)
    
if __name__ == "__main__":
    act_reduc_heur = False
    if act_reduc_heur:
        run_action_reduction_heuristic()
    train_rl_perturb_agent()