"""
RL Perturbation Agent - Deep Q-Learning for Adversarial Perturbation Generation
Implements a Deep Q-Network that learns to generate optimal perturbations
to challenge Grid2Op agents. Uses reinforcement learning to discover
the most effective adversarial strategies.
"""
import os
import time
import gc
from typing import List, Tuple, Optional, Union, Dict, Any
from collections import namedtuple, deque
from itertools import count
import numpy as np
import datetime as dt
import psutil
from utility.UtilityHelper import UtilityHelper
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import tensorflow as tf
import grid2op
from perturbation_agents.base_perturb_agent import BasePerturbationAgent
from modified_curriculum_classes.my_agent import MyAgent
import copy

# Define transition tuple for replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """Deep Q-Network for perturbation action selection."""
    
    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 128):
        """
        Initialize DQN architecture.
        
        Args:
            n_observations: Size of observation space
            n_actions: Number of possible perturbation actions
            hidden_size: Size of hidden layers
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """Initialize replay memory with given capacity."""
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args) -> None:
        """Save a transition to memory."""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int, random_state: np.random.RandomState) -> List[Transition]:
        """Sample random batch of transitions."""
        indices = random_state.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self) -> int:
        """Return current memory size."""
        return len(self.memory)

class RLPerturbationAgent(BasePerturbationAgent):
    """
    Reinforcement Learning agent that learns optimal perturbation strategies.
    
    Uses Deep Q-Learning to discover effective adversarial perturbations
    against target Grid2Op agents. The agent learns which perturbations
    cause the most disruption to target agent performance.
    """
    
    def __init__(self,
                 obs_space: grid2op.Observation.ObservationSpace,
                 agent: MyAgent,
                 max_perturb: float,
                 attr_list: List[str],
                 attr_start_idx: Dict[str, int],
                 subset: List[int] = None,
                 subset_acts: List[int] = None,
                 save_dir: str = ""):
        """
        Initialize RL perturbation agent.
        Args:
            obs_space: Grid2Op observation space
            agent: Target agent to learn perturbations against
            max_perturb: Maximum perturbation magnitude
            attr_list: List of observation attributes
            attr_start_idx: Starting indices for each attribute
            subset: Subset of observation indices to consider
            subset_acts: Subset of agent actions to target
            save_dir: Directory to save perturbation history
        """
        super().__init__(obs_space)
        
        # DQN hyperparameters (EXACT MATCH TO OLD)
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 150000
        self.tau = 0.005
        self.lr = 1e-4
        
        # Target agent and perturbation parameters
        self.agent = agent
        self.max_perturb = max_perturb
        self.attr_list = attr_list
        self.attr_start_idx = attr_start_idx
        
        # Device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        # Build perturbation action space (EXACT MATCH TO OLD)
        perturb_types_obs = ["missing", "large"]
        perturb_types_act = ["adv_exmpl"]
        
        if subset is None or len(subset) == 0:
            obs_vals = range(obs_space.n)
        else:
            obs_vals = subset
            
        if subset_acts is None or len(subset_acts) == 0:
            act_vals = list(range(len(agent.actions)))
        else:
            act_vals = list(subset_acts)
        act_vals.append(-1)
        
        self.possible_actions = [("do_nothing", 0)] + \
                                [(perturb_type, idx) for perturb_type in perturb_types_obs for idx in obs_vals] + \
                                [(perturb_type, idx) for perturb_type in perturb_types_act for idx in act_vals]
        
        self.n_actions = len(self.possible_actions)
        
        # Initialize DQN networks
        n_observations = obs_space.n
        self.policy_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer and replay memory
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        # Training state
        self.steps_done = 0
        
        # Gradient computation helpers (EXACT MATCH TO OLD)
        grad_helper_1 = np.zeros((obs_space.n, obs_space.n))
        np.fill_diagonal(grad_helper_1, 0.01)
        grad_helper_2 = np.zeros((obs_space.n, obs_space.n))
        np.fill_diagonal(grad_helper_2, -0.01)
        self.grad_helper = np.concatenate([grad_helper_1, grad_helper_2])
        
        self.save_dir = save_dir
        
        # History tracking (EXACT NAMES FROM OLD)
        self.perturb_hist = []
        self.perturb_hist_tuples = []
    
    # In RLPerturbAttacker.py, modify the perturb method:
    def perturb(self, obs):
        """Apply RL based perturbation to an observation."""
        obs_t = obs.copy()  # faster than deepcopy

        # Debug: Check what the defender would do without perturbation
        original_action = self.agent.act(obs_t, reward=None, done=False)

        # Select RL perturbation action
        obs_tensor = self.grid2op_obs_to_tensor(obs_t)
        action_idx = self.select_action(obs_tensor)
        action = self.possible_actions[action_idx]

        # Apply perturbation
        obs_perturbed = self.perform_perturb(obs_t, action)

        # Debug: Check what the defender does with perturbation
        perturbed_action = self.agent.act(obs_perturbed, reward=None, done=False)

        #if original_action != perturbed_action:
        #    print(f"[SUCCESS] Action changed from {original_action} to {perturbed_action}")

        return obs_perturbed
    
    def predict_acts(self, x:np.ndarray):
        """Fixed to handle SavedModel signature correctly."""
        # Convert to float32 as the model expects
        x = x.astype(np.float32)
        
        # The model expects 3 arguments: (input, training_flag, mask)
        # Based on the error, it needs these exact arguments
        pred = self.agent.model(x, False, None)
        
        pred = tf.nn.softmax(pred[0])
        pred = pred.numpy().reshape(-1)
        return pred
    
    def change_val_in_vect(self, vect: np.ndarray, idx: int, change: float):
        """Change value in vector - EXACT MATCH TO OLD."""
        vect_copy = np.array(vect).reshape((1, -1))
        vect_copy[0, idx] += change
        return vect_copy
    
    def compute_grad(self, vect:np.ndarray, target_idx:int):
        """Compute gradients with proper dtype and signature."""
        vects = (vect[0] + self.grad_helper).astype(np.float32)  # Ensure float32
        
        # Get predictions for all perturbed vectors
        predictions = []
        for vec in vects:
            vec_reshaped = vec.reshape(1, -1).astype(np.float32)
            # Call with correct signature: (input, training=False, mask=None)
            pred = self.agent.model(vec_reshaped, False, None)
            pred = tf.nn.softmax(pred[0]).numpy().reshape(-1)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute finite difference gradients
        gradients = (predictions[:vect.shape[1]] - predictions[vect.shape[1]:])[:, target_idx] / 0.02
        
        return gradients
    
    def create_adv_exmpl(self, obs: grid2op.Observation.BaseObservation, target_idx: int):
        """Create adversarial example - EXACT MATCH TO OLD."""
        obs_perturbed = obs.copy()
        min_opt_act = target_idx == -1
        
        if min_opt_act:
            _, target_idx = self.agent.act_with_id(obs, simulated_act=True)
        
        grads = self.compute_grad(obs_perturbed.to_vect().reshape((1, -1)), target_idx)
        obs_perturbed.to_vect()
        
        if min_opt_act:
            obs_perturbed._vectorized *= (1 - self.max_perturb * np.sign(grads))
        else:
            obs_perturbed._vectorized *= (1 + self.max_perturb * np.sign(grads))
        
        return obs_perturbed
    
    def perform_perturb(self, obs: grid2op.Observation.BaseObservation, perturb):
        """Perform perturbation - EXACT MATCH TO OLD."""
        obs_perturbed = obs.copy()
        perturb_type, perturb_idx = perturb
        
        obs_perturbed.to_vect()
        
        if perturb_type == "missing":
            obs_perturbed._vectorized[perturb_idx] = 0
            for i in [perturb_idx]:  # Handle as single index
                attr = self.attr_list[i]
                if attr != "rho":
                    continue
                r = getattr(obs_perturbed, attr)
                if type(r) == np.ndarray:
                    r_copy = r.copy()
                    idx = i - self.attr_start_idx[attr]
                    r_copy[idx] = 0
                else:
                    r_copy = 0
                setattr(obs_perturbed, attr, r_copy)
                
        elif perturb_type == "large":
            obs_perturbed._vectorized[perturb_idx] = 999999
            for i in [perturb_idx]:  # Handle as single index
                attr = self.attr_list[i]
                if attr != "rho":
                    continue
                r = getattr(obs_perturbed, attr)
                if type(r) == np.ndarray:
                    r_copy = r.copy()
                    idx = i - self.attr_start_idx[attr]
                    r_copy[idx] = 999999
                else:
                    r_copy = 999999
                setattr(obs_perturbed, attr, r_copy)
                
        elif perturb_type == "adv_exmpl":
            obs_perturbed = self.create_adv_exmpl(obs_perturbed, perturb_idx)
            obs_perturbed.rho = obs_perturbed.to_vect()[self.attr_start_idx["rho"]:self.attr_start_idx["rho"] + len(obs_perturbed.rho)]
        
        return obs_perturbed
    
    def select_action(self, obs: torch.Tensor):
        """Select action using epsilon-greedy - EXACT MATCH TO OLD."""
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        x = self.space_prng.random()
        
        if x < eps_threshold:
            return self.space_prng.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                return self.policy_net(obs).max(1).indices[0]
    
    def optimize_model(self):
        """Optimize model - EXACT MATCH TO OLD."""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size, self.space_prng)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        del transitions, batch, non_final_mask, non_final_next_states, state_batch, reward_batch, action_batch, state_action_values, next_state_values, expected_state_action_values, loss
    
    def grid2op_obs_to_tensor(self, obs: grid2op.Observation.BaseObservation, dtype=torch.float32) -> torch.Tensor:
        """Convert observation to tensor - EXACT MATCH TO OLD."""
        return torch.tensor(obs.to_vect(), dtype=dtype, device=self.device).unsqueeze(0)
    
    def train(self, env: grid2op.Environment.BaseEnv, num_episodes: int = 50):
        """Train agent - MATCHING OLD LOGIC."""
        self.steps_done = 0
        result = [("episode", "n_steps", "reward_orig", "reward_adv", "time")]
        process = psutil.Process(os.getpid())
        memory_usage = [0] * 7
        
        for i_episode in range(num_episodes):
            try:
                start = time.time()
                if i_episode % 10 == 0 or i_episode == 1:
                    print(f"start episode {i_episode} at {dt.datetime.now()} after {self.steps_done} steps with memory usage {process.memory_percent():.3f}%")
                    print([f"{m:.3f}" for m in memory_usage])
                
                state = env.reset()
                state_tensor = self.grid2op_obs_to_tensor(state)
                total_reward_orig = 0
                total_reward_adv = 0
                
                for t in count():
                    base_mem = process.memory_percent()
                    action_idx = self.select_action(state_tensor)
                    action = self.possible_actions[action_idx]
                    perturbed_obs = self.perform_perturb(state, action)
                    rl_agent_act = self.agent.act(perturbed_obs, 0, False)
                    
                    observation, reward, done, _ = env.step(rl_agent_act)
                    total_reward_orig += reward
                    reward = -reward
                    
                    if action[0] == "do_nothing":
                        reward += 0.5
                    
                    if done:
                        reward += 100000 * ((observation.max_step - observation.current_step) / observation.max_step)
                        next_state = None
                        next_state_tensor = None
                    else:
                        next_state = observation
                        next_state_tensor = self.grid2op_obs_to_tensor(next_state)
                    
                    total_reward_adv += reward
                    reward = torch.tensor([reward], device=self.device)
                    
                    action_tensor = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
                    self.memory.push(state_tensor, action_tensor, next_state_tensor, reward)
                    
                    state = next_state
                    state_tensor = next_state_tensor
                    
                    self.optimize_model()
                    
                    # Soft update target network (MATCHING OLD LOGIC)
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    memory_usage[0] += process.memory_percent() - base_mem
                    base_mem = process.memory_percent()
                    
                    with torch.no_grad():
                        for key in policy_net_state_dict:
                            base_mem_ = process.memory_percent()
                            pol = policy_net_state_dict[key].detach()
                            memory_usage[3] += process.memory_percent() - base_mem_
                            
                            base_mem_ = process.memory_percent()
                            pol = np.array(pol)*self.tau
                            memory_usage[4] += process.memory_percent() - base_mem_
                            
                            base_mem_ = process.memory_percent()
                            tar = target_net_state_dict[key].detach().numpy()*(1-self.tau)
                            memory_usage[5] += process.memory_percent() - base_mem_
                            
                            base_mem_ = process.memory_percent()
                            updated_dict = torch.from_numpy(pol + tar)
                            memory_usage[6] += process.memory_percent() - base_mem_
                            
                            del target_net_state_dict[key]
                            target_net_state_dict[key] = updated_dict
                            
                            del updated_dict, pol, tar
                    
                    memory_usage[1] += process.memory_percent() - base_mem
                    base_mem = process.memory_percent()
                    self.target_net.load_state_dict(target_net_state_dict)
                    
                    del target_net_state_dict
                    del policy_net_state_dict
                    memory_usage[2] += process.memory_percent() - base_mem
                    
                    if done:
                        result += [(i_episode, t + 1, total_reward_orig, total_reward_adv, time.time() - start)]
                        base_mem = process.memory_percent()
                        _ = gc.collect()
                        memory_usage[0] += process.memory_percent() - base_mem
                        break
                
            except KeyboardInterrupt:
                print(f"keyboard interrupt after {i_episode} episodes")
                return result
            except Exception as e:
                print(f"exception after {i_episode} episodes {e}")
                return result
        
        return result
    
    def save_model(self, filename_policy_net, filename_target_net):
        """Save model - EXACT MATCH TO OLD."""
        torch.save(self.policy_net, filename_policy_net)
        torch.save(self.target_net, filename_target_net)
    
    def load_model(self, filename_policy_net, filename_target_net=""):
        """Load model using UtilityHelper for safety."""
        from utility.UtilityHelper import UtilityHelper
        
        self.policy_net = UtilityHelper.safe_load_models(
            filename=filename_policy_net,
            device=self.device,
            classes=[DQN]  # Only need DQN class, not MyAgent
        ).to(self.device)
        
        if filename_target_net != "":
            self.target_net = UtilityHelper.safe_load_models(
                filename=filename_target_net,
                device=self.device,
                classes=[DQN]
            ).to(self.device)
    
    def save_perturb_hist(self):
        """Save perturbation history - EXACT MATCH TO OLD."""
        if len(self.perturb_hist) == 0:
            return
        
        x = 0
        while os.path.exists(os.path.join(self.save_dir, f"perturb_hist_{x}.npz")):
            x += 1
        np.savez_compressed(os.path.join(self.save_dir, f"perturb_hist_{x}.npz"), 
                          perturb_hist=self.perturb_hist, 
                          perturb_hist_tuples=self.perturb_hist_tuples)
    
    def reset(self):
        """Reset agent - EXACT MATCH TO OLD."""
        self.save_perturb_hist()
        self.perturb_hist = []
        self.perturb_hist_tuples = []