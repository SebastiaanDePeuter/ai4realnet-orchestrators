import os
from typing import Optional
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback 
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.callbacks import CallbackList

from grid2op.Reward import LinesCapacityReward
from grid2op.gym_compat import GymEnv
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from domain_shift_kpis.agents import BaseAgent

class MyAgent(SB3Agent, BaseAgent):
    def __init__(self,
                 name: str,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type=PPO,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 iter_num=None):
        if name is None:
            name = "PPO_SB3"

        SB3Agent.__init__(self, g2op_action_space, gym_act_space, gym_obs_space,
                          nn_type, nn_path, nn_kwargs, custom_load_dict,
                          gymenv, iter_num)
        BaseAgent.__init__(self, name)
        
        self._loaded = False
        if nn_path is not None:
            self.load(path=nn_path)
            self._loaded = True
        
        
    def load(self, path: Optional[str]=None):
        if path is None:
            if self._nn_path is None:
                raise Exception("The path variable should be set before loading the model.")
        else:
            self._nn_path = path
        
        super().load()
        
        self._loaded = True
        
def train(agent, env, **kwargs):
    load_path = kwargs.get("load_path", None)
    save_path = kwargs.get("save_path", None)
    save_freq = kwargs.get("save_freq", None)
    train_steps = kwargs.get("train_steps", int(1e3))
    
    if load_path is not None:
        fine_tune = True
        agent.nn_model = PPO.load(path=load_path,
                                  custom_objects={"observation_space" : env.observation_space,
                                                  "action_space": env.action_space})
        agent.nn_model.set_env(env)
        
    if save_path is None:
        save_path = os.path.join("logs", agent.name)
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
    callbacks = []
    if save_freq is not None:
        callbacks.append(CheckpointCallback(save_freq=save_freq,
                                            save_path=save_path,
                                            name_prefix=agent.name))
    
        
    # Train the model
    agent.nn_model.learn(total_timesteps=train_steps,
                         progress_bar=True,
                         callback=CallbackList(callbacks))
    
    
    # save the model
    agent.nn_model.save(os.path.join(save_path, agent.name))
    
    return agent

def evaluate(agent, env, **kwargs):
    mean_reward, std_reward = evaluate_policy(agent.nn_model, 
                                              env, 
                                              **kwargs)
    
    return np.mean(mean_reward), np.mean(std_reward)

def make_agent(name, env, env_gym):
    """make a PPO agent from environment

    Parameters
    ----------
    env : `Environment`
        grid2op.Environment
    env_gym : `GymEnv`
        A gym environment corresponding to the grid2op environment
    """    
    logs_dir = "model_logs"
    if logs_dir is not None:
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        model_path = os.path.join(logs_dir, "PPO_SB3")
    
    net_arch=[200, 200, 200]
    policy_kwargs = {}
    policy_kwargs["net_arch"] = net_arch
    
    nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": True,
            "learning_rate": 3e-4,
            "tensorboard_log": model_path,
            "policy_kwargs": policy_kwargs,
            "device": "auto"
    }
        
    agent = MyAgent(name=name,
                    g2op_action_space=env.action_space,
                    gym_act_space=env_gym.action_space,
                    gym_obs_space=env_gym.observation_space,
                    nn_kwargs=nn_kwargs
                    )
    
    return agent
