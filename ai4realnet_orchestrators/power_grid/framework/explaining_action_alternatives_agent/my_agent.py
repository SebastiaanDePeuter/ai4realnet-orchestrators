import numpy as np

import torch
from torch.distributions.normal import Normal

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation

def normalize_log_belief(x):
    return x - x.max() - np.log(np.exp(x - x.max()).sum())

def _construct_sequential(in_size, layer_sizes, out_size, final_std = np.sqrt(2.0)):
    if len(layer_sizes) == 0:
        return torch.nn.Sequential(torch.nn.Linear(in_size, out_size))
    
    layers = [torch.nn.Linear(in_size, layer_sizes[0]), torch.nn.Tanh()]
    for i in range(1, len(layer_sizes)):
        layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(layer_sizes[-1], out_size))
    return torch.nn.Sequential(*layers)

class NNAgent(torch.nn.Module):
    def __init__(self, obs_size, belief_size, N_actions, N_outcomes, state_emb_size, state_emb_layer_sizes, actor_layer_sizes, critic_layer_sizes):
        super().__init__()
        self.structured_action_space = (N_actions, N_outcomes)
        action_space_dim = np.prod(self.structured_action_space)

        self.state_embedder = _construct_sequential(obs_size+belief_size, state_emb_layer_sizes, state_emb_size)
        self.critic_head = _construct_sequential(state_emb_size, critic_layer_sizes, 1, final_std=1.0)
        self.actor_mean_head = _construct_sequential(state_emb_size, actor_layer_sizes, action_space_dim, final_std=0.01)
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, action_space_dim))

    def get_explanation(self, s, b, action=None):
        state_emb = self.state_embedder(torch.cat([s, b], 1))
        action_mean = self.actor_mean_head(state_emb)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        flat_action = probs.sample()
        action = flat_action.reshape(s.shape[0], *self.structured_action_space)

        return action

def get_multi_user_menu_choice_likelihood(explanations, thetas, beta, choice):
    a_logprobs = np.dot(explanations, thetas.T) * beta # N_actions x N_user_types
    a_logprobs_max = a_logprobs.max(axis=0)
    norm_logprobs = a_logprobs - a_logprobs_max - np.log(np.exp(a_logprobs - a_logprobs_max).sum(axis=0))
    return norm_logprobs[choice,:]

class ExplainingActionAlternativesAgent(BaseAgent):
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 NNAgent_kwargs = {"s_emb_size": 64,
                                   "state_embedder_layers": [64],
                                   "outcome_predictor_layers": [64, 64],
                                   "critic_layer_sizes": [64, 64]
                                   }
                 ):
        super().__init__(g2op_action_space)
        self._gym_act_space = gym_act_space
        self._gym_obs_space = gym_obs_space

        self.agent_kwargs = NNAgent_kwargs

        self.DM_types = None
        self.opred = None

        self.log_belief = None
        self.beta = 5.0 # standard temperature for Boltzmann user model
        # the simulated decision maker is only looking to maxmize environment reward, hence only that weight is non-zero
        self.theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def act(self, g2op_obs: BaseObservation, current_return: float, done) -> BaseAction:
        # convert grid2op observation into "gym" observation
        gym_obs = self._gym_obs_space.to_gym(g2op_obs)

        # generate explanation from outcome predictor
        with torch.no_grad():
            expl = self.opred.get_explanation(torch.from_numpy(gym_obs)[None,...], torch.from_numpy(self.log_belief)[None,...])[0,...].detach().numpy()

        # simulate user choice
        logprobs = normalize_log_belief(np.dot(expl, self.theta) * self.beta)
        user_choice = np.random.multinomial(1, np.exp(logprobs.flatten()) / np.exp(logprobs).sum() )
        chosen_action = user_choice.argmax()

        # update belief over user outcome preferences
        self.log_belief += get_multi_user_menu_choice_likelihood(expl, self.DM_types, self.beta, chosen_action)
        self.log_belief = normalize_log_belief(self.log_belief)

        # convert "gym" action back into grid2op action
        grid2op_act = self._gym_act_space.from_gym(chosen_action)
        return grid2op_act

    def load_state(self, loadstate_path):
        self.DM_types = np.load(loadstate_path[:-4] + "_DM_types.npy")

        self.opred = NNAgent(self._gym_obs_space.shape[0], self.DM_types.shape[0], self._gym_act_space.n, 9, self.agent_kwargs["s_emb_size"], self.agent_kwargs["state_embedder_layers"], self.agent_kwargs["outcome_predictor_layers"], self.agent_kwargs["critic_layer_sizes"])
        self.opred.load_state_dict(torch.load(loadstate_path))

    def reset(self, obs):
        # resample the user type and res-set the log-belief
        self.log_belief = normalize_log_belief(np.zeros(self.DM_types.shape[0], np.float32))

    def save_state(self, savestate_path):
        raise NotImplementedError

    def seed(self, seed):
        pass

if __name__=="__main__":
    from lightsim2grid import LightSimBackend
    from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace
    from grid2op.Runner import Runner

    env = grid2op.make("ai4realnet_small", backend=LightSimBackend())

    # customize observation space
    obs_attr_to_keep = ["rho", "p_or"]
    act_attr_to_keep =  ["set_line_status_simple"]
    gym_observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr_to_keep)
    gym_action_space = DiscreteActSpace(env.action_space, attr_to_keep=act_attr_to_keep)

    agent = ExplainingActionAlternativesAgent(env.action_space, gym_action_space, gym_observation_space)
    agent.load_state("outcome_predictor_weights.pth")
    agent.reset(None)

    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

    mean_rew = []
    mean_rel_ep_len = []
    for _, _, cum_reward, nb_time_step, max_ts in runner.run(nb_episode=25, max_iter=env.chronics_handler.max_timestep()):
        mean_rew.append(cum_reward)
        mean_rel_ep_len.append(nb_time_step / max_ts)
    print(np.mean(mean_rew), np.std(mean_rew), np.mean(mean_rel_ep_len), np.std(mean_rel_ep_len))