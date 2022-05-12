'''
BIO: Trainee Smith undergoes training using PPO, saving his experience and knowledge in his policy file.
When his training completes, he becomes "Agent Smith".
'''

import numpy as np

from functools import partial

import torch as th
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class BoardCNN(nn.Module):
	def __init__(self, observation_shape, features_dim, latent_dim):
		
		super().__init__()
		
		self.features_dim = features_dim
		n_input_channels = observation_shape[0]
		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, latent_dim, kernel_size=3, stride=1, padding=0),
			nn.LeakyReLU(),
			nn.Conv2d(latent_dim, latent_dim, kernel_size=2, stride=1, padding=0),
			nn.LeakyReLU(),
			nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(),
			nn.Flatten(),
		)

		# Compute shape by doing one forward pass
		with th.no_grad():
			n_flatten = self.cnn(th.rand(observation_shape)[None]).shape[1]

		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.LeakyReLU())

	def forward(self, observations: th.Tensor) -> th.Tensor:
		return self.linear(self.cnn(observations))

class MaskableActorCriticPolicy(nn.Module):
	def __init__(self, image_observation_shape, vector_observation_dim, action_space, lr=0.0001, 
        ortho_init = True, optimizer_class = th.optim.Adam, optimizer_kwargs = None,):

		super().__init__()
		
		self.image_latent_dim = 64
		self.image_feature_dim = 128
		self.vector_latent_dim = 16
		self.vector_feature_dim = 32
		self.feature_dim = self.image_feature_dim + self.vector_feature_dim
		self.latent_dim = 64

		self.image_features_extractor = BoardCNN(image_observation_shape, self.image_feature_dim, self.image_latent_dim)
		self.vector_feature_extractor = nn.Sequential(
			nn.Linear(vector_observation_dim, self.vector_latent_dim), 
			nn.LeakyReLU(),
			nn.Linear(self.vector_latent_dim, self.vector_feature_dim)
		)
		self.policy_mlp_extractor = nn.Sequential(
			nn.Linear(self.feature_dim, self.latent_dim),
			nn.LeakyReLU(),
			nn.Linear(self.latent_dim, self.latent_dim),
			)
		self.value_mlp_extractor = nn.Sequential(
			nn.Linear(self.feature_dim, self.latent_dim),
			nn.LeakyReLU(),
			nn.Linear(self.latent_dim, self.latent_dim),
			)
		self.policy_net = nn.Linear(self.latent_dim, action_space)
		self.value_net = nn.Linear(self.latent_dim, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
		if ortho_init:
			module_gains = {
				self.image_features_extractor: np.sqrt(2),
				self.vector_feature_extractor: np.sqrt(2),
                self.policy_mlp_extractor: np.sqrt(2),
                self.value_mlp_extractor: np.sqrt(2),
                self.policy_net: 0.01,
                self.value_net: 1,
            }
			for module, gain in module_gains.items():
				module.apply(partial(self.init_weights, gain=gain))

		if optimizer_kwargs is None:
			optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
			if optimizer_class == th.optim.Adam:
				optimizer_kwargs["eps"] = 1e-5
		self.optimizer = optimizer_class(self.parameters(), lr=lr, **optimizer_kwargs)

	def forward(self, image_obs, vector_obs, action_mask, deterministic = False):
		"""
		Forward pass in all the networks (actor and critic)

		:param image_obs: 2D image Observation
		:param vector_obs: 1D vector Observation
		:param action_masks: Action masks to apply to the action distribution
		:param deterministic: Whether to sample or use deterministic actions
		:return: action, value and log probability of the action
		"""
        # Preprocess the observation if needed
		image_features = self.image_features_extractor(image_obs)
		vector_features = self.vector_feature_extractor(vector_obs)
		features = th.cat((image_features, vector_features), dim=-1)
        
		# Evaluate policies & values for the given observations
		pi = self.policy_net(self.policy_mlp_extractor(features))
		values = self.value_net(self.value_mlp_extractor(features))

		# Get actions
		HUGE_NEG = th.tensor(-1e8, dtype=pi.dtype)
		action_logits = th.where(action_mask, pi, HUGE_NEG)
		action_distribution = Categorical(logits=action_logits)
		if deterministic:
			actions = th.argmax(action_distribution.probs, dim=1)
		else:
			actions = action_distribution.sample()
		log_prob = action_distribution.log_prob(actions)
		return actions, values, log_prob

	def predict(self, image_obs, vector_obs, action_masks, deterministic = False):
		"""
		Get the policy action and state from an observation (and optional state).
		Includes sugar-coating to handle different observations (e.g. normalizing images).

		:param image_obs: 2D image Observation
		:param vector_obs: 1D vector Observation
		:param action_masks: Action masks to apply to the action distribution
		:param deterministic: Whether to sample or use deterministic actions
		:return: the model's action
		"""
		self.train(False)
		image_obs = th.as_tensor(image_obs).unsqueeze(0)
		vector_obs = th.as_tensor(vector_obs).unsqueeze(0)
		action_masks = th.as_tensor(action_masks).unsqueeze(0)

		with th.no_grad():
			action_distribution = self.get_distribution(image_obs, vector_obs, action_masks)
			if deterministic:
				actions = th.argmax(action_distribution.probs, dim=1)
			else:
				actions = action_distribution.sample()
			# Convert to numpy
			actions = actions.item()

		return actions

	def evaluate_actions(self, image_obs, vector_obs, action_masks, actions):
		"""
		Evaluate actions according to the current policy,
		given the observations.

		:param image_obs: 2D image Observation
		:param vector_obs: 1D vector Observation
		:param action_masks:
		:param actions:
		:return: estimated value, log likelihood of taking those actions
			and entropy of the action distribution.
		"""
		image_features = self.image_features_extractor(image_obs)
		vector_features = self.vector_feature_extractor(vector_obs)
		features = th.cat((image_features, vector_features), dim=-1)

		# Evaluate policies & values for the given observations
		pi = self.policy_net(self.policy_mlp_extractor(features))
		values = self.value_net(self.value_mlp_extractor(features))

		# Get actions
		HUGE_NEG = th.tensor(-1e8, dtype=pi.dtype, device=pi.device)
		action_logits = th.where(action_masks, pi, HUGE_NEG)
		action_distribution = Categorical(logits=action_logits)

		log_prob = action_distribution.log_prob(actions)
		return values, log_prob, action_distribution.entropy()

	def get_distribution(self, image_obs, vector_obs, action_masks):
		"""
		Get the current policy distribution given the observations.

		:param image_obs: 2D image Observation
		:param vector_obs: 1D vector Observation
		:param action_masks:
		:return: the action distribution.
		"""
        # Preprocess the observation if needed
		image_features = self.image_features_extractor(image_obs)
		vector_features = self.vector_feature_extractor(vector_obs)
		features = th.cat((image_features, vector_features), dim=-1)
        
		# Evaluate policies
		pi = self.policy_net(self.policy_mlp_extractor(features))
		HUGE_NEG = th.tensor(-1e8, dtype=pi.dtype, device=pi.device)
		action_logits = th.where(action_masks, pi, HUGE_NEG)
		action_distribution = Categorical(logits=action_logits)
		return action_distribution

	def predict_values(self, image_obs, vector_obs):
		"""
		Get the estimated values according to the current policy given the observations.

		:param image_obs: 2D image Observation
		:param vector_obs: 1D vector Observation
		:return: the estimated values.
		"""
		image_features = self.image_features_extractor(image_obs)
		vector_features = self.vector_feature_extractor(vector_obs)
		features = th.cat((image_features, vector_features), dim=-1)
		values = self.value_net(self.value_mlp_extractor(features))
		return values

	@staticmethod
	def init_weights(module: nn.Module, gain: float = 1) -> None:
		"""
		Orthogonal initialization (used in PPO and A2C)
		"""
		if isinstance(module, (nn.Linear, nn.Conv2d)):
			nn.init.orthogonal_(module.weight, gain=gain)
			if module.bias is not None:
				module.bias.data.fill_(0.0)

class Agent:

    def __init__(self):		
        self.policy = MaskableActorCriticPolicy((8, 7, 7), 14, 6)
        self.policy.load_state_dict(th.load('agent_smith.pt'))
        self.policy.eval()

    def next_move(self, game_state, player_state):
        '''
        This method is called each time your Agent is required to choose an action
        If you're just starting out or are new to Python, you can place all your 
        code within the ### CODE HERE ### tags. If you're more familiar with Python
        and how classes and modules work, then go nuts. 
        (Although we recommend that you use the Scrims to check your Agent is working)
        '''

        # a list of all the actions your Agent can choose from
        actions_palette = ['','u','d','l','r','p']

        state_image, state_vec, action_mask = self.build_input(game_state, player_state)
        action = self.policy.predict(state_image, state_vec, action_mask, deterministic = True)

        return actions_palette[action]

    def on_game_over(self, game_state, player_state):
        pass

    def build_input(self, game_state, player_state):
        state_image = np.zeros((8, game_state.size[0], game_state.size[1]), dtype=np.float32)
        state_image[0, player_state.location[0], player_state.location[1]] = 1.0
        for pos in game_state.opponents(player_state.id):
            state_image[1, pos[0], pos[1]] = 1.0
        for pos in game_state.ammo:
            state_image[2, pos[0], pos[1]] = 1.0
        for pos in game_state.treasure:
            state_image[3, pos[0], pos[1]] = 1.0
        for pos in game_state.bombs:
            state_image[4, pos[0], pos[1]] = 1.0
        for pos in game_state.indestructible_blocks:
            state_image[5, pos[0], pos[1]] = 1.0
        for pos in game_state.soft_blocks:
            state_image[6, pos[0], pos[1]] = 1.0
        for pos in game_state.ore_blocks:
            state_image[7, pos[0], pos[1]] = 1.0

        vec = np.zeros(4, dtype=np.float32)
        vec[0] = game_state.tick_number / 1800.0
        vec[1] = player_state.ammo / 10.0
        vec[2] = player_state.hp / 3.0
        vec[3] = player_state.power / 5.0
        state_vec = np.concatenate((vec, np.array(game_state._occurred_event, dtype=np.float32)))

        action_mask = np.zeros(6, dtype=np.bool8)
        location_up = (player_state.location[0], player_state.location[1]+1)
        action_mask[1] = not(game_state.is_in_bounds(location_up)) or game_state.is_occupied(location_up)
        location_down = (player_state.location[0], player_state.location[1]-1)
        action_mask[2] = not(game_state.is_in_bounds(location_down)) or game_state.is_occupied(location_down)
        location_left = (player_state.location[0]-1, player_state.location[1])
        action_mask[3] = not(game_state.is_in_bounds(location_left)) or game_state.is_occupied(location_left)
        location_right = (player_state.location[0]+1, player_state.location[1])
        action_mask[4] = not(game_state.is_in_bounds(location_right)) or game_state.is_occupied(location_right)
        action_mask[5] = (player_state.ammo == 0) or (player_state.location in game_state.bombs)

        return state_image, state_vec, action_mask


