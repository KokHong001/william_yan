'''
BIO: Trainee Smith undergoes training using PPO, saving his experience and knowledge in his policy file.
When his training completes, he becomes "Agent Smith".
'''

import os
import sys
import glob
import time
from collections import defaultdict, deque
from typing import NamedTuple
import numpy as np
import random
import warnings

from functools import partial

import torch as th
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from workspace.coderone.dungeon.agent import PlayerState

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Logger levels
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

def get_latest_run_id(log_path, log_name = ""):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


class HumanOutputFormat():
    """A human-readable output format producing ASCII tables of key-value pairs.

    Set attribute ``max_length`` to change the maximum length of keys and values
    to write to output (or specify it when calling ``__init__``).

    :param filename_or_file: the file to write the log to
    :param max_length: the maximum length of keys and values to write to output.
        Outputs longer than this will be truncated. An error will be raised
        if multiple keys are truncated to the same value. The maximum output
        width will be ``2*max_length + 7``. The default of 36 produces output
        no longer than 79 characters wide.
    """

    def __init__(self, filename_or_file, max_length = 36):
        self.max_length = max_length
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "write"), f"Expected file or str, got {filename_or_file}"
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values, key_excluded, step) -> None:
        # Create strings for printing
        key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            elif isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[self._truncate(tag)] = ""
            # Remove tag from key
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            truncated_key = self._truncate(key)
            if truncated_key in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[truncated_key] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, string):
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def write_sequence(self, sequence):
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        closes the file
        """
        if self.own_file:
            self.file.close()


class TensorBoardOutputFormat():
    def __init__(self, folder):
        """
        Dumps key/value pairs into TensorBoard's numeric format.

        :param folder: the folder to write the log to
        """
        assert SummaryWriter is not None, "tensorboard is not installed, you can use " "pip install tensorboard to do so"
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, key_values, key_excluded, step):

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)

            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

        # Flush the output to the file
        self.writer.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self.writer = None


class Logger(object):
	"""
	The logger class.

	:param folder: the logging location
	:param output_formats: the list of output formats
	"""

	def __init__(self, folder, output_formats):
		self.name_to_value = defaultdict(float)  # values this iteration
		self.name_to_count = defaultdict(int)
		self.name_to_excluded = defaultdict(str)
		self.level = INFO
		self.dir = folder
		self.output_formats = output_formats

	def record(self, key, value, exclude = None):
		self.name_to_value[key] = value
		self.name_to_excluded[key] = exclude

	def record_mean(self, key, value, exclude = None):
		if value is None:
			self.name_to_value[key] = None
			return
		old_val, count = self.name_to_value[key], self.name_to_count[key]
		self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
		self.name_to_count[key] = count + 1
		self.name_to_excluded[key] = exclude

	def dump(self, step: int = 0) -> None:
		if self.level == DISABLED:
			return
		for _format in self.output_formats:
			_format.write(self.name_to_value, self.name_to_excluded, step)

		self.name_to_value.clear()
		self.name_to_count.clear()
		self.name_to_excluded.clear()

	def log(self, *args, level = INFO):
		if self.level <= level:
			self._do_log(args)

	def debug(self, *args):
		self.log(*args, level=DEBUG)

	def info(self, *args):
		self.log(*args, level=INFO)

	def warn(self, *args):
		self.log(*args, level=WARN)

	def error(self, *args):
		self.log(*args, level=ERROR)

	# Configuration
	# ----------------------------------------
	def set_level(self, level: int):
		self.level = level

	def get_dir(self):
		return self.dir

	def close(self):
		for _format in self.output_formats:
			_format.close()

	# Misc
	# ----------------------------------------
	def _do_log(self, args) -> None:
		for _format in self.output_formats:
			if isinstance(_format, HumanOutputFormat):
				_format.write_sequence(map(str, args))


def configure_logger(verbose, tensorboard_log, tb_log_name = ""):
	"""
	Configure the logger's outputs.

	:param verbose: the verbosity level: 0 no output, 1 info, 2 debug
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param tb_log_name: tensorboard log
	:return: The logger object
	"""
	save_path, format_strings = None, ["stdout"]

	if tensorboard_log is not None and SummaryWriter is None:
		raise ImportError("Trying to log data to tensorboard but tensorboard is not installed.")

	if tensorboard_log is not None and SummaryWriter is not None:
		latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name) - 1
		save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
		os.makedirs(save_path, exist_ok=True)
		if verbose >= 1:
			format_strings = ["stdout", "tensorboard"]
			output_formats = [HumanOutputFormat(sys.stdout), TensorBoardOutputFormat(save_path)]
		else:
			format_strings = ["tensorboard"]
			output_formats = [TensorBoardOutputFormat(save_path)]
	elif verbose == 0:
		format_strings = [""]

	logger = Logger(save_path, output_formats)
	# Only print when some files will be saved
	if len(format_strings) > 0 and format_strings != ["stdout"]:
		logger.log(f"Logging to {save_path}")

	return logger

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
	def __init__(self, image_observation_shape, vector_observation_dim, action_space, lr, 
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
		image_obs = th.as_tensor(image_obs)
		vector_obs = th.as_tensor(vector_obs)

		with th.no_grad():
			action_distribution = self.get_distribution(image_obs, vector_obs, action_masks)
			if deterministic:
				actions = th.argmax(action_distribution.probs, dim=1)
			else:
				actions = action_distribution.sample()
			# Convert to numpy
			actions = actions.cpu().numpy()

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
		features = th.cat(image_features, vector_features, dim=-1)
        
		# Evaluate policies
		pi = self.policy_net(self.policy_mlp_extractor(features))
		HUGE_NEG = th.tensor(-1e8, dtype=self.logits.dtype)
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

class RolloutBufferSamples(NamedTuple):
    image_observations: th.Tensor
    vector_observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor

class RolloutBuffer():
	"""
	Rollout buffer that also stores the invalid action masks associated with each observation.

	:param buffer_size: Max number of element in the buffer
	:param observation_space: Observation space
	:param action_space: Action space
	:param device:
	:param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
		Equivalent to classic advantage when set to 1.
	:param gamma: Discount factor
	:param n_envs: Number of parallel environments
	"""

	def __init__(self, buffer_size, image_observation_shape, vector_observation_dim, action_dim, device, gamma, gae_lambda):
		self.buffer_size = buffer_size
		self.image_observation_shape = image_observation_shape
		self.vector_observation_dim = vector_observation_dim
		self.action_dim = action_dim
		self.pos = 0
		self.full = False
		self.device = device
		self.gae_lambda = gae_lambda
		self.gamma = gamma

		self.image_observations = np.zeros((self.buffer_size,) + self.image_observation_shape, dtype=np.float32)
		self.vector_observations = np.zeros((self.buffer_size, self.vector_observation_dim), dtype=np.float32)
		self.actions = np.zeros(self.buffer_size, dtype=np.float32)
		self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
		self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
		self.returns = np.zeros(self.buffer_size, dtype=np.float32)
		self.episode_starts = np.zeros(self.buffer_size, dtype=np.float32)
		self.values = np.zeros(self.buffer_size, dtype=np.float32)
		self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
		self.generator_ready = False

		self.mask_dim = action_dim
		self.action_masks = np.ones((self.buffer_size, self.mask_dim), dtype=np.bool8)

	def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
		"""
		Post-processing step: compute the lambda-return (TD(lambda) estimate)
		and GAE(lambda) advantage.

		Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
		to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
		where R is the sum of discounted reward with value bootstrap
		(because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

		The TD(lambda) estimator has also two special cases:
		- TD(1) is Monte-Carlo estimate (sum of discounted rewards)
		- TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

		For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

		:param last_values: state value estimation for the last step (one for each env)
		:param dones: if the last step was a terminal step (one bool for each env).
		"""
		# Convert to numpy
		last_values = last_values.clone().cpu().numpy().flatten()

		last_gae_lam = 0
		for step in reversed(range(self.buffer_size)):
			if step == self.buffer_size - 1:
				next_non_terminal = 1.0 - dones
				next_values = last_values
			else:
				next_non_terminal = 1.0 - self.episode_starts[step + 1]
				next_values = self.values[step + 1]
			delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
			last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
			self.advantages[step] = last_gae_lam
		# TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
		# in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
		self.returns = self.advantages + self.values

	def add(self, image_obs, vector_obs, action, reward, action_masks, episode_start, value, log_prob):
		"""
		:param obs: Observation
		:param action: Action
		:param reward:
		:param episode_start: Start of episode signal.
		:param value: estimated value of the current state
			following the current policy.
		:param log_prob: log probability of the action
			following the current policy.
		"""
		if len(log_prob.shape) == 0:
			# Reshape 0-d tensor to avoid error
			log_prob = log_prob.reshape(-1, 1)

		self.image_observations[self.pos] = image_obs.clone().cpu().numpy()
		self.vector_observations[self.pos] = vector_obs.clone().cpu().numpy()
		self.actions[self.pos] = np.array(action).copy()
		self.rewards[self.pos] = np.array(reward).copy()
		self.action_masks[self.pos] = action_masks
		self.episode_starts[self.pos] = np.array(episode_start).copy()
		self.values[self.pos] = value.clone().cpu().numpy()
		self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
		self.pos += 1
		if self.pos == self.buffer_size:
			self.full = True

	def get(self, batch_size = None):
		assert self.full, ""
		indices = np.random.permutation(self.buffer_size)

		# Return everything, don't create minibatches
		if batch_size is None:
			batch_size = self.buffer_size

		start_idx = 0
		while start_idx < self.buffer_size:
			yield self._get_samples(indices[start_idx : start_idx + batch_size])
			start_idx += batch_size

	def _get_samples(self, batch_inds):
		data = (
			th.tensor(self.image_observations[batch_inds]).to(self.device),
			th.tensor(self.vector_observations[batch_inds]).to(self.device),
			th.tensor(self.actions[batch_inds]).to(self.device),
			th.tensor(self.values[batch_inds].flatten()).to(self.device),
			th.tensor(self.log_probs[batch_inds].flatten()).to(self.device),
			th.tensor(self.advantages[batch_inds].flatten()).to(self.device),
			th.tensor(self.returns[batch_inds].flatten()).to(self.device),
			th.tensor(self.action_masks[batch_inds].reshape(-1, self.action_dim)).to(self.device),
		)
		return RolloutBufferSamples(*data)

	def reset(self):
		self.pos = 0
		self.full = False


class PPO():
	"""
	:param learning_rate: The learning rate, it can be a function
		of the current progress remaining (from 1 to 0)
	:param n_steps: The number of steps to run for each environment per update
		(i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
	:param batch_size: Minibatch size
	:param n_epochs: Number of epoch when optimizing the surrogate loss
	:param gamma: Discount factor
	:param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
	:param clip_range: Clipping parameter, it can be a function of the current progress
		remaining (from 1 to 0).
	:param clip_range_vf: Clipping parameter for the value function,
		it can be a function of the current progress remaining (from 1 to 0).
		This is a parameter specific to the OpenAI implementation. If None is passed (default),
		no clipping will be done on the value function.
		IMPORTANT: this clipping depends on the reward scaling.
	:param normalize_advantage: Whether to normalize or not the advantage
	:param ent_coef: Entropy coefficient for the loss calculation
	:param vf_coef: Value function coefficient for the loss calculation
	:param max_grad_norm: The maximum value for the gradient clipping
	:param target_kl: Limit the KL divergence between updates,
		because the clipping is not enough to prevent large update
		see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
		By default, there is no limit on the kl div.
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param create_eval_env: Whether to create a second environment that will be
		used for evaluating the agent periodically. (Only available when passing string for the environment)
	:param policy_kwargs: additional arguments to be passed to the policy on creation
	:param verbose: the verbosity level: 0 no output, 1 info, 2 debug
	:param seed: Seed for the pseudo random generators
	:param device: Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
	:param _init_setup_model: Whether or not to build the network at the creation of the instance
	"""

	def __init__(
		self,
		learning_rate = 3e-4,
		n_steps = 2048,
		batch_size = 64,
		n_epochs = 10,
		gamma = 0.99,
		gae_lambda = 0.95,
		clip_range = 0.2,
		normalize_advantage = True,
		ent_coef = 0.0,
		vf_coef = 0.5,
		max_grad_norm = 0.5,
		tensorboard_log = None,
		verbose = 0,
		seed = None,
	):
		self.verbose = verbose
		self.image_observation_shape = (8, 7, 7)
		self.vector_observation_dim = 14
		self.action_dim = 6
		self.num_timesteps = 0
		self._num_timesteps_at_start = 0
		self.seed = seed
		self.start_time = None
		self.policy = None
		self.learning_rate = learning_rate
		self.tensorboard_log = tensorboard_log
		self.lr = None

		self.n_steps = n_steps
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.ent_coef = ent_coef
		self.vf_coef = vf_coef
		self.max_grad_norm = max_grad_norm

		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.clip_range = clip_range
		self.normalize_advantage = normalize_advantage

		if seed is not None:
			# Seed python RNG
			random.seed(seed)
			# Seed numpy RNG
			np.random.seed(seed)
			# seed the RNG for all devices (both CPU and CUDA)
			th.manual_seed(seed)

		self.train_device = 'cuda' if th.cuda.is_available() else 'cpu'

		self.policy = MaskableActorCriticPolicy(self.image_observation_shape, self.vector_observation_dim, self.action_dim, learning_rate)

		self.rollout_buffer = RolloutBuffer(
			n_steps,
			self.image_observation_shape,
			self.vector_observation_dim,
			self.action_dim,
			self.train_device,
			gamma=self.gamma,
			gae_lambda=self.gae_lambda,
		)

		self.logger = configure_logger(self.verbose, self.tensorboard_log, 'PPO')

		self.start_time = time.time()
		self._last_episode_starts = True
		self.iteration = 0
		self._n_updates = 0

		self.state_image = None
		self.state_vec = None
		self.action = None
		self.action_mask = None
		self.value = None
		self.log_prob = None

	def collect_rollouts(self, state_image, state_vec, action_mask, last_reward, done):
		"""
		Collect experiences using the current policy and fill a ``RolloutBuffer``.
		The term rollout here refers to the model-free notion and should not
		be used with the concept of rollout used in model-based RL or planning.

		This method is largely identical to the implementation found in the parent class.

		:param rollout_buffer: Buffer to fill with rollouts
		:param n_steps: Number of experiences to collect per environment
		:return: True if function returned with at least `n_rollout_steps`
			collected, False if callback terminated rollout prematurely.
		"""
		if not self._last_episode_starts:
			self.rollout_buffer.add(
				self.state_image,
				self.state_vec,
				self.action,
				last_reward,
				self.action_mask,
				done,
				self.value,
				self.log_prob,
			)

		self._last_episode_starts = done
		
		if not done:
			self.state_image = th.tensor(state_image).unsqueeze(0)
			self.state_vec = th.tensor(state_vec).unsqueeze(0)
			self.action_mask = th.tensor(action_mask)

			with th.no_grad():
				self.action, self.value, self.log_prob = self.policy(self.state_image, self.state_vec, self.action_mask)

			action = self.action.cpu().numpy()
		else:
			action = None

		self.num_timesteps += 1

		if self.rollout_buffer.full:
			with th.no_grad():
				# Compute value for the last timestep
				# Masking is not needed here, the choice of action doesn't matter.
				# We only want the value of the current observation.
				value = self.policy.predict_values(self.state_image, self.state_vec)

			self.rollout_buffer.compute_returns_and_advantage(value, done)

		return action

	def train(self) -> None:
		"""
		Update policy using the currently gathered rollout buffer.
		"""
		entropy_losses = []
		pg_losses, value_losses = [], []
		clip_fractions = []

		self.policy.to(self.train_device)

		continue_training = True
		# train for n_epochs epochs
		for epoch in range(self.n_epochs):
			approx_kl_divs = []
			# Do a complete pass on the rollout buffer
			for rollout_data in self.rollout_buffer.get(self.batch_size):
				actions = rollout_data.actions

				values, log_prob, entropy = self.policy.evaluate_actions(
					rollout_data.image_observations,
					rollout_data.vector_observations,
					rollout_data.action_masks,
					actions,
				)

				values = values.flatten()
				# Normalize advantage
				advantages = rollout_data.advantages
				if self.normalize_advantage:
					advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

				# ratio between old and new policy, should be one at the first iteration
				ratio = th.exp(log_prob - rollout_data.old_log_prob)

				# clipped surrogate loss
				policy_loss_1 = advantages * ratio
				policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
				policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

				# Logging
				pg_losses.append(policy_loss.item())
				clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float()).item()
				clip_fractions.append(clip_fraction)

				values_pred = values

				# Value loss using the TD(gae_lambda) target
				value_loss = F.mse_loss(rollout_data.returns, values_pred)
				value_losses.append(value_loss.item())

				# Entropy loss favor exploration
				if entropy is None:
					# Approximate entropy when no analytical form
					entropy_loss = -th.mean(-log_prob)
				else:
					entropy_loss = -th.mean(entropy)

				entropy_losses.append(entropy_loss.item())

				loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

				# Calculate approximate form of reverse KL Divergence for early stopping
				# see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
				# and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
				# and Schulman blog: http://joschu.net/blog/kl-approx.html
				with th.no_grad():
					log_ratio = log_prob - rollout_data.old_log_prob
					approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
					approx_kl_divs.append(approx_kl_div)

				# Optimization step
				self.policy.optimizer.zero_grad()
				loss.backward()
				# Clip grad norm
				th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.policy.optimizer.step()

			if not continue_training:
				break

		self._n_updates += self.n_epochs

		self.policy.to('cpu')
		self.rollout_buffer.reset()

		th.save(self.policy.state_dict(), "Agent_Smith.pt")

		# Logs
		self.logger.record("train/entropy_loss", np.mean(entropy_losses))
		self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
		self.logger.record("train/value_loss", np.mean(value_losses))
		self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
		self.logger.record("train/clip_fraction", np.mean(clip_fractions))
		self.logger.record("train/loss", loss.item())
		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

	def learn(self, state_image, state_vec, action_mask, last_reward, last_done):
		action = self.collect_rollouts(state_image, state_vec, action_mask, last_reward, last_done)

		if self.rollout_buffer.full:
			self.iteration += 1
			self.train()

			# Display training infos
			fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
			self.logger.record("time/iterations", self.iteration, exclude="tensorboard")
			self.logger.record("time/fps", fps)
			self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
			self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
			self.logger.dump(step=self.num_timesteps)

		return action


class Agent:

	def __init__(self):		
		self.policy = PPO(n_steps=300, tensorboard_log="./tensorboard/", verbose=1)
		self.last_episode_starts = True
		self.last_player_state = None
		self.games_won = deque([], maxlen=100)

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
		last_reward, done = self.compute_reward(game_state, player_state)
		action = self.policy.learn(state_image, state_vec, action_mask, last_reward, done)

		return actions_palette[action]

	def on_game_over(self, game_state, player_state):
		state_image, state_vec, action_mask = self.build_input(game_state, player_state)
		last_reward, done = self.compute_reward(game_state, player_state)
		self.policy.learn(state_image, state_vec, action_mask, last_reward, done)

		self.last_episode_starts = True

		self.games_won.append(float(game_state.winner[0]==player_state.id))
		print("Game won:", np.array(self.games_won).mean())

	def compute_reward(self, game_state, player_state):
		done = game_state.is_over
		if done:
			if player_state.hp > player_state.info[0].hp:
				reward = 1000
			else:
				reward = -1000
		else:
			reward = 0
			last_reward = 0 if self.last_episode_starts else self.last_player_state.reward
			reward += player_state.reward - last_reward
			last_hp = player_state.hp if self.last_episode_starts else self.last_player_state.hp
			reward += (player_state.hp - last_hp) * 100
			last_opp_hp = player_state.info[0].hp if self.last_episode_starts else self.last_player_state.info[0].hp
			reward += (last_opp_hp - player_state.info[0].hp) * 100

			self.last_player_state = player_state

		return reward, done

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


