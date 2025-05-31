from flgo.utils.fmodule import FModule
import torch.nn as nn
from robomimic.models.policy_nets import RNNGMMActorNetwork
from robomimic.config.config import Config
from collections import OrderedDict
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as BaseNets
from  collections import OrderedDict, defaultdict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.models.base_nets as BaseNets

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)

class PolicyModel(nn.Module):
    def __init__(self, obs_config, obs_key_shape, ac_dim, algo_config_rnn, algo_config_gmm, actor_layer_dims=[], obs_normalization_stats=None):
        super().__init__()
        self.obs_config = obs_config
        self.obs_key_shape = obs_key_shape
        self.obs_shapes, self.goal_shapes,_ = self._create_shapes(obs_config.modalities, obs_key_shape)
        self.ac_dim = ac_dim
        self.actor_layer_dims = actor_layer_dims
        self.obs_config_encoder = obs_config['encoder']
        self.algo_config_rnn = algo_config_rnn
        self.algo_config_gmm = algo_config_gmm
        self._create_networks()
        self.obs_normalization_stats = obs_normalization_stats
        self.all_obs_keys = list(self.obs_key_shape.keys())

    def set_obs_normalization_stats(self, obs_normalization_stats):
        self.obs_normalization_stats = obs_normalization_stats

    def get_obs_normalization_stats(self):
        return self.obs_normalization_stats

    def _create_shapes(self, obs_config_modalities, obs_key_shapes):
        obs_shapes = OrderedDict()
        goal_shapes = OrderedDict()
        subgoal_shapes = OrderedDict()
        for k in obs_key_shapes:
            if "obs" in obs_config_modalities and k in [obs_key for modality in obs_config_modalities.obs.values() for
                                                        obs_key in modality]:
                obs_shapes[k] = obs_key_shapes[k]
            if "goal" in obs_config_modalities and k in [obs_key for modality in obs_config_modalities.goal.values() for
                                                         obs_key in modality]:
                goal_shapes[k] = obs_key_shapes[k]
            if "subgoal" in obs_config_modalities and k in [obs_key for modality in
                                                            obs_config_modalities.subgoal.values() for obs_key in
                                                            modality]:
                subgoal_shapes[k] = obs_key_shapes[k]
        return obs_shapes, goal_shapes, subgoal_shapes

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.actor_layer_dims,
            num_modes=self.algo_config_gmm.num_modes,
            min_std=self.algo_config_gmm.min_std,
            std_activation=self.algo_config_gmm.std_activation,
            low_noise_eval=self.algo_config_gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config_encoder),
            **BaseNets.rnn_args_from_config(self.algo_config_rnn),
        )
        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config_rnn["horizon"]
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config_rnn.get("open_loop", False)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]
        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(input_batch)
        # return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _prepare_observation(self, obs, device):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            obs (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
        """
        obs = TensorUtils.to_tensor(obs)
        obs = TensorUtils.to_batch(obs)
        obs = TensorUtils.to_device(obs, device)
        obs = TensorUtils.to_float(obs)
        if self.obs_normalization_stats is not None:
            # ensure obs_normalization_stats are torch Tensors on proper device
            obs_normalization_stats = TensorUtils.to_float(
                TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), device)
            )
            # limit normalization to obs keys being used, in case environment includes extra keys
            obs = OrderedDict({k: obs[k] for k in self.all_obs_keys})
            obs = ObsUtils.normalize_obs(obs, obs_normalization_stats=obs_normalization_stats)
        else:
            obs = OrderedDict({k: obs[k] for k in self.all_obs_keys})
        return obs

    def get_action(self, obs_dict, goal_dict=None, device='cuda'):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        obs_dict = self._prepare_observation(obs_dict, device)
        assert not self.nets.training
        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=device)
            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))
        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs
        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        log_probs = dists.log_prob(batch["actions"])
        predictions = OrderedDict(log_probs=log_probs)
        return predictions

    def forward(self, batch):
        input_batch = self.process_batch_for_training(batch)
        input_batch = self.postprocess_batch_for_training(input_batch, self.get_obs_normalization_stats())
        predictions = self._forward_training(input_batch)
        loss = -predictions["log_probs"].mean()
        return loss

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        # batch = TensorUtils.to_device(batch, device)
        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(TensorUtils.to_tensor(obs_normalization_stats))
        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]
        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_obs(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])
        recurse_helper(batch)
        return batch


def get_model():
    algo_config_rnn = Config(**{
            "enabled": True,
            "horizon": 10,
            "hidden_dim": 400,
            "rnn_type": "LSTM",
            "num_layers": 2,
            "open_loop": False,
            "kwargs": {
                "bidirectional": False
            }
    })
    algo_config_gmm = Config(**{
        "enabled": True,
        'num_modes': 5,
        "min_std": 0.0001,
        "std_activation": "softplus",
        "low_noise_eval": True,
    })
    ac_dim = 7
    obs_key_shapes = OrderedDict([('object', [10]), ('robot0_eef_pos', [3]), ('robot0_eef_quat', [4]), ('robot0_gripper_qpos', [2])])
    obs_config = Config(**{
        "modalities": {
            "obs": {
                "low_dim": [
                    "object",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                ],
                "rgb": [],
                "depth": [],
                "scan": []
            },
            "goal": {
                "low_dim": [],
                "rgb": [],
                "depth": [],
                "scan": []
            }
        },
        "encoder": {
            "low_dim": {
                "core_class": None,
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {}
            },
            "rgb": {
                "core_class": "VisualCore",
                "core_kwargs": {
                    "feature_dimension": 64,
                    "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": {
                        "pretrained": False,
                        "input_coord_conv": False
                    },
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": {
                        "num_kp": 32,
                        "learnable_temperature": False,
                        "temperature": 1.0,
                        "noise_std": 0.0
                    }
                },
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {
                    "crop_height": 76,
                    "crop_width": 76,
                    "num_crops": 1,
                    "pos_enc": False
                }
            },
            "depth": {
                "core_class": "VisualCore",
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {}
            },
            "scan": {
                "core_class": "ScanCore",
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {}
            }
        }
    })
    return PolicyModel(obs_config, obs_key_shapes, ac_dim, algo_config_rnn, algo_config_gmm)