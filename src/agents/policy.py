"""Small image encoder with named activation points for future actor analysis."""

from collections import OrderedDict

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SmallCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 16, 3, stride=2, padding=1)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(16, 32, 3, stride=2, padding=1)),
                    ("relu2", nn.ReLU()),
                    ("conv3", nn.Conv2d(32, 32, 3, stride=2, padding=1)),
                    ("relu3", nn.ReLU()),
                ]
            )
        )
        self.projection = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("linear", nn.Linear(32 * 11 * 11, features_dim)),
                    ("relu", nn.ReLU()),
                ]
            )
        )

    def forward(self, observations):
        # SB3 handles uint8 -> float / 255 exactly once before this call.
        return self.projection(self.conv(observations))


def policy_kwargs(features_dim=128):
    return dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs={"features_dim": features_dim},
        net_arch=dict(pi=[64], vf=[64]),
        activation_fn=nn.ReLU,
        share_features_extractor=True,
        normalize_images=True,
    )


def activation_points(policy):
    """Names only: collection and interventions are intentionally deferred."""
    return {
        "encoder.conv1": policy.features_extractor.conv.relu1,
        "encoder.conv2": policy.features_extractor.conv.relu2,
        "encoder.conv3": policy.features_extractor.conv.relu3,
        "encoder.features": policy.features_extractor.projection.relu,
        "actor.hidden": policy.mlp_extractor.policy_net[1],
    }


def action_probabilities(model, observations):
    model.policy.set_training_mode(False)
    with torch.inference_mode():
        tensor, _ = model.policy.obs_to_tensor(observations)
        return model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()
