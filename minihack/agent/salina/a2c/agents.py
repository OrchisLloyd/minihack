#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import TAgent, instantiate_class
from salina.agents import Agents
from minihack.agent.salina.agents import *


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def masked_tensor(tensor0, tensor1, mask):
    """Compute a tensor by combining two tensors with a mask

    :param tensor0: a Bx(N) tensor
    :type tensor0: torch.Tensor
    :param tensor1: a Bx(N) tensor
    :type tensor1: torch.Tensor
    :param mask: a B tensor
    :type mask: torch.Tensor
    :return: (1-m) * tensor 0 + m *tensor1 (averafging is made ine by line)
    :rtype: tensor0.dtype
    """
    s = tensor0.size()
    assert s[0] == mask.size()[0]
    m = mask
    for i in range(len(s) - 1):
        m = mask.unsqueeze(-1)
    m = m.repeat(1, *s[1:])
    m = m.float()
    out = ((1.0 - m) * tensor0 + m * tensor1).type(tensor0.dtype)
    return out


class GRUAgent(TAgent):
    def __init__(self, n_input, n_output, obs_name, z_name):
        super().__init__()
        self.gru = nn.GRUCell(n_input, n_output)
        self.n_output = n_output
        self.obs_name = obs_name
        self.z_name = z_name

    def forward(self, t=None, **args):
        if not t is None:
            _input = self.get((self.obs_name, t))
            B = _input.size()[0]
            if t == 0:
                _input_z = torch.zeros(B, self.n_output, device=_input.device)
            else:
                _input_z = self.get((self.z_name, t - 1))
            initial_state = self.get(("env/initial_state", t))

            # Re-initialize state if we are in an env initial_state
            _input_z = masked_tensor(
                _input_z,
                torch.zeros(B, self.n_output, device=_input.device),
                initial_state,
            )

            out = self.gru(_input, _input_z)
            self.set((self.z_name, t), out)
        else:
            T = self.workspace.time_size()
            for t in range(T):
                self.forward(t=t)


class A2C_PolicyAgent(TAgent):
    def __init__(self, n_input, hidden_size, n_layers, n_actions, input_name):
        super().__init__()
        sizes = [hidden_size for _ in range(n_layers)]
        self.mlp = mlp([n_input] + sizes + [n_actions], nn.ReLU)
        self.input_name = input_name

    def forward(self, stochastic, t=None, **args):
        if not t is None:
            _input = self.get((self.input_name, t))
            scores = self.mlp(_input)
            probs = torch.softmax(scores, dim=-1)
            self.set(("action_probs", t), probs)

            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)
        else:
            _input = self.get(self.input_name)
            scores = self.mlp(_input)
            probs = torch.softmax(scores, dim=-1)
            self.set("action_probs", probs)


class A2C_CriticAgent(TAgent):
    def __init__(self, n_input, hidden_size, n_layers, input_name):
        super().__init__()
        sizes = [hidden_size for _ in range(n_layers)]
        self.mlp = mlp([n_input] + sizes + [1], nn.ReLU)
        self.input_name = input_name

    def forward(self, **args):
        _input = self.get(self.input_name)
        c = self.mlp(_input).squeeze(-1)
        self.set("critic", c)


def policy_mlp(encoder, n_layers, hidden_size):
    """One common recurrent NN for policy and critic"""
    encoder = dict(encoder)
    encoder["output_name"] = "policy_input"
    encoder_policy = instantiate_class(encoder)

    policy = A2C_PolicyAgent(
        n_input=encoder_policy.output_size(),
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_actions=8,
        input_name="policy_input",
    )
    return Agents(encoder_policy, policy)


def critic_mlp(encoder, n_layers, hidden_size):
    """One common recurrent NN for policy and critic"""
    encoder = dict(encoder)
    encoder["output_name"] = "critic_input"
    encoder_critic = instantiate_class(encoder)

    critic = A2C_CriticAgent(
        n_input=encoder_critic.output_size(),
        n_layers=n_layers,
        hidden_size=hidden_size,
        input_name="critic_input",
    )
    return Agents(encoder_critic, critic)


def policy_gru(encoder, n_layers, hidden_size):
    encoder = dict(encoder)
    encoder["output_name"] = "policy_input"
    encoder_policy = instantiate_class(encoder)
    gru = GRUAgent(
        n_input=encoder_policy.output_size(),
        n_output=hidden_size,
        obs_name="policy_input",
        z_name="policy_z",
    )
    policy = A2C_PolicyAgent(
        n_input=hidden_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_actions=8,
        input_name="policy_z",
    )
    return Agents(encoder_policy, gru, policy)


def critic_gru(encoder, n_layers, hidden_size):
    encoder = dict(encoder)
    encoder["output_name"] = "critic_input"
    encoder_critic = instantiate_class(encoder)
    gru = GRUAgent(
        n_input=encoder_critic.output_size(),
        n_output=hidden_size,
        obs_name="critic_input",
        z_name="critic_z",
    )
    critic = A2C_CriticAgent(
        n_input=hidden_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        input_name="critic_z",
    )
    return Agents(encoder_critic, gru, critic)
