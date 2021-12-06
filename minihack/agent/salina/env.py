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
import minihack


def make_env(**args):
    # Create environment instances for actors
    if args["save_tty"]:
        savedir = ""  # NLE choses location
    else:
        savedir = None

    kwargs = dict(
        savedir=savedir,
        archivefile=None,
        observation_keys=args["obs_keys"].split(","),
        penalty_step=args["penalty_step"],
        penalty_time=args["penalty_time"],
        penalty_mode=args["fn_penalty_step"],
    )
    env = gym.make(args["name"], **kwargs)

    return env
