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

from salina import TAgent
from minihack.agent.salina.model import BaseNet

import torch
from nle.env.base import DUNGEON_SHAPE


class MiniHackObservationAgent(TAgent):
    # Transform obervation+previous action to a vector of size output_size using a MLP

    def __init__(self, output_name, args):
        super().__init__()
        # self.obs_model = ObservationToVector(use_tty_only=use_tty_only)
        self.obs_model = BaseNet(DUNGEON_SHAPE, args)
        self.output_name = output_name

    def forward(self, t=None, **args):
        if not t is None:
            d = {
                k[4:]: self.workspace.get(k, t)
                for k in self.workspace.keys()
                if k.startswith("env/")
            }
            vector = self.obs_model(d)
            self.set((self.output_name, t), vector)
        else:
            for t in range(self.workspace.time_size()):
                self.forward(t=t, **args)

    def output_size(self):
        return 256
