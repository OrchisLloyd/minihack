#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from salina import TAgent
from minihack.agent.salina.models.base import BaseNet
from nle.env.base import DUNGEON_SHAPE


class MiniHackObservationAgent(TAgent):
    # Transform obervation+previous action to a vector of size output_size using a MLP

    def __init__(self, output_name, args, device):
        super().__init__()
        self.obs_model = BaseNet(DUNGEON_SHAPE, args, device)
        self.output_name = output_name

    def forward(self, t=None, **args):
        if t is not None:
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
