# Adapted from the Diffusion-QL project
# Original source: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
# SPDX-License-Identifier: Apache-2.0
# Original Copyright 2022 Twitter, Inc and Zhendong Wang
# 
# Modifications made by Suzan Ece Ada, 2024
# Note: These modifications include restrictions on commercial use. 
# For details, see the LICENSE file in this repository.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents_ae.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_a_layer = nn.Linear(256, action_dim)
        self.final_s_layer = nn.Linear(256, state_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)


class MLP_AE(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_AE, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())
        self.final_a_layer = nn.Sequential(nn.Linear(256, 256),
                                           nn.Mish(), nn.Linear(256, action_dim))
        self.final_s_layer = nn.Sequential(nn.Linear(256, 256),
                                           nn.Mish(), nn.Linear(256, state_dim))

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)


class MLP_P5(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_P5, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 128),
                                       nn.Mish(),
                                       nn.Linear(128, 256),
                                       nn.Mish())

        self.final_a_layer = nn.Linear(256, action_dim)
        self.final_s_layer = nn.Linear(256, state_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)

class MLP2(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP2, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 512),
                                       nn.Mish(),
                                       nn.Linear(512, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 512),
                                       nn.Mish())

        self.final_a_layer = nn.Linear(512, action_dim)
        self.final_s_layer = nn.Linear(512, state_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)


class MLP_AE2(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_AE2, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 512),
                                       nn.Mish(),
                                       nn.Linear(512, 256),
                                       nn.Mish())

        self.final_a_layer = nn.Sequential(nn.Linear(256, 512),
                                           nn.Mish(), nn.Linear(512, action_dim))
        self.final_s_layer = nn.Sequential(nn.Linear(256,512),
                                           nn.Mish(), nn.Linear(512, state_dim))

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)

class MLP_AE_P5(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP_AE_P5, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 128),
                                       nn.Mish())

        self.final_a_layer = nn.Sequential(nn.Linear(128, 256),
                                           nn.Mish(), nn.Linear(256, action_dim))
        self.final_s_layer = nn.Sequential(nn.Linear(128,256),
                                           nn.Mish(), nn.Linear(256, state_dim))

    def forward(self, x, time, state):

        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_a_layer(x), self.final_s_layer(x)