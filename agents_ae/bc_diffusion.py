# Adapted from the Diffusion-QL project
# Original source: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
# SPDX-License-Identifier: Apache-2.0
# Original Copyright 2022 Twitter, Inc and Zhendong Wang
# 
# Modifications made by Suzan Ece Ada, 2024
# Note: These modifications include restrictions on commercial use. 
# For details, see the LICENSE file in this repository.

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from agents_ae.diffusion import Diffusion
from agents_ae.model import MLP
from agents_ae.model import MLP_AE
from agents_ae.model import MLP_P5
from agents_ae.model import MLP2
from agents_ae.model import MLP_AE2
from agents_ae.model import MLP_AE_P5


class Diffusion_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=2e-4,
                 model_type='MLP',
                 ):
        if model_type == 'MLP_AE':
            self.model = MLP_AE(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'MLP2':
            self.model = MLP2(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'MLP_AE2':
            self.model = MLP_AE2(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'MLP_P5':
            self.model = MLP_P5(state_dim=state_dim, action_dim=action_dim, device=device)
        elif model_type == 'MLP_AE_P5':
            self.model = MLP_AE_P5(state_dim=state_dim, action_dim=action_dim, device=device)
        else:
            self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
            print('model_type not found, using MLP')

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            loss_a,loss_s  = self.actor.loss(action, state)
            loss = loss_a + loss_s
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['actor_loss'].append(0.)
            metric['bc_loss'].append(loss.item())
            metric['ql_loss'].append(0.)
            metric['critic_loss'].append(0.)

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))

