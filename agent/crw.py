import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.ddpg import DDPGAgent


class CRW(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 crw_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.,
                 temperature=1):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.crw = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, crw_rep_dim))
        self.temperature=temperature
        self.weight_init()

    def weight_init(self):
        self.apply(utils.weight_init)

    def forward(self, state, next_state):
        state = self.aug(state)
        state = self.normalize_obs(state)
        state = torch.clamp(state, -self.clip_val, self.clip_val)
        state = self.crw(state)

        next_state = self.aug(next_state)
        next_state = self.normalize_obs(next_state)
        next_state = torch.clamp(next_state, -self.clip_val, self.clip_val)
        next_state = self.crw(next_state)


        B,N = state.shape
        A12 = F.softmax(torch.bmm(state.reshape(B,N,1),next_state.reshape(B,1,N))/self.temperature,dim=-1)
        A21 = A12.transpose(dim0=-1,dim1=-2)
        As =  torch.bmm(A12,A21)
        logits=torch.diagonal(As,dim1=-1,dim2=-2)
        intrinsic_reward = -torch.log(logits).sum()
        return intrinsic_reward


class CRWAgent(DDPGAgent):
    def __init__(self, crw_rep_dim, update_encoder, crw_scale=1., **kwargs):
        super().__init__(**kwargs)
        self.crw_scale = crw_scale
        self.update_encoder = update_encoder

        self.crw = CRW(self.obs_dim, self.hidden_dim, crw_rep_dim,
                       self.encoder, self.aug, self.obs_shape,
                       self.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        # optimizers
        self.crw_opt = torch.optim.Adam(self.crw.parameters(), lr=self.lr)
        self.running_loss=0
        self.crw.train()

    def update_crw(self, obs, next, step):
        if(self.running_loss<1e-3):
            print("Init")
            self.crw.weight_init()
            
        metrics = dict()

        loss = self.crw(obs,next).mean()

        self.crw_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.crw_opt.step()

        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['crw_loss'] = loss.item()

        if self.running_loss<1e-3:
            self.running_loss=loss
        else:
            self.running_loss=self.running_loss*0.75+ loss*0.25

        return metrics

    def compute_intr_reward(self, obs, next_obs, step):
        prediction_error = torch.clamp(self.crw(obs, next_obs), min=-10,max=10)
        reward = self.crw_scale * prediction_error / (torch.sqrt(prediction_error) + 1e-8)
        return reward

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_crw(obs, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # update critic
        metrics.update(  self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
