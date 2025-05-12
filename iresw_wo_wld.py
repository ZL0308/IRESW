import pickle
import gzip

import numpy as np
import torch
import torch.nn as nn
import gym.spaces
import imageio
import matplotlib.pyplot as plt
import wandb

from skimo.rolf.algorithms import BaseAgent
from skimo.rolf.algorithms.dataset import ReplayBufferEpisode, SeqSampler
from skimo.rolf.utils import Logger, Info, StopWatch, LinearDecay
from skimo.rolf.utils.pytorch import optimizer_cuda, count_parameters
from skimo.rolf.utils.pytorch import copy_network, soft_copy_network
from skimo.rolf.utils.pytorch import to_tensor, RandomShiftsAug, AdamAMP
from skimo.rolf.networks.distributions import TanhNormal, mc_kl
from skimo.rolf.networks.dreamer import DenseDecoderTanh, ActionDecoder, Decoder
from skimo.rolf.networks.tdmpc_model import TDMPCModel, Encoder, LSTMEncoder
import torch.nn.functional as F

from iresw_rollout import IreswRolloutRunner


class IreswMetaAgent(BaseAgent):

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._std_decay = LinearDecay(cfg.max_std, cfg.min_std, cfg.std_step)
        self._horizon_decay = LinearDecay(1, cfg.n_skill, cfg.horizon_step)
        self._update_iter = 0
        self._ac_dim = cfg.skill_dim
        self._ob_dim = gym.spaces.flatdim(ob_space)

        self.skill_wld = TDMPCModel(cfg, self._ob_space, cfg.skill_dim, self._dtype)
        self.skill_wld_target = TDMPCModel(cfg, self._ob_space, cfg.skill_dim, self._dtype)
        copy_network(self.skill_wld_target, self.skill_wld)
        self.skill_prior = ActionDecoder(
            cfg.state_dim,
            cfg.skill_dim,
            [cfg.num_units] * cfg.num_layers,
            cfg.dense_act,
            cfg.log_std,
        )
        self.obs_decoder = Decoder(cfg.decoder, cfg.state_dim, self._ob_space)  # used to pretraining the encoder
        self.to(self._device)

    def state_dict(self):
        return {
            "skill_prior": self.skill_prior.state_dict(),
            "skill_wld": self.skill_wld.state_dict(),
            "skill_wld_target": self.skill_wld_target.state_dict(),
            "obs_decoder": self.obs_decoder.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.skill_prior.load_state_dict(ckpt["skill_prior"])
        self.skill_wld.load_state_dict(ckpt["skill_wld"])
        self.skill_wld_target.load_state_dict(ckpt["skill_wld_target"])
        self.obs_decoder.load_state_dict(ckpt["obs_decoder"])
        self.to(self._device)

    @property
    def ac_space(self):
        return self._ac_space

    @torch.no_grad()
    def estimate_value(self, state, ac, horizon):
        value, discount = 0, 1
        for t in range(horizon):
            state, reward = self.skill_wld.imagine_step(state, ac[t])
            value += discount * reward
            discount *= self._cfg.rl_discount
        value += discount * torch.min(*self.skill_wld.critic(state, self.skill_prior.act(state)))
        return value

    @torch.no_grad()
    def act(self, ob, mean=None, is_train=True, warmup=False):
        """Returns skills given an observation `ob`."""
        ob = {k: v[None] for k, v in ob.items()}

        for module in (self.skill_wld, self.skill_prior):
            module.eval()

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = preprocess(to_tensor(ob, self._device, self._dtype), self._cfg.env == "maze")

            if self._cfg.phase == "pretrain" or warmup or not self._cfg.use_cem:
                feat = self.skill_wld.encoder(ob)
                ac = self.skill_prior.act(feat, deterministic=not is_train).squeeze(0)
            else:
                ac, mean = self.plan(ob, mean, is_train)

            flat_ac = ac.cpu().numpy()
            ac = gym.spaces.unflatten(self._ac_space, flat_ac)

        for module in (self.skill_wld, self.skill_prior):
            module.train()

        return ac, mean

    @torch.no_grad()
    def plan(self, ob, prev_mean=None, is_train=True):
        """Plan given an observation `ob` using trajectory-level CEM over skill sequences."""
        cfg = self._cfg
        # 1. Determine horizon and encode observation
        h = int(self._horizon_decay(self._step))
        s = self.skill_wld.encoder(ob)  # [state_dim]

        # 2. Prepare trajectory counts
        num_policy = cfg.num_policy_traj
        num_sample = cfg.num_sample_traj
        num_tot = num_policy + num_sample

        # 3. Generate policy skill trajectories
        z = s.repeat(num_policy, 1)  # [num_policy, state_dim]
        policy_skills = []  # list of [num_policy, ac_dim]
        for t in range(h):
            skill_t = self.skill_prior.act(z)
            policy_skills.append(skill_t)
            z, _ = self.skill_wld.imagine_step(z, skill_t)
        policy_skills = torch.stack(policy_skills, dim=0)  # [h, num_policy, ac_dim]

        # 4. Initialize CEM parameters
        mean = torch.zeros(h, self._ac_dim, device=s.device)
        std = torch.full_like(mean, 2.0)
        if prev_mean is not None and h == prev_mean.size(0) > 1:
            mean[:-1].copy_(prev_mean[1:])

        # 5. Preallocate sample skill tensor
        sample_skill = torch.empty(h, num_sample, self._ac_dim, device=s.device)

        # Repeat state for all trajectories
        z0 = s.repeat(num_tot, 1)  # [num_tot, state_dim]

        # 6. CEM optimization (trajectory-level)
        for _ in range(cfg.cem_iter):
            # 6.1 Sample candidate trajectories (preserve time-correlation)
            sample_skill.normal_()
            sample_skill.mul_(std.unsqueeze(1)).add_(mean.unsqueeze(1))
            sample_skill.clamp_(-0.999, 0.999)

            # 6.2 Concatenate policy and sampled trajectories: [h, num_tot, ac_dim]
            skills = torch.cat([sample_skill, policy_skills], dim=1)

            # 6.3 Evaluate full-trajectory returns (per trajectory)
            # estimate_value returns shape [num_tot, 1] after squeeze
            returns = self.estimate_value(z0, skills, h).squeeze(-1)  # [num_tot]

            # 6.4 Select elite trajectories (by total return)
            elite_val, idxs = returns.topk(cfg.num_elites, largest=True)
            elite_skills = skills[:, idxs, :]  # [h, num_elites, ac_dim]

            # 6.5 Compute normalized weights over elites
            shifted = elite_val - elite_val.max()
            weights = torch.softmax(cfg.cem_temperature * shifted, dim=0)  # [num_elites]
            w = weights.view(1, -1, 1)  # [1, E, 1]

            # 6.6 Update mean and std across time dimension
            new_mean = (w * elite_skills).sum(dim=1)  # [h, ac_dim]
            new_std = torch.sqrt((w * (elite_skills - new_mean.unsqueeze(1)) ** 2).sum(dim=1))
            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
            std = torch.clamp(new_std, self._std_decay(self._step), 2.0)

        # 7. Sample final action from first-step elites
        choice = np.random.choice(cfg.num_elites, p=weights.cpu().numpy())
        selected = elite_skills[0, choice]
        if is_train:
            selected = selected + std[0] * torch.randn_like(std[0])

        return selected.clamp(-0.999, 0.999), mean


class IreswSkillAgent(BaseAgent):

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._update_iter = 0

        self._ac_dim = ac_dim = gym.spaces.flatdim(self._ac_space)
        hidden_dims = [cfg.num_units] * cfg.num_layers
        self.encoder = Encoder(cfg.encoder, ob_space, cfg.state_dim)

        if cfg.lstm:
            skill_input_dim = ac_dim + cfg.state_dim
            self.skill_encoder = LSTMEncoder(
                skill_input_dim, cfg.skill_dim, cfg.lstm_units, 1, log_std=cfg.log_std
            )
        else:
            skill_input_dim = cfg.skill_horizon * ac_dim + cfg.state_dim
            self.skill_encoder = DenseDecoderTanh(
                skill_input_dim, cfg.skill_dim, hidden_dims, cfg.dense_act, cfg.log_std
            )
        self.skill_decoder = ActionDecoder(
            cfg.state_dim + cfg.skill_dim,
            ac_dim,
            hidden_dims,
            cfg.dense_act,
            cfg.log_std,
        )
        self.to(self._device)

    def state_dict(self):
        return {
            "skill_decoder": self.skill_decoder.state_dict(),
            "encoder": self.encoder.state_dict(),
            "skill_encoder": self.skill_encoder.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.skill_decoder.load_state_dict(ckpt["skill_decoder"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.skill_encoder.load_state_dict(ckpt["skill_encoder"])
        self.to(self._device)

    @property
    def ac_space(self):
        return self._ac_space

    @torch.no_grad()
    def act(self, ob, mean=None, cond=None, is_train=True, warmup=False):
        """Returns action and the skill_prior's activation given an observation `ob`."""
        obs_batch = {k: np.expand_dims(v, axis=0) for k, v in ob.items()}

        for module in (self.action_wld, self.skill_decoder):
            module.eval()

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            obs_tensor = preprocess(to_tensor(obs_batch, self._device, self._dtype), self._cfg.env == "maze")
            feat = self.action_wld.encoder(obs_tensor)
            if cond is not None:
                cond = to_tensor(cond, self._device, self._dtype).unsqueeze(0)
            action_latent = self.skill_decoder.act(
                feat, cond, deterministic=True
            )  # Tensor [1, latent_dim]
            action_np = action_latent.squeeze(0).cpu().numpy()
            action = gym.spaces.unflatten(self._ac_space, action_np)
        for module in (self.action_wld, self.skill_decoder):
            module.train()
        return action, mean


class IreswAgent(BaseAgent):

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._update_iter = 0

        meta_ac_space = gym.spaces.Box(-1, 1, [cfg.skill_dim])
        self.meta_agent = IreswMetaAgent(cfg, ob_space, meta_ac_space)
        self.skill_agent = IreswSkillAgent(cfg, ob_space, ac_space)
        self._aug = RandomShiftsAug()

        # The high level policy is regularized by a skill prior.
        if cfg.phase == "rl":
            self.fixed_skill_prior = IreswMetaAgent(cfg, ob_space, meta_ac_space)
            self.log_alpha = torch.tensor(
                np.log(cfg.alpha_init), device=self._device, requires_grad=True
            )

        self._build_optims()
        self._build_buffers()
        self._log_creation()

        # Load pretrained world_model.
        if cfg.phase == "rl" and cfg.pretrain_ckpt_path is not None:
            Logger.warning(f"Load pretrained checkpoint {cfg.pretrain_ckpt_path}")
            ckpt = torch.load(cfg.pretrain_ckpt_path, map_location=self._device, weights_only=False)
            ckpt = ckpt["agent"]
            ckpt["fixed_skill_prior"] = ckpt["meta_agent"].copy()
            self.load_state_dict(ckpt)

    def get_runner(self, cfg, env, env_eval):
        return IreswRolloutRunner(cfg, env, env_eval, self)

    def _build_buffers(self):
        cfg = self._cfg

        self._horizon = cfg.n_skill * cfg.skill_horizon

        # Per-episode replay buffer.
        buffer_keys = ["ob", "ac", "rew", "done"]
        sampler = SeqSampler(cfg.n_skill, sample_last_more=cfg.sample_last_more)
        self.hl_buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )
        buffer_keys = ["ob", "meta_ac", "ac", "rew", "done"]
        sampler = SeqSampler(self._horizon)
        self.ll_buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )
        self.meta_agent.set_buffer(self.hl_buffer)
        self.skill_agent.set_buffer(self.ll_buffer)

        # Load data for pre-training.
        buffer_keys = ["ob", "ac", "done"]
        sampler = SeqSampler(self._horizon + 1)
        self._pretrain_buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )
        self._pretrain_val_buffer = ReplayBufferEpisode(
            buffer_keys, None, sampler.sample_func_tensor, cfg.precision
        )
        Logger.info("Load pre-training data")
        data = pickle.load(gzip.open(cfg.pretrain.data_path, "rb"))
        data_size = len(data)
        Logger.info(f"Load {data_size} trajectories")
        for i, d in enumerate(data):
            if len(d["obs"]) < len(d["dones"]):
                continue
            if cfg.env == "calvin":
                # Only use the first 21 states of non-floating objects.
                d["obs"] = d["obs"][:, :21]
            new_d = dict(ob=d["obs"], ac=d["actions"], done=d["dones"])
            new_d["done"][-1] = 1.0
            if i < data_size * cfg.pretrain.split.train:
                self._pretrain_buffer.store_episode(new_d, False)
            else:
                self._pretrain_val_buffer.store_episode(new_d, False)

        if cfg.env == "maze":
            self._overlay = imageio.imread("envs/assets/maze_40.png")

    def _log_creation(self):
        Logger.info("Creating a SkiMo agent (TD-MPC)")

    def _build_optims(self):
        cfg = self._cfg
        hl_agent = self.meta_agent
        ll_agent = self.skill_agent
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self.hl_modules = [hl_agent.skill_prior, hl_agent.skill_wld, hl_agent.obs_decoder]
        self.ll_modules = [ll_agent.encoder, ll_agent.skill_decoder, ll_agent.skill_encoder]

        # Optimize the skill dynamics and skills jointly.
        if cfg.joint_training:
            self.joint_modules = self.hl_modules + self.ll_modules
            self.joint_optim = adam_amp(self.joint_modules, cfg.joint_lr)
        else:
            self.hl_model_optim = adam_amp(
                [hl_agent.skill_wld, hl_agent.obs_decoder], cfg.model_lr
            )
            self.hl_actor_optim = adam_amp(hl_agent.skill_prior, cfg.actor_lr)

            self.ll_actor_optim = adam_amp(self.ll_modules, cfg.actor_lr)

        if cfg.phase == "rl":
            actor_modules = [hl_agent.skill_prior]
            model_modules = [hl_agent.skill_wld, hl_agent.obs_decoder]
            if cfg.sac:
                actor_modules += [hl_agent.skill_wld.encoder]
            self.actor_optim = adam_amp(actor_modules, cfg.actor_lr)
            self.model_optim = adam_amp(model_modules, cfg.model_lr)
            self.alpha_optim = adam_amp(self.log_alpha, cfg.alpha_lr)

    def set_step(self, step):
        self.meta_agent.set_step(step)
        self.skill_agent.set_step(step)
        self._step = step

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self.hl_buffer.store_episode(rollouts[0], one_more_ob=True)
        self.ll_buffer.store_episode(rollouts[1], one_more_ob=True)

    def buffer_state_dict(self):
        return dict(
            hl_buffer=self.hl_buffer.state_dict(),
            ll_buffer=self.ll_buffer.state_dict(),
        )

    def load_buffer_state_dict(self, state_dict):
        self.hl_buffer.append_state_dict(state_dict["hl_buffer"])
        self.ll_buffer.append_state_dict(state_dict["ll_buffer"])

    def state_dict(self):
        ret = {
            "meta_agent": self.meta_agent.state_dict(),
            "skill_agent": self.skill_agent.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
        }

        if self._cfg.joint_training:
            ret["joint_optim"] = self.joint_optim.state_dict()
        else:
            ret["hl_model_optim"] = self.hl_model_optim.state_dict()
            ret["hl_actor_optim"] = self.hl_actor_optim.state_dict()
            ret["ll_actor_optim"] = self.ll_actor_optim.state_dict()

        if self._cfg.phase == "rl":
            ret["fixed_skill_prior"] = self.fixed_skill_prior.state_dict()
            ret["actor_optim"] = self.actor_optim.state_dict()
            ret["model_optim"] = self.model_optim.state_dict()
            ret["log_alpha"] = self.log_alpha.cpu().detach().numpy()
            ret["alpha_optim"] = self.alpha_optim.state_dict()
        return ret

    def load_state_dict(self, ckpt):
        self.meta_agent.load_state_dict(ckpt["meta_agent"])
        self.skill_agent.load_state_dict(ckpt["skill_agent"])
        if self._cfg.phase == "rl":
            self.fixed_skill_prior.load_state_dict(ckpt["fixed_skill_prior"])

        if self._cfg.joint_training:
            self.joint_optim.load_state_dict(ckpt["joint_optim"])
            optimizer_cuda(self.joint_optim, self._device)
        else:
            self.hl_model_optim.load_state_dict(ckpt["hl_model_optim"])
            self.hl_actor_optim.load_state_dict(ckpt["hl_actor_optim"])
            self.ll_actor_optim.load_state_dict(ckpt["ll_actor_optim"])
            optimizer_cuda(self.hl_model_optim, self._device)
            optimizer_cuda(self.hl_actor_optim, self._device)
            optimizer_cuda(self.ll_model_optim, self._device)
            optimizer_cuda(self.ll_actor_optim, self._device)

        if "model_optim" in ckpt:
            self.model_optim.load_state_dict(ckpt["model_optim"])
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
            self.log_alpha.data = torch.tensor(ckpt["log_alpha"], device=self._device)
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])
            optimizer_cuda(self.hl_model_optim, self._device)
            optimizer_cuda(self.hl_actor_optim, self._device)
            optimizer_cuda(self.alpha_optim, self._device)
        self.to(self._device)

    def update(self):
        """Sample a batch from the replay buffer and make one world_model update in each iteration."""
        train_info = Info()
        sw_data, sw_train = StopWatch(), StopWatch()
        train_iter = self._cfg.train_iter
        if self.warm_up_training():
            self.warm_up_iter = (
                    self._cfg.warm_up_step * self._cfg.train_iter // self._cfg.train_every
            )
            train_iter += self.warm_up_iter
        for _ in range(train_iter):
            sw_data.start()
            hl_batch = self.hl_buffer.sample(self._cfg.batch_size)
            sw_data.stop()

            sw_train.start()
            hl_train_info = self._update_hl_network(hl_batch)
            train_info.add(hl_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")

        return train_info.get_dict()

    def _update_hl_network(self, batch):
        """Updates skill dynamics world_model and high-level policy."""
        cfg = self._cfg
        info = Info()
        mse = nn.MSELoss(reduction="none")
        scalars = cfg.scalars
        hl_agent = self.meta_agent
        max_kl = cfg.max_divergence

        if cfg.freeze_model:
            hl_agent.skill_wld.encoder.requires_grad_(False)
            hl_agent.skill_wld.dynamics.requires_grad_(False)

        # ob: {k: BxTx`ob_dim[k]`}, ac: BxTx`ac_dim`, rew: BxTx1
        o, ac, rew = batch["ob"], batch["ac"], batch["rew"]
        done = batch["done"]
        o = preprocess(o, self._cfg.env == "maze", aug=self._aug)

        hl_feat = flip(hl_agent.skill_wld.encoder(o))
        # Avoid gradients for the skill prior and world_model target
        with torch.no_grad():
            sp_feat = flip(self.fixed_skill_prior.skill_wld.encoder(o))
            hl_feat_target = flip(hl_agent.skill_wld_target.encoder(o))

        ob = flip(o, cfg.n_skill + 1)
        ac = flip(ac)
        rew = flip(rew)
        done = flip(done)

        with torch.autocast(cfg.device, enabled=self._use_amp):
            # Trains skill dynamics world_model.
            z_next_pred = hl_feat[0]
            rewards = []

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            prior_divs = []
            q_preds = [[], []]
            q_targets = []
            intrinsic_rewards_list = []

            alpha = self.log_alpha.exp().detach()
            for t in range(cfg.n_skill):
                z = z_next_pred
                z_next_pred, reward_pred = hl_agent.skill_wld.imagine_step(z, ac[t])
                if cfg.sac:
                    z = ob[t]["ob"]
                q_pred = hl_agent.skill_wld.critic(z, ac[t])

                with torch.no_grad():
                    # `z` for contrastive learning
                    z_next = hl_feat_target[t + 1]

                    # `z` for `q_target`
                    z_next_q = hl_feat[t + 1]
                    ac_next_dist = hl_agent.skill_prior(z_next_q)
                    ac_next = ac_next_dist.rsample()
                    if cfg.sac:
                        z_next_q = ob[t + 1]["ob"]
                    q_next = torch.min(*hl_agent.skill_wld_target.critic(z_next_q, ac_next))

                    # Skill prior regularization.
                    prior_dist = self.fixed_skill_prior.skill_prior(sp_feat[t + 1])
                    prior_div = torch.clamp(
                        mc_kl(ac_next_dist, prior_dist), -max_kl, max_kl
                    )

                    prior_divs.append(prior_div)
                    if cfg.use_prior and cfg.prior_reg_critic:
                        q_next -= (cfg.fixed_alpha or alpha) * prior_div

                    q_target = rew[t] + (1 - done[t].long()) * cfg.rl_discount * q_next
                rewards.append(reward_pred.detach())
                q_preds[0].append(q_pred[0].detach())
                q_preds[1].append(q_pred[1].detach())
                q_targets.append(q_target)

                rho = scalars.rho ** t
                consistency_loss += rho * mse(z_next_pred, z_next).mean(dim=1)
                reward_loss += rho * mse(reward_pred, rew[t])
                value_loss += rho * (
                        mse(q_pred[0], q_target) + mse(q_pred[1], q_target)
                )

                # Additional reward prediction loss.
                reward_pred = hl_agent.skill_wld.reward(
                    torch.cat([hl_feat[t], ac[t]], dim=-1)
                ).squeeze(-1)
                reward_loss += mse(reward_pred, rew[t])
                # Additional value prediction loss.
                obs = hl_feat[t] if not cfg.sac else ob[t]["ob"]
                q_pred = hl_agent.skill_wld.critic(obs, ac[t])
                value_loss += mse(q_pred[0], q_target) + mse(q_pred[1], q_target)

            # If only using SAC, world_model loss is nothing but the critic loss.
            if cfg.sac:
                consistency_loss *= 0
                reward_loss *= 0

            model_loss = (
                    scalars.consistency * consistency_loss.clamp(max=1e5)
                    + scalars.hl_reward * reward_loss.clamp(max=1e5) * 0.5
                    + scalars.hl_value * value_loss.clamp(max=1e5)
            ).mean()
            model_loss.register_hook(lambda grad: grad * (1 / cfg.n_skill))
        model_grad_norm = self.model_optim.step(model_loss)

        # Trains high-level policy.
        with torch.autocast(cfg.device, enabled=self._use_amp):
            actor_loss = 0
            skill_prior_loss = torch.tensor(0.0, device=self._device)
            alpha = self.log_alpha.exp().detach()
            actor_prior_divs = []
            hl_feat = flip(hl_agent.skill_wld.encoder(o)) if cfg.sac else hl_feat.detach()
            z = hl_feat[0]

            # Computes `actor_loss` based on imagined states, `skill_prior_loss` based on encoded ground-truth states.
            for t in range(cfg.n_skill):
                a, ac_dist = hl_agent.skill_prior.act(z, return_dist=True)
                ac_dist_state = hl_agent.skill_prior(hl_feat[t])
                rho = scalars.rho ** t
                if cfg.sac:
                    z = ob[t]["ob"]

                # Calculate intrinsic reward
                with torch.no_grad():
                    state_error = F.smooth_l1_loss(z, hl_feat_target[t + 1], reduction='none').mean(dim=1)
                    state_transition = torch.norm(hl_feat_target[t + 1] - hl_feat_target[t], p=2, dim=1)
                    intrinsic_reward = rho * (
                            cfg.error_weight * state_error + cfg.transition_weight * state_transition).clamp(
                        max=1e3) * 0.5
                intrinsic_rewards_list.append(intrinsic_reward)

                # Add intrinsic reward to the Q-value estimate from the main network
                q_with_intrinsic = torch.min(*hl_agent.skill_wld.critic(z, a)) + intrinsic_reward
                actor_loss += -rho * q_with_intrinsic

                info["actor_std"] = ac_dist.base_dist.base_dist.scale.mean().item()

                # Skill prior regularization.
                with torch.no_grad():
                    prior_dist = self.fixed_skill_prior.skill_prior(sp_feat[t])
                prior_div = torch.clamp(
                    mc_kl(ac_dist_state, prior_dist), -max_kl, max_kl
                ).mean()
                if cfg.use_prior:
                    skill_prior_loss += rho * (cfg.fixed_alpha or alpha) * prior_div
                info["actor_prior_div"] = prior_div.item()
                actor_prior_divs.append(prior_div)

                z_next_pred, _ = hl_agent.skill_wld.imagine_step(z, a)
                z_next_pred = z_next_pred.detach()
                z = z_next_pred

            skill_prior_loss = (
                (scalars.hl_prior * skill_prior_loss).clamp(-1e5, 1e5).mean()
            )
            if cfg.use_prior:
                skill_prior_loss.register_hook(lambda grad: grad * (1 / cfg.n_skill))
            actor_loss = actor_loss.clamp(-1e5, 1e5).mean()
            actor_loss.register_hook(lambda grad: grad * (1 / cfg.n_skill))
        actor_grad_norm = self.actor_optim.step(actor_loss + skill_prior_loss)

        prior_divs = torch.cat(prior_divs)
        actor_prior_divs = torch.stack(actor_prior_divs)
        # Update alpha.
        if cfg.fixed_alpha is None:
            with torch.autocast(cfg.device, enabled=self._use_amp):
                alpha = self.log_alpha.exp()
                alpha_loss = alpha * (
                        cfg.target_divergence - actor_prior_divs.mean().detach()
                )
            self.alpha_optim.step(alpha_loss)

        # Update world_model target.
        self._update_iter += 1
        if self._update_iter % cfg.target_update_freq == 0:
            soft_copy_network(
                hl_agent.skill_wld_target, hl_agent.skill_wld, cfg.target_update_tau
            )

        # For logging.
        q_targets = torch.cat(q_targets)
        q_preds[0] = torch.cat(q_preds[0])
        q_preds[1] = torch.cat(q_preds[1])
        info["value_target_min"] = q_targets.min().item()
        info["value_target_max"] = q_targets.max().item()
        info["value_target"] = q_targets.mean().item()
        info["value_predicted_min0"] = q_preds[0].min().item()
        info["value_predicted_min1"] = q_preds[1].min().item()
        info["value_predicted0"] = q_preds[0].mean().item()
        info["value_predicted1"] = q_preds[1].mean().item()
        info["value_predicted_max0"] = q_preds[0].max().item()
        info["value_predicted_max1"] = q_preds[1].max().item()
        if cfg.use_prior:
            value_skill_prior = (cfg.fixed_alpha or alpha) * prior_divs
            info["value_skill_prior"] = value_skill_prior.mean().item()
            info["value_skill_prior_min"] = value_skill_prior.min().item()
            info["value_skill_prior_max"] = value_skill_prior.max().item()
        info["skill_prior_loss"] = skill_prior_loss.mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["actor_loss"] = actor_loss.mean().item()
        info["model_loss"] = model_loss.mean().item()
        info["consistency_loss"] = consistency_loss.mean().item()
        info["reward_loss"] = reward_loss.mean().item()
        info["critic_loss"] = value_loss.mean().item()
        info["alpha"] = cfg.fixed_alpha or alpha.item()
        info["alpha_loss"] = alpha_loss.item() if cfg.fixed_alpha is None else 0
        rewards = torch.cat(rewards)
        info["reward_predicted"] = rewards.mean().item()
        info["reward_predicted_min"] = rewards.min().item()
        info["reward_predicted_max"] = rewards.max().item()
        info["reward_gt"] = rew.mean().item()
        info["reward_gt_max"] = rew.max().item()
        info["reward_gt_min"] = rew.min().item()
        if intrinsic_rewards_list:
            info["intrinsic_reward_max"] = torch.cat(intrinsic_rewards_list).max().item()  # 这里用最大值进行监督，因为最大值对模型的影响最大
            info["intrinsic_reward_mean"] = torch.cat(intrinsic_rewards_list).mean().item()

        return info.get_dict()


    def pretrain(self):
        train_info = Info()
        sw_data, sw_train = StopWatch(), StopWatch()
        for _ in range(self._cfg.pretrain.train_iter):
            sw_data.start()
            batch = self._pretrain_buffer.sample(self._cfg.pretrain.batch_size)
            sw_data.stop()

            sw_train.start()
            _train_info = self._pretrain(batch)
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")

        info = train_info.get_dict()
        Logger.info(
            f"[HL] skill_prior loss: {info['hl_actor_loss']:.3f}  world_model loss: {info['hl_model_loss']:.3f}  consistency loss: {info['hl_consistency_loss']:.3f}"
        )
        Logger.info(
            f"[LL] skill_prior loss: {info['ll_actor_loss']:.3f}  kl loss: {info['ll_vae_kl_loss']:.3f}"
        )
        return info

    def pretrain_eval(self):
        batch = self._pretrain_val_buffer.sample(self._cfg.pretrain.batch_size)
        return self._pretrain(batch, is_train=False)

    def _pretrain(self, batch, is_train=True):
        cfg = self._cfg
        B, H, L = cfg.pretrain.batch_size, cfg.skill_horizon, cfg.n_skill
        scalars = cfg.scalars
        hl_agent = self.meta_agent
        ll_agent = self.skill_agent
        info = Info()
        mse = nn.MSELoss(reduction="none")

        ob, ac = batch["ob"], batch["ac"]
        o = dict(ob=ob)
        o = preprocess(o, self._cfg.env == "maze", aug=self._aug)
        if ac.shape[1] == L * H + 1:
            ac = ac[:, :-1, :]  # (B, L×H, ac_dim)

        # pretrian the skill encoder and decoder
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # encode the obs for ll
            ll_embed = ll_agent.action_wld.encoder(o)
            x = ac.view(B, L, -1)  # reshape action，   (B, L, H×ac_dim)
            if cfg.lstm:
                x = torch.cat(
                    [ac.view(B, L, H, -1), ll_embed[:, :-1].view(B, L, H, -1)], dim=-1,
                    # reshape  (B, L, H, ac_dim) #512 10 10 138
                ).view(B, L, -1)
            else:
                x = torch.cat([x, ll_embed[:, :-1:H]], dim=-1)
            z_dist = ll_agent.skill_encoder(x)  # encode the action into skill
            z = z_dist.rsample()  # sample skill

            if cfg.phase == "pretrain":
                z_repeat = z.unsqueeze(-2).expand(-1, -1, H, -1)  # (B, L, H, z_dim)
                ac_pred = ll_agent.skill_decoder.act(
                    ll_embed[:, :-1, :].view(B, L, H, -1), z_repeat, deterministic=True
                    # predict the action by skill and obs
                )
                ll_actor_loss = (
                        scalars.ll_actor * mse(ac_pred, ac.view(ac_pred.shape)).mean()  # behaviour cloning
                )
                vae_kl_div = mc_kl(z_dist, "tanh")
                vae_kl_div_clipped = torch.clamp(
                    vae_kl_div, -cfg.max_divergence, cfg.max_divergence
                ).mean()
                ll_vae_kl_loss = scalars.encoder_kl * vae_kl_div_clipped.mean()
            else:
                ll_vae_kl_loss = ll_actor_loss = vae_kl_div_clipped = torch.tensor(0.0)

            # ===== pretrain the world models =====

            hl_o = dict(ob=o["ob"][:, ::H])
            hl_feat = flip(hl_agent.skill_wld.encoder(hl_o))
            with torch.no_grad():
                hl_feat_target = flip(hl_agent.skill_wld_target.encoder(hl_o))
            hl_ac = flip(z)  # generated by skill encoder

            # update the obs encoder and decoder in high level
            hl_ob_pred = hl_agent.obs_decoder(hl_feat)
            hl_recon_losses = {k: -hl_ob_pred[k].log_prob(flip(v)).mean() for k, v in hl_o.items()}
            hl_recon_loss = sum(hl_recon_losses.values())



            hl_consistency_loss = 0.0

            for t in range(L):

                a = hl_ac[t] if cfg.joint_training else hl_ac[t].detach()
                hl_next_pred, _ = hl_agent.skill_wld.imagine_step(hl_feat[t], a)
                hl_next_target = hl_feat_target[t + 1]
                hl_consistency_loss += (scalars.rho ** t) * mse(hl_next_pred, hl_next_target).mean(dim=1)


            hl_model_loss = scalars.hl_model * hl_recon_loss + scalars.consistency * hl_consistency_loss.clamp(
                max=1e4).mean()
            hl_model_loss.register_hook(lambda grad: grad * (1 / L))

        hl_actor_loss = scalars.hl_actor * mc_kl(
            TanhNormal(
                flip(z_dist.base_dist.base_dist.loc.detach()),
                flip(z_dist.base_dist.base_dist.scale.detach()),
                1,
            ),
            hl_agent.skill_prior(hl_feat[:-1].detach()) if not cfg.joint_training else hl_agent.skill_prior(
                hl_feat[:-1])
        ).mean()

        hl_loss = hl_actor_loss + hl_model_loss
        ll_loss = ll_actor_loss + ll_vae_kl_loss
        info["hl_loss"] = hl_loss.item()
        info["ll_loss"] = ll_loss.item()
        info["hl_actor_loss"] = hl_actor_loss.item()
        info["hl_model_loss"] = hl_model_loss.item()
        info["ll_actor_loss"] = ll_actor_loss.item()
        info["ll_vae_kl_loss"] = ll_vae_kl_loss.item()
        info["vae_kl_div"] = vae_kl_div_clipped.mean().item()
        info["hl_consistency_loss"] = hl_consistency_loss.mean().item()
        info["hl_recon_loss"] = hl_recon_loss.item()
        info["actor_std"] = hl_agent.skill_prior(hl_feat[:-1]).base_dist.base_dist.scale.mean().item()

        if cfg.joint_training:
            joint_grad_norm = self.joint_optim.step(hl_loss + ll_loss)
            info["joint_grad_norm"] = joint_grad_norm.item()
        else:
            hl_model_grad_norm = self.hl_model_optim.step(hl_model_loss)
            hl_actor_grad_norm = self.hl_actor_optim.step(hl_actor_loss)
            ll_actor_grad_norm = self.ll_actor_optim.step(ll_actor_loss)
            info["hl_model_grad_norm"] = hl_model_grad_norm.item()
            info["hl_actor_grad_norm"] = hl_actor_grad_norm.item()
            info["ll_actor_grad_norm"] = ll_actor_grad_norm.item()

        if is_train:
            self._update_iter += 1
            if self._update_iter % cfg.target_update_freq == 0:
                soft_copy_network(
                    hl_agent.skill_wld_target, hl_agent.skill_wld, cfg.target_update_tau
                )
                soft_copy_network(
                    ll_agent.action_wld_target, ll_agent.action_wld, cfg.target_update_tau
                )

        return info.get_dict()

    def _visualize(self, ob_gt, ob_pred):
        """Visualizes the prediction and ground truth states.
        Args:
            ob_gt: `n_vis`x(HxL)x`ob_dim` Ground truth observation.
            ob_pred: `n_vis`x(HxL)x`ob_dim` Predicted observation.
        """
        env_name = self._cfg.env
        n_vis = self._cfg.pretrain.n_vis

        if env_name != "maze":
            return

        extent = (0, 40, 0, 40)
        ob_gt = (ob_gt[:, :, :2] + 0.5) * 40
        ob_gt = np.concatenate([ob_gt[i] for i in range(n_vis)], 0)
        ob_pred = (ob_pred[:, :, :2] + 0.5) * 40
        ob_pred = np.clip(ob_pred, 0, 40)
        ob_pred = np.concatenate([ob_pred[i] for i in range(n_vis)], 0)

        fig, axs = plt.subplots(1, 3, clear=True)
        for ax in axs.reshape(-1):
            ax.imshow(self._overlay, alpha=0.3, extent=extent)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlim(0, 40)
            ax.set_ylim(0, 40)
            ax.axis("off")

        def render(ax, data, cmap, title, legend=True):
            s = ax.scatter(
                40 - data[:, 1], data[:, 0], s=3, c=np.arange(len(data)), cmap=cmap,
            )
            ax.set_title(title)
            if legend:
                cbar = fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)

        render(axs[0], ob_gt, "summer", "Ground truth")
        render(axs[1], ob_pred, "copper", "Predicted Rollout")

        render(axs[2], ob_gt, "summer", "All", False)
        render(axs[2], ob_pred, "copper", "All", False)

        img = wandb.Image(fig)
        plt.close(fig)
        return img


def flip(x, l=None):
    if isinstance(x, dict):
        return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
    else:
        return x.transpose(0, 1)


def preprocess(ob, is_maze, aug=None):
    img_keys = [k for k, v in ob.items() if v.ndim >= 4]
    state_keys = [k for k, v in ob.items() if is_maze and v.ndim < 4]

    for k in img_keys:
        x = ob[k]
        x.div_(255.0).sub_(0.5)  # in-place 归一化
        ob[k] = aug(x) if aug else x  # 只有在有 aug 时才调用，避免每次都做判断

    for k in state_keys:
        x = ob[k]
        shp = x.shape
        flat = x.view(-1, shp[-1])
        flat[:, :2].div_(40.0).sub_(0.5)
        flat[:, 2:].div_(10.0)
        ob[k] = flat.view(shp)

    return ob
