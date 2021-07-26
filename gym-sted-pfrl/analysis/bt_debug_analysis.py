import argparse
import numpy

import gym
import gym.spaces
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.policies import GaussianHeadWithFixedCovariance, SoftmaxCategoricalHead

# debug import
from matplotlib import pyplot as plt


"""
Mon but ici c'est de loader l'agent que j'ai entrainé avec la normalisation et qui semblait bien performer et de voir
les actions (donc puissances de STED) qu'il choisit
"""

"""
def calc_shape(shape, layers):
    _shape = numpy.array(shape[1:])
    for layer in layers:
        _shape = (_shape + 2 * numpy.array(layer.padding) - numpy.array(layer.dilation) * (
                    numpy.array(layer.kernel_size) - 1) - 1) / numpy.array(layer.stride) + 1
        _shape = _shape.astype(int)
    return (shape[0], *_shape)


class Policy(nn.Module):
    def __init__(
            self, in_channels=1, action_size=1, obs_size=1152,
            activation=nn.functional.leaky_relu
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_size = obs_size
        self.activation = activation
        super(Policy, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        out_shape = calc_shape(obs_size, self.layers)
        self.linear = nn.Linear(32 * numpy.prod(out_shape), action_size)
        self.policy = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.policy(x)
        return x


class ValueFunction(nn.Module):
    def __init__(
            self, in_channels=1, action_size=1, obs_size=1152,
            activation=torch.tanh
    ):
        self.in_channels = in_channels
        self.action_size = action_size
        self.obs_size = obs_size
        self.activation = activation
        super(ValueFunction, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.Conv2d(16, 32, 4, stride=2),
        ])
        out_shape = calc_shape(obs_size, self.layers)
        self.linears = nn.ModuleList([
            nn.Linear(32 * numpy.prod(out_shape), 64),
            nn.Linear(64, 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.view(x.size(0), -1)
        for layer in self.linears:
            x = self.activation(layer(x))
        return x


def main():
    # this main uses the fully trained agent
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:TimedSTEDdebugBleach-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--exp_id", type=str, default="debug")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10 ** 5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-n-runs", type=int, default=5)
    parser.add_argument("--reward-scale-factor", type=float, default=1.)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    # logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # args.outdir = experiments.prepare_output_dir(args, args.outdir, exp_id=args.exp_id, make_backup=not args.dry_run)

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    train_env = make_env(test=False)
    timestep_limit = train_env.spec.max_episode_steps
    obs_space = train_env.observation_space
    action_space = train_env.action_space

    obs_size = obs_space.shape
    policy = Policy(obs_size=obs_size)
    vf = ValueFunction(obs_size=obs_size)
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    agent = pfrl.agents.PPO(
        model,
        opt,
        gpu=args.gpu,
        minibatch_size=args.batchsize,
        max_grad_norm=1.0,
        update_interval=100
    )
    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            eval_env=eval_env,
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
        )
"""

# """
def main():
    # in this main I want to manually select the max sted power every time and see what happens :)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_sted:TimedSTEDdebugBleach-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1.)
    parser.add_argument("--render", action="store_true", default=False)
    args = parser.parse_args()

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    train_env = make_env(test=False)
    eval_env = make_env(test=True)

    n_runs = 50
    n_steps = 13
    # episode_rewards_array = numpy.zeros((n_runs, n_steps))
    episode_info = numpy.zeros((n_runs, n_steps, 3))
    earned_rewards = []
    for i in range(n_runs):
        print(f"starting run {i+1} of {n_runs}")
        eval_env.reset()
        for j in range(n_steps):
            n_molecs_init = eval_env.temporal_datamap.base_datamap.sum()
            _, _, _, info = eval_env.step(1)
            n_molecs_post = eval_env.temporal_datamap.base_datamap.sum()
            episode_info[i, j, 0] = info["sted_power"]
            episode_info[i, j, 1] = n_molecs_init
            episode_info[i, j, 2] = n_molecs_post

    # rewards = (episode_info[:, :, 1] - episode_info[:, :, 2]) / episode_info[:, :, 1]
    # rewards = numpy.mean(rewards, axis=0)
    # plt.plot(rewards)
    # plt.show()

    # save_path = "gym-sted-pfrl/analysis"
    # numpy.save(save_path + "/max_power_spam_agent_bug_fix", episode_info)
    numpy.save("max_power_spam_agent_bug_fix_v2_yo", episode_info)

    # eval_env.reset()
    # n_steps = 13
    # rewards = []
    # for i in range(n_steps):
    #     print(f"step {i}")
    #     # print(f"flash_tstep b4 step = {eval_env.temporal_experiment.flash_tstep}")
    #     # -1 corresponds to a STED power of 0, 1 corresponds to a STED power of 5e-3
    #     # lower/higher than -1/1 clips to -1/1
    #     _, _, _, info = eval_env.step(0)
    #     reward = (info["n molecules init"] - info["n molecules post"]) / info["n molecules init"]
    #     rewards.append(reward)
    #     # print(f"flash_tstep after step = {eval_env.temporal_experiment.flash_tstep}")
    #     # print(info["sted_power"])
    #     # plt.imshow(info["sted_image"])
    #     # plt.title(f"step {i}")
    #     # plt.show()
    # plt.plot(rewards)
    # plt.show()

# """

if __name__ == "__main__":
    main()