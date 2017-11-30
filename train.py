"""
Implement functions for each training process to update the params of the policy and value net.
The function to start reading from is function train.
"""

import random
from itertools import count
import sys

import numpy as np
import torch
import torch.nn.functional as F
import gym

from constructors import build_policy_net, build_value_net
from wrappers import FloatTensorFromNumpyVar, FloatTensorVar, ZeroTensorVar
from utils import run_episode, run_value_net


def ensure_share_grads(shared_net, local_net):
    for param, shared_param in zip(local_net.parameters(), shared_net.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train_value_net(value_net, shared_value_net, shared_value_optim, episode):
    """
    Update the value network using first visit Monte Carlo return
    to minimize the L1 loss between actual and predicted values of states visited during the epidode.

    :param episode: a list of EpisodeStep (specified in utils.py)
    """

    # Calculate return from the first visit to each state
    visited_states = set()
    states, returns = [], []
    for t in range(len(episode)):
        s, G = episode[t].state, episode[t].G
        str_s = s.astype(int).tostring()  # Fastest hashable state representation
        if str_s not in visited_states:
            visited_states.add(str_s)
            states.append(s)

            # Monte-Carlo return
            returns.append(G)

    states = FloatTensorFromNumpyVar(np.array(states))
    returns = FloatTensorVar([returns])

    # Train the value network on states, returns
    shared_value_optim.zero_grad()

    loss = F.l1_loss(value_net(states), returns)
    loss.backward()

    # Turn NaN to 0
    for w in value_net.parameters():
        w.grad.data[w.grad.data != w.grad.data] = 0

    ensure_share_grads(shared_value_net, value_net)
    shared_value_optim.step()


def train_policy_net(policy_net, shared_policy_net, shared_policy_optim, episode, value_net, args, process_i=-1):
    """
    Update the policy net using policy gradient formulation with entropy bonus.

    :param episode: a list of EpisodeStep (specified in utils.py)
    :param args: an object which holds all hyperparam setting
    """

    # Compute baselines
    baselines = [run_value_net(value_net, step.state) for step in episode]
    baselines = FloatTensorVar(baselines)

    # Calculate log probability of action and episode entropy
    log_act_probs = ZeroTensorVar(len(episode))
    entropy = ZeroTensorVar(1)
    for idx, step in enumerate(episode):
        log_act_probs[idx] = step.act_prob
        entropy = entropy + step.entropy

    returns = FloatTensorVar([step.G for step in episode])

    # if process_i == 0:
    #     print(log_act_probs)
    #     print(baselines)
    #     print(returns)
    #     sys.stdout.flush()

    # Call backward pass and update param
    shared_policy_optim.zero_grad()
    # neg_perf = (log_act_probs * (baselines - returns)).sum() - args.entropy_weight * entropy
    neg_perf = - args.entropy_weight * entropy
    neg_perf = neg_perf / len(episode)
    neg_perf.backward()

    # Turn NaNs to 0
    for w in policy_net.parameters():
        w.grad.data[w.grad.data != w.grad.data] = 0

    ensure_share_grads(shared_policy_net, policy_net)
    shared_policy_optim.step()


def train(shared_policy_net, shared_policy_optim,
          shared_value_net, shared_value_optim, process_i, args):
    """
    Seeds each training process based on its rank.
    Build local version of policy and value network.
    Run the policy in the environment and update the policy and value network using Monte Carlo return.
    Synchronize the params of both policy and value network with the shared policy and value network after every update.

    :param process_i: the rank of this process
    :param args: an object which holds all hyperparam setting
    """

    # Create env
    env = gym.make(args.env_name)

    # Each training process is init with a different seeds
    random.seed(process_i)
    np.random.seed(process_i)
    torch.manual_seed(process_i)
    env.seed(process_i)

    # create local policy and value net and sync params
    policy_net = build_policy_net(args)
    policy_net.load_state_dict(shared_policy_net.state_dict())

    value_net = build_value_net(args)
    value_net.load_state_dict(shared_value_net.state_dict())

    for episode_i in count():
        episode = run_episode(policy_net, env, args, process_i=process_i)

        if process_i == 0:
            print(f'process: {process_i}, episode: {episode_i}, episode length: {len(episode)}, G: {episode[0].G}')
            sys.stdout.flush()

        train_value_net(value_net, shared_value_net, shared_value_optim, episode)
        value_net.load_state_dict(shared_value_net.state_dict())

        train_policy_net(policy_net, shared_policy_net, shared_policy_optim,
                         episode, value_net, args, process_i=process_i)
        policy_net.load_state_dict(shared_policy_net.state_dict())
