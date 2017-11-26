from wrappers import FloatTensorFromNumpyVar

import numpy as np
from namedlist import namedlist
import torch

EpisodeStep = namedlist('EpisodeStep', 'state act act_prob reward entropy G', default=0)


def np_to_torch_state(state):
    # Reshape state size and type
    state = np.expand_dims(state, 0)
    state = FloatTensorFromNumpyVar(state)

    return state


def run_episode(policy_net, env, args):
    """
    Run the policy net in the environment for one episode.
    Calculate the discounted value of each states visited.
    Cache the value of the entropy bonus.

    :param env: an environment conforming to openAI gym interface
    :param args: an object which holds all hyperparam settings
    :return: a list of EpisodeStep
    """

    episode = []
    state = env.reset()

    while True:
        # Pick action
        state_torch = np_to_torch_state(state)
        act_dist = policy_net(state_torch)
        act = torch.multinomial(act_dist.data, 1)[0, 0]

        # Calculate entropy bonus
        entropy = (- act_dist * torch.log(act_dist + args.SMALL)).sum()

        # Take action
        new_state, reward, done, _ = env.step(act)

        # Record step information
        episode.append(EpisodeStep(
            state=state,
            act=act,
            act_prob=act_dist[0, act],
            reward=reward,
            entropy=entropy
        ))

        if done:
            break
        else:
            state = new_state

    # Calculate discounted state value
    for i, step in enumerate(reversed(episode)):
        if i == 0:
            step.G = step.reward
        else:
            step.G = step.reward + args.gamma*episode[len(episode)-i].G

    return episode


def run_value_net(value_net, state):
    result = value_net(FloatTensorFromNumpyVar(np.array([state])))
    return result.data[0][0]
