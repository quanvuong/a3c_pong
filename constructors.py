"""
Functions that take in layers size and return policy and value network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_value_net(args):
    """
    :param args: an object which holds all hyperparam setting
    :return: a FFN with RELU non-linearity.
    """

    class ValueNet(nn.Module):
        def __init__(self, args):
            super().__init__()
            layers = args.value_net_layers

            self.input_linear = nn.Linear(layers[0], layers[1])
            self.output_linear = nn.Linear(layers[1], layers[2])
            self.hid_linears = nn.ModuleList([nn.Linear(layers[1], layers[1])
                                              for _ in range(args.val_net_num_hid_layer)])

        def forward(self, state):

            x = F.relu(self.input_linear(state))

            for layer in self.hid_linears:
                x = F.relu(layer(x))

            output = self.output_linear(x)
            return output

    return ValueNet(args)


def build_policy_net(args):
    """
    :param args: an object which holds all hyperparam setting
    :return: a FFN with RELU non-linearity followed by softmax.
    """

    class PolicyNet(torch.nn.Module):
        def __init__(self, args):
            super(PolicyNet, self).__init__()
            layers = args.policy_net_layers

            self.input_linear = nn.Linear(layers[0], layers[1])
            self.output_linear = nn.Linear(layers[1], layers[2])
            self.hid_linears = nn.ModuleList([nn.Linear(layers[1], layers[1])
                                              for _ in range(args.pol_num_hid_layer)])

        def forward(self, state):
            x = F.relu(self.input_linear(state))

            for layer in self.hid_linears:
                x = F.relu(layer(x))

            x = self.output_linear(x)

            return F.softmax(x)

    return PolicyNet(args)
