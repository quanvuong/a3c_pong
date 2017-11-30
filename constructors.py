"""
Functions that take in layers size and return policy and value network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_value_net(args, layers):
    """
    :param layers: a list of size 3
        with index 0, 1, 2 specifying the input size, hidden layer size and output size respectively.
    :return: a FFN with 2 hidden layers and RELU non-linearity.
        The returned object has attribute layers, which is the same as the input layers to this function.
    """

    class ValueNet(nn.Module):
        def __init__(self, args, layers):
            super().__init__()
            self.input_linear = nn.Linear(layers[0], layers[1])
            self.output_linear = nn.Linear(layers[1], layers[2])
            self.hid_linears = nn.ModuleList([nn.Linear(layers[1], layers[1])
                                              for _ in range(args.val_net_num_hid_layer)])
            self.layers = layers

        def forward(self, state):

            x = F.relu(self.input_linear(state))

            for layer in self.hid_linears:
                x = F.relu(layer(x))

            output = self.output_linear(x)
            return output

    return ValueNet(args, layers)


def build_policy_net(layers):
    """
    :param layers: a list of size 3
        with index 0, 1, 2 specifying the input size, hidden layer size and output size respectively.
    :return: a FFN with 1 hidden layer and RELU non-linearity followed by softmax.
        The returned object has attribute layers, which is the same as the input layers to this function.
    """

    class PolicyNet(torch.nn.Module):
        def __init__(self, layers):
            super(PolicyNet, self).__init__()
            self.linear1 = torch.nn.Linear(layers[0], layers[1])
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(layers[1], layers[2])
            self.sf = torch.nn.Softmax()
            self.layers = layers

        def forward(self, state):
            before_sf = self.linear2(self.relu(self.linear1(state)))
            dist = self.sf(before_sf)
            return dist

    return PolicyNet(layers)
