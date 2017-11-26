"""
Functions that take in layers size and return policy and value network
"""

import torch


def build_value_net(layers):
    """
    :param layers: a list of size 3
        with index 0, 1, 2 specifying the input size, hidden layer size and output size respectively.
    :return: A FFN with 2 hidden layers and RELU non-linearity.
        The returned object has attribute layers, which is the same as the input layers to this function.
    """

    value_net = torch.nn.Sequential(
                  torch.nn.Linear(layers[0], layers[1]),
                  torch.nn.ReLU(),
                  torch.nn.Linear(layers[1], layers[1]),
                  torch.nn.ReLU(),
                  torch.nn.Linear(layers[1], layers[2]),
    )
    value_net.layers = layers
    return value_net


def build_policy_net(layers):
    """
    :param layers: a list of size 3
        with index 0, 1, 2 specifying the input size, hidden layer size and output size respectively.
    :return: a torch.nn.Module which implements a FFN with 1 hidden layer and RELU non-linearity followed by softmax.
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
