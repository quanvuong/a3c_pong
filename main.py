"""
Implements two main functions:
    main: build the policy and value networks and start the a3c training processes
    start_training_processes: start training processes and wait for them to finish
"""

import argparse
import os
import sys

from constructors import build_policy_net, build_value_net
from optim import SharedRMSProp
from train import train

import torch.multiprocessing as mp


def start_training_processes(args, shared_policy_net, shared_value_net):
    """
    :param args: an object which holds all hyperparam values
    """

    processes = []

    for process_i in range(args.cpu_count):
        arguments = (shared_policy_net,
                     shared_value_net,
                     process_i, args)

        p = mp.Process(target=train, args=arguments)

        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f'pid: {p.pid} exits with exitcode {p.exitcode}')
            sys.stdout.flush()
            sys.exit(1)


def main(args):
    """
    Build the policy and value network, whose parameters are moved to shared memory.
    Start the a3c training processes.

    :param args: an object which holds all hyperparam values
    """

    shared_policy_net = build_policy_net(args).share_memory()
    shared_value_net = build_value_net(args).share_memory()

    start_training_processes(
        args,
        shared_policy_net,
        shared_value_net,
    )


if __name__ == '__main__':
    # args holds all hyper param and game setting as attribute
    args = argparse.ArgumentParser()
    args.lr = 1e-4
    args.entropy_weight = 0.01
    args.env_name = 'Pong-ram-v0'

    # SMALL is used in log(num + SMALL) in case num is 0 to prevent NaN
    args.SMALL = 1e-10

    # gamma is discount rate used to calculate state value
    args.gamma = 0.99

    args.pol_num_hid_layer = 5
    args.val_net_num_hid_layer = 5

    # state size, hidden layer size, output size
    args.policy_net_layers = [128, 1024, 6]
    args.value_net_layers = [128, 1024, 1]

    try:
        cpu_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        cpu_count = 1
        # cpu_count = os.cpu_count() # Uncomment this line to use all available cpus.
    args.cpu_count = cpu_count

    print(f'Using {args.cpu_count} cores')
    sys.stdout.flush()

    main(args)

