import argparse
import os
import sys

from constructors import build_policy_net, build_value_net
from optim import SharedRMSProp
from train import train

import torch.multiprocessing as mp


def start_training_processes(args, shared_policy_net, shared_policy_optim,
                             shared_value_net, shared_value_optim):

    processes = []
    #
    # train(shared_policy_net, shared_policy_optim,
    #       shared_value_net, shared_value_optim, 0, args)

    for process_i in range(args.cpu_count+1):
        arguments = (shared_policy_net, shared_policy_optim,
                     shared_value_net, shared_value_optim,
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

    # state size, hidden layer size, output size
    policy_net_layers = [128, 256, 6]
    value_net_layers = [128, 256, 1]

    shared_policy_net = build_policy_net(policy_net_layers).share_memory()
    shared_value_net = build_value_net(value_net_layers).share_memory()

    shared_policy_optim = SharedRMSProp(shared_policy_net.parameters(), lr=args.lr)
    shared_value_optim = SharedRMSProp(shared_value_net.parameters(), lr=args.lr)

    start_training_processes(
        args,
        shared_policy_net, shared_policy_optim,
        shared_value_net, shared_value_optim
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

    try:
        cpu_count = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        # cpu_count = os.cpu_count()
        cpu_count = 1
    args.cpu_count = cpu_count

    print(f'Using {args.cpu_count} number of cores')

    main(args)

