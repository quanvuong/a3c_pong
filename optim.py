"""
Implement RMSProp with shared statistics among all the training processes
"""

import torch
from torch.optim import RMSprop


class SharedRMSProp(RMSprop):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        # State initialization
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = torch.zeros(1)
                # Get grad of param variable
                grad = param.data
                state['square_avg'] = grad.new().resize_as_(grad).zero_()
                state['grad_avg'] = grad.new().resize_as_(grad).zero_()

        self.share_memory()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                state['grad_avg'].share_memory_()

    def step(self, closure=None):

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                state = self.state[param]

                # Incre step counter
                state['step'] += 1

                # Receive necessary info
                grad = param.grad.data
                square_avg = state['square_avg']
                alpha = group['alpha']

                # Compute avg
                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])

                # Update param
                param.data.addcdiv_(-group['lr'], grad, avg)