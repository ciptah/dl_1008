"""
Module that do stochastic gradient descent optionally with momentum method or Nesterov's Accelerated Gradient method 
"""
from torch.optim.optimizer import Optimizer, required

class SGD_Mmtm(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:

        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=0, NAG=False, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, NAG=NAG, dampening=dampening, weight_decay=weight_decay)
        super(SGD_Mmtm, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            clousre (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            NAG = group['NAG']
            
            for p in group['params']:
                d_p = p.grad.data
                
        return loss
