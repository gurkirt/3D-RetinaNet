import torch, pdb
import torch.optim as optim
# from .madamw import Adam as AdamM
# from .adamw import Adam as AdamW
# from torch.optim.lr_scheduler import MultiStepLR

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, MILESTONES, GAMMAS, last_epoch=-1):
        self.MILESTONES = MILESTONES
        self.GAMMAS = GAMMAS
        assert len(GAMMAS) == len(MILESTONES), 'MILESTONES and GAMMAS should be of same length GAMMAS are of len ' + (len(GAMMAS)) + ' and MILESTONES '+ str(len(MILESTONES))
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.MILESTONES:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            index = self.MILESTONES.index(self.last_epoch)
            return [group['lr'] * self.GAMMAS[index] for group in self.optimizer.param_groups]
    
    def print_lr(self):
        print([[group['name'], group['lr']] for group in self.optimizer.param_groups])

def get_optim(args, net):
    freeze_layers = ['backbone_net.layer'+str(n) for n in range(1, args.freezeupto+1)]
    params = []
    solver_print_str = '\n\nSolver configs are as follow \n\n\n'
    for key, value in net.named_parameters():
        
        if args.freezeupto>0 and (key.find('backbone_net.conv1')>-1 or key.find('backbone_net.bn1')>-1): # Freeze first conv layer and bn layer in resnet
            value.requires_grad = False
            continue
        
        if key.find('backbone_net')>-1: 
            for layer_id in freeze_layers:
                if key.find(layer_id)>-1:
                    value.requires_grad = False    
                    continue
        
        if not value.requires_grad:
            continue
        
        lr = args.lr
        wd = args.weight_decay
        
        if args.optim == 'ADAM':
            wd = 0.0
        
        if "bias" in key:
            lr = lr*2.0
        
        if args.optim == 'SGD':
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd, "momentum":args.momentum}]
        else:
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd}]
        
        print_l = key +' is trained at the rate of ' + str(lr)
        print(print_l)
        solver_print_str += print_l + '\n'
        
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(params)
    elif args.optim == 'ADAM':
        optimizer = optim.Adam(params)
    else:
        raise NotImplementedError('Define optimiser type')
    
    solver_print_str += 'optimizer is '+ args.optim + '\nDone solver configs\n\n'

    scheduler = WarmupMultiStepLR(optimizer, args.MILESTONES, args.GAMMAS)

    return optimizer, scheduler, solver_print_str
