import torch.optim as optim
from models.optim.adamw import AdamW


def build_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    elif args.optim == 'adam':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
    return optimizer

