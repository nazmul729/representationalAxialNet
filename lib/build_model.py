from . import models

def build_model(args):
    if args.experimentType == "quaternion":
        print("Start quaternion here")
    else:
        model = models.__dict__[args.model](num_classes=args.num_classes)
    
    #if args.model == "axial35s":
    #       model = axialnet.axial35s(args.num_classes)
    #  else:
      
    return model
