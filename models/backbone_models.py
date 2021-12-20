from .resnetFPN import resnetfpn
import torch
import pdb

def backbone_models(args):

    assert args.ARCH.startswith('resnet')

    modelperms = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}
    model_3d_layers = {'resnet50': [[0, 1, 2], [0, 2], [0, 2, 4], [0, 1]], 
                       'resnet101': [[0, 1, 2], [0, 2], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [0, 1]]}
    assert args.ARCH in modelperms, 'Arch shoudl from::>' + \
        ','.join([m for m in modelperms])

    if args.MODEL_TYPE.endswith('-NL'):
        args.non_local_inds = [[], [1, 3], [1, 3, 5], []]
    else:
        args.non_local_inds = [[], [], [], []]

    base_arch, MODEL_TYPE = args.ARCH, args.MODEL_TYPE
    perms = modelperms[base_arch]

    args.model_perms = modelperms[base_arch]
    args.model_3d_layers = model_3d_layers[base_arch]

    model = resnetfpn(args)

    if args.MODE == 'train':
        if MODEL_TYPE.startswith('RCN'):
            model.identity_state_dict()
        if MODEL_TYPE.startswith('RCGRU') or MODEL_TYPE.startswith('RCLSTM'):
            model.recurrent_conv_zero_state()
        if not MODEL_TYPE.startswith('SlowFast'):
            load_dict = torch.load(args.MODEL_PATH)
            model.load_my_state_dict(load_dict)

    return model
