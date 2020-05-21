
import os
import torch
import argparse
from modules import utils
from train import train_main
from data import Read
from torchvision import transforms
from data.transforms import Resize
from models.retinanet_shared_heads import build_retinanet_shared_heads


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    # Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
    parser.add_argument('--mode', default='train', help='mode can be train or test define subsets accordingly')
    parser.add_argument('--basenet', default='resnet50', help='pretrained base model')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    # parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--shared_heads', default=0, type=int,help='4 head layers')
    parser.add_argument('--num_head_layers', default=4, type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
    #  Name of the dataset only voc or coco are supported
    parser.add_argument('--dataset', default='aarav', help='pretrained base model')
    parser.add_argument('--train_subsets', default='train_3,', type=str,help='Training subsets seprated by ,')
    parser.add_argument('--val_subsets', default='val_3,', type=str,help='Validation subsets seprated by ,')
    parser.add_argument('--test_subsets', default='test,', type=str,help='Testing subsets seprated by ,')
    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--min_size', default=600, type=int, help='Input Size for FPN')
    parser.add_argument('--max_size', default=1000, type=int, help='Input Size for FPN')
    #  data loading argumnets
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
    # Number of worker to load data in parllel
    parser.add_argument('--num_workers', '-j', default=0, type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')
    parser.add_argument('--resume', default=0, type=int, help='Resume from given iterations')
    parser.add_argument('--max_iter', default=25000, type=int, help='Number of training iterations')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--loss_type', default='focal', type=str, help='loss_type')
    parser.add_argument('--milestones', default='10000,20000', type=str, help='Chnage the lr @')
    parser.add_argument('--gammas', default='0.1,0.1', type=str, help='Gamma update for SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')

    # Freeze layers or not 
    parser.add_argument('--fbn','--freeze_bn', default=True, type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
    parser.add_argument('--freezeupto', default=1, type=int, help='layer group number in ResNet up to which needs to be frozen')

    # Loss function matching threshold
    parser.add_argument('--positive_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    parser.add_argument('--negative_threshold', default=0.4, type=float, help='Min Jaccard index for matching')

    # Evaluation hyperparameters
    parser.add_argument('--save_detections', default=False, type=str2bool, help='eval iterations')
    parser.add_argument('--eval_iters', default='90000', type=str, help='eval iterations')
    parser.add_argument('--intial_val', default=5000, type=int, help='Initial number of training iterations before evaluation')
    parser.add_argument('--val_step', default=25000, type=int, help='Number of training iterations before evaluation')
    parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
    parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
    parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int, help='topk for evaluation')

    # Progress logging
    parser.add_argument('--log_start', default=149, type=int, help='start loging after k steps for text/tensorboard') # Let initial ripples settle down
    parser.add_argument('--log_step', default=10, type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--tensorboard', default=False, type=str2bool, help='Use tensorboard for loss/evalaution visualization')

    # Program arguments
    parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
    parser.add_argument('--multi_gpu', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

    # Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use
    parser.add_argument('--data_root', default='/mnt/mercury-fast/datasets/', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
    parser.add_argument('--save_root', default='/mnt/mercury-fast/datasets/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
    parser.add_argument('--model_dir', default='', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/

    ## Parse arguments
    args = parser.parse_args()

    args = utils.set_args(args) # set directories and subsets fo datasets

    ## set random seeds and global settings
    np.random.seed(args.man_seed)
    torch.manual_seed(args.man_seed)
    torch.cuda.manual_seed_all(args.man_seed)
    torch.set_default_tensor_type('torch.FloatTensor')

    args.exp_name = utils.create_exp_name(args)
    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.save_root): #if save directory doesn't exist create it
        os.makedirs(args.save_root)


    if args.mode == 'train':
        args.subsets = args.train_subsets
        train_transform = transforms.Compose([
                            #transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.05),
                            Resize(args.min_size, args.max_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=args.means, std=args.stds)])

        train_dataset = Read(args, train=True, subsets transform=train_transform)
        print('Done Loading Dataset Train Dataset :::>>>\n',train_dataset.print_str)
        ## For validation set
        full_test = False
        args.subsets = args.train_subsets
        skip_step = 3
    else:
        args.subsets = args.test_subsets
        assert args.mode == 'test', 'mode must be train or test'
        full_test=True
        skip_step = 1

    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])
    val_dataset = Read(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)

    args.agents = val_dataset.agents
    args.actions = val_dataset.actions
    args.duplexes = val_dataset.duplexes
    args.triplets = val_dataset.triplets
    args.locations = val_dataset.locations
    args.num_agents = len(val_dataset.agents)
    args.num_actions = len(val_dataset.actions)
    args.num_duplexes = len(val_dataset.duplexes)
    args.num_triplets = len(val_dataset.triplets)
    args.num_locations = len(val_dataset.locations)
    args.num_classes =  1 + args.num_agents+args.num_actions+args.num_duplexes+args.num_triplets+args.num_locations # one for objectness
    args.label_types = ['agent', 'action', 'duplexes', 'triplets', 'location']
    args.num_classes_list = [args.num_agents, args.num_actions, args.num_duplexes, args.num_triplets, args.num_locations]
    
    args.num_label_type = train_dataset.num_label_type
    args.nlts = args.num_label_type
    args.use_bias = args.use_bias>0
    args.head_size = 256
    
    net = build_retinanet_shared_heads(args).cuda()
    
    if args.multi_gpu:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    
    if args.mode == 'train':
        if args.fbn:
            if args.multi_gpu:
                net.module.backbone_net.apply(utils.set_bn_eval)
            else:
                net.backbone_net.apply(utils.set_bn_eval)
            
        train(args, net, train_dataset, val_dataset)
    else:
        test(args, net)

if __name__ == "__main__":
    main()