
import os
import torch
import argparse
import numpy as np
from modules import utils
from train import train
from data import Read
from torchvision import transforms
from data.transforms import Resize
from models.retinanet_shared_heads import build_retinanet_shared_heads
from test import test
from build_tubes import build_tubes


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    parser.add_argument('--MODE', default='train', 
                        help='mode can be train or test define SUBSETS accordingly, build tubes')
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument('--NET_DEPTH', default='resnet50', 
                        type=str, help='pretrained base model')
    parser.add_argument('--NET_TYPE', default='resnet50',
                        type=str, help='pretrained base model')
    parser.add_argument('--SEQ_LEN', default=1,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--HEAD_LAYERS', default=4, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--CLS_HEAD_TIME_SIZE', default=3, 
                        type=int, help='Temporal kernel size of classification head')
    parser.add_argument('--REG_HEAD_TIME_SIZE', default=3,
                    type=int, help='Temporal kernel size of regression head')
    #  Name of the dataset only voc or coco are supported
    parser.add_argument('--DATASET', default='aarav', 
                        type=str,help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,', 
                        type=str,help='Training SUBSETS seprated by ,')
    parser.add_argument('--VAL_SUBSETS', default='', 
                        type=str,help='Validation SUBSETS seprated by ,')
    parser.add_argument('--TEST_SUBSETS', default='', 
                        type=str,help='Testing SUBSETS seprated by ,')
    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--MIN_SIZE', default=600, 
                        type=int, help='Input Size for FPN')
    parser.add_argument('--MAX_SIZE', default=1000, 
                        type=int, help='Input Size for FPN')
    #  data loading argumnets
    parser.add_argument('--BATCH_SIZE', default=16, 
                        type=int, help='Batch size for training')
    # Number of worker to load data in parllel
    parser.add_argument('--NUM_WORKERS', '-j', default=0, 
                        type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--OPTIM', default='SGD', 
                        type=str, help='Optimiser type')
    parser.add_argument('--RESUME', default=0, 
                        type=int, help='Resume from given iterations')
    parser.add_argument('--MAX_ITERS', default=25000, 
                        type=int, help='Number of training iterations')
    parser.add_argument('--LR', '--learning-rate', 
                        default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--MOMENTUM', default=0.9, 
                        type=float, help='momentum')
    parser.add_argument('--loss_type', default='focal', 
                        type=str, help='loss_type')
    parser.add_argument('--MILESTONES', default='10000,20000', 
                        type=str, help='Chnage the lr @')
    parser.add_argument('--GAMMAS', default='0.1,0.1', 
                        type=str, help='Gamma update for SGD')
    parser.add_argument('--WEIGHT_DECAY', default=1e-4, 
                        type=float, help='Weight decay for SGD')
    # Freeze layers or not 
    parser.add_argument('--FBN','--FREEZE_BN', default=True, 
                        type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
    parser.add_argument('--FREEZE_UPTO', default=1, 
                        type=int, help='layer group number in ResNet up to which needs to be frozen')

    # Loss function matching threshold
    parser.add_argument('--POSTIVE_THRESHOLD', default=0.5, 
                        type=float, help='Min threshold for Jaccard index for matching')
    parser.add_argument('--NEGTIVE_THRESHOLD', default=0.4,
                        type=float, help='Max threshold Jaccard index for matching')

    # Evaluation hyperparameters
    parser.add_argument('--EVAL_ITERS', default='30000', 
                        type=str, help='eval iterations')
    parser.add_argument('--INTIAL_VAL', default=5000, 
                        type=int, help='Initial number of training iterations before evaluation')
    parser.add_argument('--VAL_STEP', default=5000, 
                        type=int, help='Number of training iterations before evaluation')
    parser.add_argument('--IOU_THRESH', default=0.5, 
                        type=float, help='Evaluation threshold')
    parser.add_argument('--CONF_THRESH', default=0.05, 
                        type=float, help='Confidence threshold for evaluation')
    parser.add_argument('--NMS_THRESH', default=0.45, 
                        type=float, help='NMS threshold')
    parser.add_argument('--TOPK', default=25, 
                        type=int, help='topk for evaluation')

    ## paths hyper parameters
    parser.add_argument('--COMPUTE_PATHS', default=False, 
                        type=str2bool, help='eval iterations')
    parser.add_argument('--PATHS_IOUTH', default=0.1,
                        type=float, help='Iouth for building paths')
    parser.add_argument('--PATHS_COST_TYPE', default='scoreiou',
                        type=str, help='eval iterations')
    parser.add_argument('--PATHS_JUMP_GAP', default=2,
                        type=int, help='eval iterations')
    parser.add_argument('--PATHS_MIN_LEN', default=6,
                        type=int, help='eval iterations')
    parser.add_argument('--PATHS_MINSCORE', default=0.3,
                        type=float, help='eval iterations')
    
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_TUBES', default=False, type=str2bool, help='eval iterations')
    parser.add_argument('--TUBES_ALPHA', default=5,
                        type=float, help='eval iterations')
    parser.add_argument('--TUBE_TOPK', default=3,
                        type=int, help='eval iterations')
    
    # Progress logging
    parser.add_argument('--LOG_START', default=149, 
                        type=int, help='start loging after k steps for text/tensorboard') 
                        # Let initial ripples settle down
    parser.add_argument('--LOG_STEP', default=10, 
                        type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--TENSORBOARD', default=TRUE,
                        type=str2bool, help='Use tensorboard for loss/evalaution visualization')

    # Program arguments
    parser.add_argument('--MAN_SEED', default=123, 
                        type=int, help='manualseed for reproduction')
    parser.add_argument('--MULTI_GPUS', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

    # Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use
    parser.add_argument('--DATA_ROOT', default='/mnt/mercury-fast/datasets/', 
                    help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
    parser.add_argument('--SAVE_ROOT', default='/mnt/mercury-alpha/', 
                    help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
    parser.add_argument('--MODEL_PATH', default='', 
                    help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/

    ## Parse arguments
    args = parser.parse_args()

    args = utils.set_args(args) # set directories and SUBSETS fo datasets

    ## set random seeds and global settings
    np.random.seed(args.MAN_SEED)
    torch.manual_seed(args.MAN_SEED)
    torch.cuda.manual_seed_all(args.MAN_SEED)
    torch.set_default_tensor_type('torch.FloatTensor')

    args.exp_name = utils.create_exp_name(args)
    args.SAVE_ROOT += args.DATASET+'/'
    args.SAVE_ROOT = args.SAVE_ROOT+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.SAVE_ROOT): #if save directory doesn't exist create it
        os.makedirs(args.SAVE_ROOT)

    assert args.mode in ['train','test','tubes'], 'mode must be from ' + ','.join(['train','test','tubes'])
    if args.mode == 'train':
        args.SUBSETS = args.TRAIN_SUBSETS
        train_transform = transforms.Compose([
                            #transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.05),
                            Resize(args.MIN_SIZE, args.MAX_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=args.MEANS, std=args.STDS)])

        train_dataset = Read(args, train=True, transform=train_transform)
        print('Done Loading Dataset Train Dataset :::>>>\n',train_dataset.print_str)
        ## For validation set
        full_test = False
        args.SUBSETS = args.VAL_SUBSETS
        skip_step = 3
    else:
        args.SUBSETS = args.TEST_SUBSETS
        full_test=True
        skip_step = 1

    val_transform = transforms.Compose([ 
                        Resize(args.MIN_SIZE, args.MAX_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.MEANS,std=args.STDS)])
    val_dataset = Read(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)

    args.agent = val_dataset.agent
    args.action = val_dataset.action
    args.duplex = val_dataset.duplex
    args.triplet = val_dataset.triplet
    args.loc = val_dataset.loc
    args.num_agent = len(val_dataset.agent)
    args.num_action = len(val_dataset.action)
    args.num_duplex = len(val_dataset.duplex)
    args.num_triplet = len(val_dataset.triplet)
    args.num_loc = len(val_dataset.loc)
    args.num_classes =  1 + args.num_agent+args.num_action+args.num_duplex+args.num_triplet+args.num_loc 
    # one for objectness
    args.label_types = ['agent', 'action', 'duplex', 'triplet', 'loc']
    args.num_classes_list = [args.num_agent, args.num_action, args.num_duplex, args.num_triplet, args.num_loc]
    args.all_classes =  [args.agent, args.action, args.duplex, args.triplet, args.loc]
    args.num_label_type = val_dataset.num_label_type
    args.nlts = args.num_label_type
    args.head_size = 256

    if args.mode != 'tubes':
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

    elif args.mode == 'test':
        test(args, net, val_dataset)
    elif args.mode == 'tubes':
        build_tubes(args, val_dataset)
    

if __name__ == "__main__":
    main()
