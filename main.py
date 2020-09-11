
import os
import sys
import torch
import argparse
import numpy as np
from modules import utils
from train import train
from data import Read
from torchvision import transforms
import data.transforms as vtf
from models.retinanet import build_retinanet
from gen_dets import gen_dets, eval_framewise_dets
from tubes import build_eval_tubes
from val import val

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    parser.add_argument('--MODE', default='train', 
                        help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument('--ARCH', default='resnet50', 
                        type=str, help=' base arch')
    parser.add_argument('--MODEL_TYPE', default='C2D',
                        type=str, help=' base model')
    parser.add_argument('--ANCHOR_TYPE', default='RETINA',
                        type=str, help='type of anchors to be used in model')
    parser.add_argument('--MODEL_PATH', default='',
                        help='Location to where imagenet pretrained models exists')  # /mnt/mars-fast/datasets/
    parser.add_argument('--SEQ_LEN', default=4,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--HEAD_LAYERS', default=0, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--NUM_FEATURE_MAPS', default=5, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--CLS_HEAD_TIME_SIZE', default=3, 
                        type=int, help='Temporal kernel size of classification head')
    parser.add_argument('--REG_HEAD_TIME_SIZE', default=1,
                    type=int, help='Temporal kernel size of regression head')
    #  Name of the dataset only voc or coco are supported
    parser.add_argument('--DATASET', default='aarav', 
                        type=str,help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_3,', 
                        type=str,help='Training SUBSETS seprated by ,')
    parser.add_argument('--VAL_SUBSETS', default='', 
                        type=str,help='Validation SUBSETS seprated by ,')
    parser.add_argument('--TEST_SUBSETS', default='val_3', 
                        type=str,help='Testing SUBSETS seprated by ,')
    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--MIN_SIZE', default=416, 
                        type=int, help='Input Size for FPN')
    parser.add_argument('--MAX_SIZE', default=576, 
                        type=int, help='Input Size for FPN')
    #  data loading argumnets
    parser.add_argument('-b','--BATCH_SIZE', default=8, 
                        type=int, help='Batch size for training')
    # Number of worker to load data in parllel
    parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                        type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--OPTIM', default='SGD', 
                        type=str, help='Optimiser type')
    parser.add_argument('--RESUME', default=0, 
                        type=int, help='Resume from given iterations')
    parser.add_argument('--MAX_ITERS', default=30000, 
                        type=int, help='Number of training iterations')
    parser.add_argument('-l','--LR', '--learning-rate', 
                        default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--MOMENTUM', default=0.9, 
                        type=float, help='momentum')
    parser.add_argument('--MILESTONES', default='15000,25000', 
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
    parser.add_argument('--INTIAL_VAL', default=1000, 
                        type=int, help='Initial number of training iterations before evaluation')
    parser.add_argument('--VAL_STEP', default=2500, 
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
    parser.add_argument('--TUBES_TOPK', default=3,
                        type=int, help='eval iterations')
    
    # Progress logging
    parser.add_argument('--LOG_START', default=0, 
                        type=int, help='start loging after k steps for text/tensorboard') 
                        # Let initial ripples settle down
    parser.add_argument('--LOG_STEP', default=1, 
                        type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--TENSORBOARD', default=1,
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


    ## Parse arguments
    args = parser.parse_args()

    args = utils.set_args(args) # set directories and SUBSETS fo datasets
    args.MULTI_GPUS = False if args.BATCH_SIZE == 1 else args.MULTI_GPUS
    ## set random seeds and global settings
    np.random.seed(args.MAN_SEED)
    torch.manual_seed(args.MAN_SEED)
    torch.cuda.manual_seed_all(args.MAN_SEED)
    torch.set_default_tensor_type('torch.FloatTensor')

    args = utils.create_exp_name(args)

    utils.setup_logger(args)
    logger = utils.get_logger(__name__)
    logger.info(sys.version)

    assert args.MODE in ['train','val','gen_dets','eval_frames', 'eval_tubes'], 'MODE must be from ' + ','.join(['train','test','tubes'])
    
    if args.MODE == 'train':
        args.SUBSETS = args.TRAIN_SUBSETS
        train_transform = transforms.Compose([
                            vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                            vtf.ToTensorStack(),
                            vtf.Normalize(mean=args.MEANS, std=args.STDS)])
        train_dataset = Read(args, train=True, skip_step=args.SEQ_LEN, transform=train_transform)
        logger.info('Done Loading Dataset Train Dataset')
        ## For validation set
        full_test = False
        args.SUBSETS = args.VAL_SUBSETS
        skip_step = args.SEQ_LEN*8
    else:
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.VAL_SUBSETS #args.TEST_SUBSETS + args.VAL_SUBSETS
        full_test = False #args.MODE != 'train'
        args.skip_beggning = 0
        args.skip_ending = 0
        if args.MODEL_TYPE == 'C2D':
            skip_step = args.SEQ_LEN
        else:
            skip_step = args.SEQ_LEN/2
        skip_step = args.SEQ_LEN*8


    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS,std=args.STDS)])
    

    val_dataset = Read(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)
    logger.info('Done Loading Dataset Validation Dataset')

    args.num_classes =  val_dataset.num_classes
    # one for objectness
    args.label_types = val_dataset.label_types
    args.num_label_types = val_dataset.num_label_types
    args.all_classes =  val_dataset.all_classes
    args.num_classes_list = val_dataset.num_classes_list
    args.num_ego_classes = val_dataset.num_ego_classes
    args.ego_classes = val_dataset.ego_classes
    args.head_size = 256

    if args.MODE in ['train', 'val','gen_dets']:
        net = build_retinanet(args).cuda()
        if args.MULTI_GPUS:
            logger.info('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)

    if args.MODE == 'train':
        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.apply(utils.set_bn_eval)
            else:
                net.backbone.apply(utils.set_bn_eval)
        train(args, net, train_dataset, val_dataset)
    elif args.MODE == 'val':
        val(args, net, val_dataset)
    elif args.MODE == 'gen_dets':
        gen_dets(args, net, val_dataset)
        eval_framewise_dets(args, val_dataset)
    elif args.MODE == 'eval_frames':
        eval_framewise_dets(args, val_dataset)
    elif args.MODE == 'eval_tubes':
        build_eval_tubes(args, val_dataset)
    

if __name__ == "__main__":
    main()
