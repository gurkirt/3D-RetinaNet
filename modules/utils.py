import os
import shutil
import socket
import getpass
import numpy as np
from modules.box_utils import nms


def copy_source(source_dir):
    if not os.path.isdir(source_dir):
        os.system('mkdir -p ' + source_dir)
    
    for dirpath, dirs, files in os.walk('./', topdown=True):
        for file in files:
            if file.endswith('.py'): #fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)


def set_args(args):
    args.MILESTONES = [int(val) for val in args.MILESTONES.split(',')]
    args.GAMMAS = [float(val) for val in args.GAMMAS.split(',')]
    args.EVAL_ITERS = [int(val) for val in args.EVAL_ITERS.split(',')]

    args.TRAIN_SUBSETS = [val for val in args.TRAIN_SUBSETS.split(',') if len(val)>1]
    args.VAL_SUBSETS = [val for val in args.VAL_SUBSETS.split(',') if len(val)>1]
    args.TEST_SUBSETS = [val for val in args.TEST_SUBSETS.split(',') if len(val)>1]
    
    ## check if subsets are okay
    possible_subets = ['test']
    for idx in range(1,4):
        possible_subets.append('train_'+str(idx))        
        possible_subets.append('val_'+str(idx))        

    if len(args.VAL_SUBSETS) < 1:
        args.VAL_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
    if len(args.TEST_SUBSETS) < 1:
        args.TEST_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
        args.TEST_SUBSETS.append('test')
    
    print(args)
    for subsets in [args.TRAIN_SUBSETS, args.VAL_SUBSETS, args.TEST_SUBSETS]:
        for subset in subsets:
            assert subset in possible_subets, 'subest should from one of these '+''.join(possible_subets)

    args.DATASET = args.DATASET.lower()
    args.NET_DEPTH = args.NET_DEPTH.lower()

    args.MEANS =[0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username
    
    
    print('\n\n ', username, ' is using ', hostname, '\n\n')
    if username == 'gurkirt':
        args.model_dir = '/mnt/mars-gamma/global-models/pytorch-imagenet/'
        if hostname == 'mars':
            args.DATA_ROOT = '/mnt/mercury-fast/datasets/'
            args.SAVE_ROOT = '/mnt/mercury-alpha/'
            args.vis_port = 8097
        elif hostname == 'venus':
            args.DATA_ROOT = '/mnt/mercury-fast/datasets/'
            args.SAVE_ROOT = '/mnt/mercury-alpha/'
            args.vis_port = 8095
        elif hostname == 'mercury':
            args.DATA_ROOT = '/mnt/mercury-fast/datasets/'
            args.SAVE_ROOT = '/mnt/mercury-alpha/'
            args.vis_port = 8098
        else:
            raise('ERROR!!!!!!!! Specify directories')
    
    print('Your working directories are', args.DATA_ROOT, args.SAVE_ROOT)
    return args

def create_exp_name(args):
    splits = ''.join([split[0]+split[-1] for split in args.TRAIN_SUBSETS])
    return '{:s}{:s}{:d}x{:d}-{:d}x{:d}-{:s}{:s}-h{:d}x{:d}x{:d}-bn{:d}f{:d}-bs{:02d}'.format(
        args.NET_DEPTH, args.NET_TYPE,
        args.MIN_SIZE, args.MAX_SIZE,
        args.SEQ_LEN, args.SEQ_STEP,
        args.DATASET, splits, 
        args.HEAD_LAYERS, args.CLS_HEAD_TIME_SIZE,
        args.REG_HEAD_TIME_SIZE,
        int(args.FBN), args.FREEZE_UPTO, 
        args.BATCH_SIZE)

# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') > -1:
        m.eval()
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        

def get_individual_labels(gt_boxes, tgt_labels, nlt):

    max_num = np.sum(tgt_labels>-1)
    new_gts = np.zeros((max_num, 5))
    # if max_num<1: # fix it
    #     print('maxnum lower than 1 ', max_num, nlt)
    # pdb.set_trace()
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>-1:
                new_gts[ccc, :4] = gt_boxes[n,:]
                new_gts[ccc, 4] = tgt_labels[n,t]
                ccc += 1
    assert ccc == max_num, 'cc and max num should be the same'
    return new_gts


def get_individual_location_labels(gt_boxes, tgt_labels):
    return [gt_boxes, tgt_labels]


def filter_detections(args, scores, decoded_boxes_batch):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([])
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    ids, counts = nms(boxes, scores, args.nms_thresh, args.TOPK*20)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    return cls_dets


def filter_detections_with_confidences(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([]), np.asarray([])
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences_ = confidences[c_mask,:].clone().view(-1, numc)
    ids, counts = nms(boxes, scores, args.NMS_THRESH, args.TOPK*20)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)]].cpu().numpy()
    confidences_ = confidences_[ids[:min(args.TOPK,counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences_)).astype(np.float32)

    return cls_dets, save_data


def eval_strings():
    return ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ",    
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "]
