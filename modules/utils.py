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

    args.milestones = [int(val) for val in args.milestones.split(',')]
    args.gammas = [float(val) for val in args.gammas.split(',')]
    args.eval_iters = [int(val) for val in args.eval_iters.split(',')]

    args.train_subsets = [val for val in args.train_subsets.split(',')]
    args.val_subsets = [val for val in args.val_subsets.split(',')]
    args.test_subsets = [val for val in args.test_subsets.split(',')]
    
    ## check if subsets are okay
    possible_subets = ['test']
    for idx in range(1,4):
        possible_subets.append('train_'+str(idx))        
        possible_subets.append('val_'+str(idx))        
        
    for subsets in [args.train_subsets, args.val_subsets, args.test_subsets]:
        for subset in subsets:
            assert subset in possible_subets, 'subest should from one of these '+''.join(possible_subets)

    args.dataset = args.dataset.lower()
    args.basenet = args.basenet.lower()

    args.means =[0.485, 0.456, 0.406]
    args.stds = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username
    
    

    print('\n\n ', username, ' is using ', hostname, '\n\n')
    if username == 'gurkirt':
        args.model_dir = '/mnt/mars-gamma/global-models/pytorch-imagenet/'
        if hostname == 'mars':
            args.data_root = '/mnt/mars-fast/datasets/'
            args.save_root = '/mnt/mercury-alpha/'
            args.vis_port = 8097
        elif hostname == 'venus':
            args.data_root = '/mnt/venus-fast/datasets/'
            args.save_root = '/mnt/mercury-alpha/'
            args.vis_port = 8095
        elif hostname == 'mercury':
            args.data_root = '/mnt/mercury-fast/datasets/'
            args.save_root = '/mnt/mercury-alpha/'
            args.vis_port = 8098
        else:
            raise('ERROR!!!!!!!! Specify directories')
    
    print('Your working directories are', args.data_root, args.save_root)
    return args

def create_exp_name(args):
    return 'FPN{:d}x{:d}-{:s}{:02d}-{:s}-hl{:01d}s{:01d}-bn{:d}f{:d}b{:d}-bs{:02d}'.format(
                                            args.min_size, args.max_size, args.dataset, args.split_id, args.basenet,
                                            args.num_head_layers, args.shared_heads, int(args.fbn), 
                                            args.freezeupto, int(args.use_bias),
                                            args.batch_size)

# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') > -1:
        m.eval()
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        

def get_individual_labels(gt_boxes, tgt_labels, nlt):
    
    # print(gt_boxes.shape, tgt_labels.shape)
    
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
    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([])
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk*20)  # idsn - ids after nms
    scores = scores[ids[:min(args.topk,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.topk,counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    return cls_dets

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