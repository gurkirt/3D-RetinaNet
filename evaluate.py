
""" 
    
    Adapted from:
    Modification by: Gurkirt Singh
    Modification started: 2nd April 2019
    large parts of this files are from many github repos
    mainly adopted from
    https://github.com/gurkirt/realtime-action-detection

    Please don't remove above credits and give star to these repos
    Licensed under The MIT License [see LICENSE for details]    

"""

import os
import pdb
import time, json
import socket
import getpass 
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR
from modules import utils
from modules.utils import str2bool
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from data import DetectionDataset, custum_collate
from models.retinanet_shared_heads import build_retinanet_shared_heads
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from data.transforms import Resize
from train import validate

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
# anchor_type to be used in the experiment
parser.add_argument('--anchor_type', default='kmeans', help='kmeans or default')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet50', help='pretrained base model')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
parser.add_argument('--shared_heads', default=0, type=int,help='4 head layers')
parser.add_argument('--num_head_layers', default=4, type=int,help='0 mean no shareding more than 0 means shareing')
parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='coco', help='pretrained base model')
# Input size of image only 600 is supprted at the moment 
parser.add_argument('--min_size', default=600, type=int, help='Input Size for FPN')
parser.add_argument('--max_size', default=1000, type=int, help='Input Size for FPN')
#  data loading argumnets
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=8, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')
parser.add_argument('--loss_type', default='mbox', type=str, help='loss_type')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--eval_iters', default='90000', type=str, help='Chnage the lr @')

# Freeze batch normlisatio layer or not 
parser.add_argument('--fbn', default=True, type=bool, help='if less than 1 mean freeze or else any positive values keep updating bn layers')
parser.add_argument('--freezeupto', default=1, type=int, help='if 0 freeze or else keep updating bn layers')

# Evaluation hyperparameters
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
parser.add_argument('--topk', default=100, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
parser.add_argument('--log_step', default=10, type=int, help='Log after k steps for text/Visdom/tensorboard')
parser.add_argument('--tensorboard', default=False, type=str2bool, help='Use tensorboard for loss/evalaution visualization')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss/evalaution visualization')
parser.add_argument('--vis_port', default=8098, type=int, help='Port for Visdom Server')

# Program arguments
parser.add_argument('--man_seed', default=1, type=int, help='manualseed for reproduction')
parser.add_argument('--multi_gpu', default=1, type=int, help='If  more than then use all visible GPUs by default only one GPU used ') 

# source or dstination directories
parser.add_argument('--data_root', default='/mnt/mercury-fast/datasets/', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default='/mnt/mercury-fast/datasets/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
parser.add_argument('--model_dir', default='', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/


## Parse arguments
args = parser.parse_args()

args = utils.set_args(args, 'test') # set directories and subsets fo datasets

## set random seeds and global settings
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
torch.cuda.manual_seed_all(args.man_seed)
torch.set_default_tensor_type('torch.FloatTensor')


def main():
    
    args.exp_name = utils.create_exp_name(args)

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'


    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])

    val_dataset = DetectionDataset(args, train=False, image_sets=args.val_sets, 
                            transform=val_transform, full_test=False)

    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)

    args.data_dir = val_dataset.root
    args.num_classes = len(val_dataset.classes) + 1
    args.classes = val_dataset.classes
    args.head_size = 256
    
    net = build_retinanet_shared_heads(args).cuda()

    if args.multi_gpu>0:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    net.eval()

    for iteration in args.eval_iters:
        args.det_itr = iteration
        log_file = open("{pt:s}/testing-{it:06d}-{date:%m-%d-%Hx}.log".format(pt=args.save_root, it=iteration, date=datetime.datetime.now()), "w", 10)
        log_file.write(args.exp_name + '\n')
        
        args.model_path = args.save_root + 'model_{:06d}.pth'.format(iteration)
        log_file.write(args.model_path+'\n')
    
        net.load_state_dict(torch.load(args.model_path))

        print('Finished loading model %d !' % iteration)
        # Load dataset
        val_data_loader = data_utils.DataLoader(val_dataset, int(args.batch_size), num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        net.eval() # switch net to evaluation mode
        if args.dataset != 'coco':
            mAP, ap_all, ap_strs , det_boxes = validate(args, net, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)
        else:
            mAP, ap_all, ap_strs , det_boxes = validate_coco(args, net, val_data_loader, val_dataset, iteration, log_file, iou_thresh=args.iou_thresh)

        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str+'\n')
        ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()


def validate_coco(args, net, val_data_loader, val_dataset, iteration_num, resFile_txt, iou_thresh=0.5):
    """Test a FPN network on an image database."""
    print('Validating at ', iteration_num)

    annFile='{}/instances_{}.json'.format(args.data_dir,args.val_sets[0])
    cocoGT=COCO(annFile)
    coco_dets = []
    resFile = args.save_root + 'detections-{:05d}.json'.format(args.det_itr)
    # resFile_txt = open(args.save_root + 'detections-{:05d}.txt'.format(args.det_itr), 'w')
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    det_boxes = [[] for _ in range(num_classes-1)]
    gt_boxes = []
    print_time = True
    val_step = 50
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = nn.Sigmoid().cuda()
    if args.loss_type == 'mbox':
        activation = nn.Softmax(dim=2).cuda()

    idlist = val_dataset.idlist
    all_ids = val_dataset.ids

    with torch.no_grad():
        for val_itr, (images, targets, batch_counts, img_indexs, wh) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)

            images = images.cuda(0, non_blocking=True)
            decoded_boxes, conf_data = net(images)

            conf_scores_all = activation(conf_data).clone()
            
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                
                coco_image_id = int(all_ids[img_indexs[b]][1][8:])
                width, height = wh[b][0], wh[b][1]
                o_width, o_height = wh[b][2], wh[b][3]
                # print(wh[b])
                gt = targets[b, :batch_counts[b]].numpy()
                gt_boxes.append(gt)
                decoded_boxes_b = decoded_boxes[b]
                conf_scores = conf_scores_all[b].clone()
                #Apply nms per class and obtain the results
                for cl_ind in range(1, num_classes):
                    # pdb.set_trace()
                    scores = conf_scores[:, cl_ind].squeeze()
                    if args.loss_type == 'yolo':
                        scores = conf_scores[:, cl_ind].squeeze() * conf_scores[:, 0].squeeze() * 5.0
                    # scoresth, _ = torch.sort(scores, descending=True)
                    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                    # c_mask = scores.gt(min(max(max_scoresth, args.conf_thresh), min_scoresth))  # greater than minmum threshold
                    scores = scores[c_mask].squeeze()
                    # print('scores size',scores.size())
                    if scores.dim() == 0:
                        # print(len(''), ' dim ==0 ')
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    boxes = decoded_boxes_b.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk*10)  # idsn - ids after nms
                    scores = scores[ids[:counts]].cpu().numpy()
                    pick = min(scores.shape[0], args.topk)
                    scores = scores[:pick]
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    boxes = boxes[:pick, :]

                    cls_id = cl_ind-1
                    if len(idlist)>0:
                        cls_id = idlist[cl_ind-1]
                    # pdb.set_trace()
                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])
                        # box_ = [round(boxes[ik, 0], 1), round(boxes[ik, 1],1), round(boxes[ik, 2],1), round(boxes[ik, 3], 1)]
                        box_ = [round(boxes[ik, 0]*o_width/width,1), round(boxes[ik, 1]*o_height/height,1), round(boxes[ik, 2]*o_width/width,1), round(boxes[ik, 3]*o_height/height,1)]
                        # box_ = [round(box_*o_width/width,1), round(), round(boxes[ik, 2],1), round(boxes[ik, 3], 1)]
                        box_[2] = round(box_[2] - box_[0], 1)
                        box_[3] = round(box_[3] - box_[1], 1)
                        box_ = [float(b) for b in box_]
                        coco_dets.append({"image_id" : int(coco_image_id), "category_id" : int(cls_id), 
                                          "bbox" : box_, "score" : float(scores[ik]),
                                        })

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)
                count += 1
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(ts - tf))
    
    # print('Evaluating detections for itration number ', iteration_num)
    mAP, ap_all, ap_strs , det_boxes = evaluate_detections(gt_boxes, det_boxes, val_dataset.classes, iou_thresh=iou_thresh)
    
    for ap_str in ap_strs:
        print(ap_str)
        resFile_txt.write(ap_str+'\n')
    ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
    print(ptr_str)
    resFile_txt.write(ptr_str)
    
    print('saving results :::::')
    with open(resFile,'w') as f:
        json.dump(coco_dets, f)
    
    cocoDt=cocoGT.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGT, cocoDt, 'bbox')
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    resFile_txt.write(ptr_str)
    # pdb.set_trace()
    eval_strings = utils.eval_strings()
    ptr_str = ''
    for sid, val in enumerate(cocoEval.stats):
        ptr_str += eval_strings[sid] + str(val) + '\n'
    print('\n\nPrintning COCOeval Generated results\n\n ')
    print(ptr_str)
    resFile_txt.write(ptr_str)
    return mAP, ap_all, ap_strs , det_boxes
    
if __name__ == '__main__':
    main()
