
"""

This script contain valiudation code at the time of training

"""
import time
import torch
import numpy as np
from modules import utils
from modules.evaluation import evaluate
from modules.box_utils import decode
from modules.utils import get_individual_labels


def validate(args, net,  val_data_loader, val_dataset, iteration_num):
    """Test a FPN network on an image database."""
    iou_thresh = args.iou_thresh
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()
    if args.loss_type == 'mbox':
        activation = torch.nn.Softmax(dim=2).cuda()

    det_boxes = []
    gt_boxes_all = []
    for nlt in range(args.nlts):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        gt_boxes_all.append([])

    with torch.no_grad():
        for val_itr, (images, batch_counts, gt_boxes, gt_targets, img_indexs, wh) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            images = images.cuda(0, non_blocking=True)
            decoded_boxes, confidence = net(images)
            confidence = activation(confidence)

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                width, height = wh[b][0], wh[b][1]
                gt_boxes_batch = gt_boxes[b, :batch_counts[b]].numpy()
                decoded_boxes_batch = decoded_boxes[b]
                gt_labels_batch =  gt_targets[b, :batch_counts[b]].numpy()
                cc = 1 
                
                for nlt in range(args.nlts):
                    frame_gt = get_individual_labels(gt_boxes_batch, gt_labels_batch[:, nlt,:], nlt)
                    gt_boxes_all[nlt].append(frame_gt)
                    num_c = args.num_classes_list[nlt]
                    
                    for cl_ind in range(num_c):
                        scores = confidence[b, :, cc].squeeze().clone()
                        cc += 1
                        cls_dets = utils.filter_detections(args, scores, decoded_boxes_batch)
                        det_boxes[nlt][cl_ind].append(cls_dets)

                count += 1
            

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))

    print('Evaluating detections for itration number ', iteration_num)
    
    return evaluate(gt_boxes_all, det_boxes, args.all_classes, iou_thresh=iou_thresh)