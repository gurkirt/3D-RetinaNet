
""" 
 Testing 
"""

import os
import time, json
import datetime
import numpy as np
import torch
import pdb
import torch.utils.data as data_utils
from modules import utils
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from data import custum_collate

def test(args, net, val_dataset):
    
    net.eval()

    val_data_loader = data_utils.DataLoader(val_dataset, int(args.batch_size), num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)

    for iteration in args.EVAL_ITERS:
        args.det_itr = iteration
        print('Testing at ', iteration)
        log_file = open("{pt:s}/testing-{it:06d}-{date:%m-%d-%Hx}.log".format(pt=args.save_root, it=iteration, date=datetime.datetime.now()), "w", 10)
        args.det_save_dir = "{pt:s}/detections-{it:06d}/".format(pt=args.save_root, it=iteration)
        if not os.path.isdir(args.det_save_dir): #if save directory doesn't exist create it
            os.makedirs(args.det_save_dir)
        log_file.write(args.exp_name + '\n')
        args.model_path = args.save_root + 'model_{:06d}.pth'.format(iteration)
        log_file.write(args.model_path+'\n')
        net.load_state_dict(torch.load(args.model_path))
        
        print('Finished loading model %d !' % iteration )
        
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        
        net.eval() # switch net to evaluation mode        
        mAP, ap_all, ap_strs = perform_test(args, net, val_data_loader, val_dataset, iteration)
        
        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()


def perform_test(args, net,  val_data_loader, val_dataset, iteration):

    """Test a FPN network on an image database."""
    iou_thresh = args.iou_thresh
    # print('Validating at ', iteration)
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    print_time = True
    val_step = 50
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()
    if args.loss_type == 'mbox':
        activation = torch.nn.Softmax(dim=2).cuda()

    det_boxes = [[]]
    gt_boxes_all = []
    
    # for nlt in range(args.nlts):
    #     numc = args.num_classes_list[nlt]
    # det_boxes.append([])
    # gt_boxes_all.append([])

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
                index = img_indexs[b]
                annot_info = val_dataset.ids[index]
                frame_num = annot_info[1]
                video_id = annot_info[0]
                videoname = val_dataset.video_list[video_id]
                # save_name = '/{:s}/{:08d}.npy'.format(videoname, frame_num)
                # video_name = val_dataset.vidlist
                width, height = wh[b][0], wh[b][1]
                gt_boxes_batch = gt_boxes[b, :batch_counts[b]].numpy()
                decoded_boxes_batch = decoded_boxes[b]
                confidence_batch = confidence[b]
                gt_labels_batch =  gt_targets[b, :batch_counts[b]].numpy()
                temp_gt = np.asarray([0 for _ in range(gt_boxes_batch.shape[0])])
                frame_gt = np.hstack((gt_boxes_batch, temp_gt[:,np.newaxis]))
                if len(gt_boxes.shape)<2:
                    print(frame_gt.shape, frame_gt)
                    pdb.set_trace()
                gt_boxes_all.append(frame_gt)
                scores = confidence_batch[:, 0].squeeze().clone()
                cls_dets, save_data = utils.filter_detections_with_confidences(args, scores, decoded_boxes_batch, confidence_batch)
                det_boxes[0].append(cls_dets)
                count += 1
                
                save_dir = '{:s}/{}'.format(args.det_save_dir, videoname)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                save_name = '{:s}/{:08d}.npy'.format(save_dir, frame_num)
                np.save(save_name, save_data)

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

    print('Evaluating detections for itration number ', iteration)
    # all_classes =  [args.agents, args.action, args.duplex, args.triplet, args.loc]
    return evaluate_detections(gt_boxes_all, det_boxes, ['Agentness'], iou_thresh=iou_thresh)
    # return evaluate(gt_boxes_all, det_boxes, all_classes, iou_thresh=iou_thresh)
    
if __name__ == '__main__':
    main()
