from modules.box_utils import point_form, jaccard
from modules.anchor_box_base import anchorBox
import torch, pdb
import numpy as np
from data import Read
import argparse

parser = argparse.ArgumentParser(description='prepare VOC dataset')
# anchor_type to be used in the experiment
parser.add_argument('--base_dir', default='/mnt/mercury-alpha/aarav/', help='Location to root directory for the dataset') 
# /mnt/mars-fast/datasets/

input_dim = 300
feature_size = [75, 38, 19, 10, 5]
thresh = 0.5

def  get_unique_anchors(scales):
        # print(print_str)
        anchorbox = anchorBox('defined3', input_dim=input_dim, scale_ratios=scales)
        anchors = anchorbox.forward()
        print(anchors.size())
        unique_anchors = anchors.numpy()
        unique_anchors[:,0] = unique_anchors[:,0]*0
        unique_anchors[:,1] = unique_anchors[:,1]*0
        anchors = np.unique(unique_anchors, axis=0)
        return torch.from_numpy(anchors)

def get_dataset_boxes(base_dir, dataset, train_sets):
    train_dataset = Read(args, train=True, skip_step=args.SEQ_LEN, transform=train_transform)
    all_boxes = None
    for index in range(len(trainlist)):
        annot_info = trainlist[index]
        img_id = annot_info[1]
        targets = np.asarray(annot_info[3])
        bboxes = torch.FloatTensor(annot_info[2])
        # print(bboxes)
        bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
        bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
        bboxes[:,0] = bboxes[:,0] * 0.0
        bboxes[:,1] = bboxes[:,1] * 0.0
        if all_boxes is None:
                all_boxes = bboxes
        else:
                # pdb.set_trace()
                all_boxes = torch.cat((all_boxes, bboxes),0)
        return all_boxes

def get_center(b_idx, boxes, c):
        # pdb.set_trace()
        mask = b_idx==c
        mask = mask.squeeze()
        new_boxes = boxes[mask,:]
        return new_boxes.mean(0)

def get_area(centers):
        return centers[:,2]*centers[:,3]

def kmean_whs(base_dir):
    for dataset in ['voc','coco']:
        scales = [1.,]
        if dataset == 'coco':
            train_sets = ['train2017']
            val_sets = ['val2017']
            max_itr = 10
        else:
            train_sets = ['train2007', 'val2007', 'train2012', 'val2012']
            val_sets = ['test2007']
            max_itr = 10
        
        unique_anchors = get_unique_anchors(scales)
        centers = unique_anchors.clone()
        print(unique_anchors.size())
        numc = centers.size(0)
        boxes = get_dataset_boxes(base_dir, dataset, train_sets)
        print('Initial centers\n', centers, boxes.size())
        print('Areas of each:::', get_area(centers))
        overlaps = jaccard(boxes, centers)
        all_recall, best_center_idx = overlaps.max(1, keepdim=True)
        count = all_recall.size(0)
        print(scales)
        print('{:s} recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                        100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))

        for itr in range(max_itr):
            overlaps = jaccard(boxes, centers)
            all_recall, best_center_idx = overlaps.max(1, keepdim=True)
            for c in range(numc):
                centers[c,:] = get_center(best_center_idx, boxes, c)
            print('Train Set: {:s}::{:d} recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                    itr, 100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))
        
        print(centers)
        print('Areas of each:::', get_area(centers))
        overlaps = jaccard(boxes, centers)
        all_recall, best_center_idx = overlaps.max(1, keepdim=True)
        count = all_recall.size(0)
        print(scales)
        print('Train Set: {:s}:: recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                        100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))

        # print(centers)
        boxes = get_dataset_boxes(base_dir, dataset, val_sets)
        overlaps = jaccard(boxes, centers)
        all_recall, best_center_idx = overlaps.max(1, keepdim=True)
        count = all_recall.size(0)
        print(scales)
        print('Val Set: {:s}:: recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                        100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))
        

if __name__ == '__main__':
    args = parser.parse_args()
    kmean_whs(args.base_dir)