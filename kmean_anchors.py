from modules.box_utils import point_form, jaccard
from make_anchors.base_anchors import anchorBox
from modules.evaluation import get_gt_frames
import torch, pdb, json, os
import numpy as np
from data import VideoDataset
import argparse

parser = argparse.ArgumentParser(description='prepare VOC dataset')
# anchor_type to be used in the experiment
parser.add_argument('--base_dir', default='/mnt/mercury-alpha/', help='Location to root directory for the dataset') 
# /mnt/mars-fast/datasets/

feature_size = [75, 38, 19, 10, 5]
feature_size = [1, 1, 1, 1, 1]
thresh = 0.5

def  get_unique_anchors():
        # print(print_str)
        anchorbox = anchorBox()
        anchors = anchorbox.forward(feature_size)
        print(anchors.size())
        unique_anchors = anchors.numpy()
        unique_anchors[:,0] = unique_anchors[:,0]*0
        unique_anchors[:,1] = unique_anchors[:,1]*0
        anchors = np.unique(unique_anchors, axis=0)
        return torch.from_numpy(anchors)

def get_dataset_boxes(base_dir, dataset, train_sets):
    anno_file = os.path.join(base_dir, dataset, 'annots_12fps_full_v1.0.json')
    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    print(train_sets)
    _, gt_frames = get_gt_frames(final_annots, train_sets, 'agent_ness')
    print('Length of gt frames', len(gt_frames))
    all_boxes = None
    for name, frame in gt_frames.items():
        if len(frame)==0:
            continue
        boxes = []
        for box in frame:
            boxes.append(box[0])
        boxes = torch.FloatTensor(boxes).view(-1,4)
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]
        boxes[:,0] = boxes[:,0] * 0.0
        boxes[:,1] = boxes[:,1] * 0.0
        
        if all_boxes is None:
            all_boxes = boxes
        else:
            all_boxes = torch.cat((all_boxes, boxes),0)
    
    print('Total number of boxes', all_boxes.shape)
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
    for dataset in ['road']:
        if dataset == 'coco':
            train_sets = ['train2017']
            val_sets = ['val2017']
            max_itr = 10
        else:
            train_sets = ['train_1', 'train_2', 'train_3']
            val_sets = ['val_1', 'val_2','val_3']
            # val_sets = ['test']
            max_itr = 10
        
        unique_anchors = get_unique_anchors()
        centers = unique_anchors.clone()
        print(unique_anchors.size())
        numc = centers.size(0)
        boxes = get_dataset_boxes(base_dir, dataset, train_sets)
        print('mins', boxes[:,2].min(), boxes[:,3].min())
        print('maxes', boxes[:,2].max(), boxes[:,3].max())
        print('mean', boxes[:,2].mean(), boxes[:,3].mean())
        print('std', boxes[:,2].std(), boxes[:,3].std())
        print('Initial centers\n', centers, boxes.size())
        print('Areas of each:::', get_area(centers))
        overlaps = jaccard(boxes, centers)
        all_recall, best_center_idx = overlaps.max(1, keepdim=True)
        count = all_recall.size(0)
        # print(scales)
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
        # print(scales)
        print('Train Set: {:s}:: recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                        100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))

        # print(centers)
        boxes = get_dataset_boxes(base_dir, dataset, val_sets)
        overlaps = jaccard(boxes, centers)
        all_recall, best_center_idx = overlaps.max(1, keepdim=True)
        count = all_recall.size(0)
        # print(scales)
        print('Val Set: {:s}:: recall more than 0.5 {:.02f} average is {:.02f}'.format(dataset, 
                        100.0*torch.sum(all_recall>thresh)/count, torch.mean(all_recall)))
        

if __name__ == '__main__':
    args = parser.parse_args()
    kmean_whs(args.base_dir)