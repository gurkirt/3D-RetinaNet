"""

Copyright (c) 2019 Gurkirt Singh 
 All Rights Reserved.

"""

import torch.nn as nn
import torch.nn.functional as F
import torch, pdb, time
from modules import box_utils


# Credits:: from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/smooth_l1_loss.py
# smooth l1 with beta
def smooth_l1_loss(input, target, beta=1. / 9, reduction='sum'):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()


def sigmoid_focal_loss(preds, labels, num_pos, alpha, gamma):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''
    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = preds * labels + (1.0 - preds) * (1.0 - labels)
    focal_weight = alpha_factor * ((1-pt) ** gamma)
    loss = (loss * focal_weight).sum() / num_pos
    return loss


def one_hot_labels(tgt_labels, nlt, numc):
    # pdb.set_trace()
    # tgt_labels = tgt_labels.squeeze(1)
    # print(tgt_labels.shape[0] + 1, numc)
    labels = tgt_labels.new_zeros(tgt_labels.shape[0] + 1, numc)
    # if nlt == 4:
    #     present = [0]
    # else:
    # present = [1]
    # print(tgt_labels.shape)
    for n in range(tgt_labels.shape[0]):
        c = 0
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>=0:
                labels[n+1, tgt_labels[n,t]] = 1
                c = 1
            else:
                break
        
        # present.append(c)
    # print(present)
    
    return labels #, torch.cuda.ByteTensor(present, device=tgt_labels.device)


class FocalLoss(nn.Module):
    def __init__(self, args, alpha=0.25, gamma=2.0):
        """Implement YOLO Loss.
        Basically, combines focal classification loss
         and Smooth L1 regression loss.
        """
        super(FocalLoss, self).__init__()
        self.positive_threshold = args.positive_threshold
        self.negative_threshold = args.negative_threshold
        self.num_agents = args.num_agents
        self.num_actions = args.num_actions
        self.num_duplexes = args.num_duplexes
        self.num_triplets = args.num_triplets
        self.num_locations = args.num_locations
        # # self.num_classes_list = args.num_classes_list 
        self.nlts = args.nlts
        self.num_classes_list = args.num_classes_list
        # self.bce_loss = nn.BCELoss(reduction='sum').cuda()
        self.alpha = 0.25
        self.gamma = 2.0


    def forward(self, confidence, predicted_locations, gt_boxes, gt_labels, counts, anchors, img_index):
        ## gt_boxes, gt_labels, counts, ancohor_boxes
        
        """
        
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4): predicted locations.
            boxes list of len = batch_size and nx4 arrarys
            anchors: (num_anchors, 4)

        """

        confidence = torch.sigmoid(confidence)
        binary_preds = confidence[:,:, 0]
        cc = 1 
        agent_preds = confidence[:,:,cc:cc+self.num_agents]
        cc += self.num_agents
        action_preds = confidence[:,:,cc:cc+self.num_actions]
        cc += self.num_actions
        duplex_preds = confidence[:,:,cc:cc+self.num_duplexes]
        cc += self.num_duplexes
        triplet_preds = confidence[:,:,cc:cc+self.num_triplets]
        cc += self.num_triplets
        location_preds = confidence[:,:,cc:cc+self.num_locations]

        # binary_preds = confidence[:,:, 0]
        # rest_preds = confidence[:,:,1:]
        # num_classes = confidence.size(2)-1
        
        gt_locations = []
        labels = []
        labels_bin = []

        nlts = self.nlts
        # gt_labels = gt_labels
        # all_labels = [[] for _ rnage(self.nlts)]
        all_labels_agents = []
        all_labels_actions = []
        all_labels_duplexes = []
        all_labels_triplets = []
        all_labels_locations = []
        # position_present = []
        
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            # pdb.set_trace()
            for b in range(gt_boxes.shape[0]):
                gt_boxes_batch = gt_boxes[b, :counts[b], :]
                # print(gt_boxes.shape, counts[b])
                gt_labels_batch = torch.cuda.LongTensor([i for i in range(counts[b])])
                # gt_labels = gt_labels.type(torch.cuda.LongTensor)
                conf, loc = box_utils.match_anchors_wIgnore(gt_boxes_batch, gt_labels_batch, 
                    anchors, pos_th=self.positive_threshold, nge_th=self.negative_threshold )
                gt_locations.append(loc)
                labels_bin.append(conf)
                
                dumy_conf = conf.clone()
                dumy_conf[dumy_conf<0] = 0
                batch_labels = []
                for nlt in range(nlts):
                    labels = one_hot_labels(gt_labels[b,:counts[b], nlt, :], nlt, self.num_classes_list[nlt])
                    # pdb.set_trace()
                    batch_labels.append(labels[dumy_conf,:])
                    # if nlt == 3: ## position 
                    #     position_present.append(present[dumy_conf])
                
                all_labels_agents.append(batch_labels[0])
                all_labels_actions.append(batch_labels[1])
                all_labels_duplexes.append(batch_labels[2])
                all_labels_triplets.append(batch_labels[3])
                all_labels_locations.append(batch_labels[4])
            
            # position_present = torch.stack(position_present, 0)
            all_labels_agents = torch.stack(all_labels_agents, 0).float()
            all_labels_actions = torch.stack(all_labels_actions, 0).float()
            all_labels_duplexes = torch.stack(all_labels_duplexes, 0).float()
            all_labels_triplets = torch.stack(all_labels_triplets, 0).float()
            all_labels_locations = torch.stack(all_labels_locations, 0).float()
            gt_locations = torch.stack(gt_locations, 0)
            labels_bin = torch.stack(labels_bin, 0).float()

        pos_mask = labels_bin > 0
        num_pos = max(1.0, float(pos_mask.sum()))
        
        predicted_locations = predicted_locations[pos_mask].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask].reshape(-1, 4)
        regression_loss = smooth_l1_loss(predicted_locations, gt_locations)/(num_pos * 4.0)
        
        mask = labels_bin > -1 # Get mask to remove ignore examples
        # agent_loss
        preds = agent_preds[mask].reshape(-1, self.num_agents) # Remove Ignore preds
        labels = all_labels_agents[mask].reshape(-1, self.num_agents) # Remove Ignore labels
        agent_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)
        
        preds = action_preds[mask].reshape(-1, self.num_actions) # Remove Ignore preds
        labels = all_labels_actions[mask].reshape(-1, self.num_actions) # Remove Ignore labels
        action_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)

        preds = duplex_preds[mask].reshape(-1, self.num_duplexes) # Remove Ignore preds
        labels = all_labels_duplexes[mask].reshape(-1, self.num_duplexes) # Remove Ignore labels
        duplex_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)
        
        preds = triplet_preds[mask].reshape(-1, self.num_triplets) # Remove Ignore preds
        labels = all_labels_triplets[mask].reshape(-1, self.num_triplets) # Remove Ignore labels
        triplet_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)

        preds = location_preds[mask].reshape(-1, self.num_locations) # Remove Ignore preds
        labels = all_labels_locations[mask].reshape(-1, self.num_locations) # Remove Ignore labels
        location_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)

        preds = binary_preds[mask] # Remove Ignore preds
        labels_bin[labels_bin>0] = 1
        labels = labels_bin[mask] # Remove Ignore labels
        binary_loss = sigmoid_focal_loss(preds, labels, num_pos, self.alpha, self.gamma)
        
        # print(num_pos, location_loss.item(), triplet_loss.item(), duplex_loss.item(), action_loss.item(), agent_loss.item())
        return regression_loss, binary_loss + location_loss + triplet_loss + duplex_loss + action_loss + agent_loss