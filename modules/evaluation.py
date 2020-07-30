'''

Author:: Gurkirt Singh

'''

import os
import json
import time
import pdb
import numpy as np
import scipy.io as io  # to save detection as mat files
from data.aarav import is_part_of_subsets, get_filtered_tubes
from modules.tube_helper import get_tube_3Diou, make_det_tube


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # print('voc_ap() - use_07_metric:=' + str(use_07_metric))
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def pr_to_ap(pr):
    """
    Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)


def get_gt_of_cls(gt_boxes, cls):
    cls_gt_boxes = []
    for i in range(gt_boxes.shape[0]):
        # print('something here', gt_boxes, gt_boxes.shape, len(gt_boxes.shape), gt_boxes.shape[-1])
        if len(gt_boxes.shape) > 1 and int(gt_boxes[i, -1]) == cls:
            cls_gt_boxes.append(gt_boxes[i, :-1])
    return np.asarray(cls_gt_boxes)


def compute_iou(cls_gt_boxes, box):

    ious = np.zeros(cls_gt_boxes.shape[0])

    for m in range(cls_gt_boxes.shape[0]):
        gtbox = cls_gt_boxes[m]

        xmin = max(gtbox[0], box[0])
        ymin = max(gtbox[1], box[1])
        xmax = min(gtbox[2], box[2])
        ymax = min(gtbox[3], box[3])
        iw = np.maximum(xmax - xmin, 0.)
        ih = np.maximum(ymax - ymin, 0.)
        if iw > 0 and ih > 0:
            intsc = iw*ih
        else:
            intsc = 0.0
        # print (intsc)
        union = (gtbox[2] - gtbox[0]) * (gtbox[3] - gtbox[1]) + \
            (box[2] - box[0]) * (box[3] - box[1]) - intsc
        ious[m] = intsc/union

    return ious


def evaluate_detections(gt_boxes, det_boxes, classes=[], iou_thresh=0.5):

    ap_strs = []
    num_frames = len(gt_boxes)
    print('Evaluating for ', num_frames, 'frames')
    ap_all = np.zeros(len(classes), dtype=np.float32)
    # loop over each class 'cls'
    for cls_ind, class_name in enumerate(classes):
        scores = np.zeros(num_frames * 2000)
        istp = np.zeros(num_frames * 2000)
        det_count = 0
        num_postives = 0.0
        for nf in range(num_frames):  # loop over each frame 'nf'
                # if len(gt_boxes[nf])>0 and len(det_boxes[cls_ind][nf]):
            # get frame detections for class cls in nf
            frame_det_boxes = np.copy(det_boxes[cls_ind][nf])
            # get gt boxes for class cls in nf frame
            cls_gt_boxes = get_gt_of_cls(np.copy(gt_boxes[nf]), cls_ind)
            num_postives += cls_gt_boxes.shape[0]
            # check if there are dection for class cls in nf frame
            if frame_det_boxes.shape[0] > 0:
                # sort in descending order
                sorted_ids = np.argsort(-frame_det_boxes[:, -1])
                for k in sorted_ids:  # start from best scoring detection of cls to end
                    box = frame_det_boxes[k, :-1]  # detection bounfing box
                    score = frame_det_boxes[k, -1]  # detection score
                    ispositive = False  # set ispostive to false every time
                    # we can only find a postive detection
                    if cls_gt_boxes.shape[0] > 0:
                        # if there is atleast one gt bounding for class cls is there in frame nf
                        # compute IOU between remaining gt boxes
                        iou = compute_iou(cls_gt_boxes, box)
                        # and detection boxes
                        # get the max IOU window gt index
                        maxid = np.argmax(iou)
                        # check is max IOU is greater than detection threshold
                        if iou[maxid] >= iou_thresh:
                            ispositive = True  # if yes then this is ture positive detection
                            # remove assigned gt box
                            cls_gt_boxes = np.delete(cls_gt_boxes, maxid, 0)
                    # fill score array with score of current detection
                    scores[det_count] = score
                    if ispositive:
                        # set current detection index (det_count)
                        istp[det_count] = 1
                        #  to 1 if it is true postive example
                    det_count += 1
        if num_postives < 1:
            num_postives = 1
        scores = scores[:det_count]
        istp = istp[:det_count]
        argsort_scores = np.argsort(-scores)  # sort in descending order
        istp = istp[argsort_scores]  # reorder istp's on score sorting
        fp = np.cumsum(istp == 0)  # get false positives
        tp = np.cumsum(istp == 1)  # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_postives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # compute average precision using voc2007 metric
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        # print(cls_ind,classes[cls_ind], cls_ap)
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)

    print('mean ap ', np.mean(ap_all))
    return np.mean(ap_all), ap_all, ap_strs


def filter_labels(labels, n):
    new_labels = []
    for k in range(labels.shape[1]):
        if labels[n, k] > -1:
            new_labels.append(int(labels[n, k]))
    return new_labels


def evaluate_locations(gt_boxes, det_boxes, classes, iou_thresh=0.5):

    ap_strs = []
    num_frames = len(gt_boxes)
    print('Evaluating for ', num_frames, 'frames')
    ap_all = np.zeros(len(classes), dtype=np.float32)
    num_c = len(classes)

    scores = np.zeros((num_frames * 2000, num_c))
    istp_all = np.zeros((num_frames * 2000, num_c))
    det_count = 0
    gt_count = 0.0
    recall_count = 0.0
    for nf in range(num_frames):  # loop over each frame 'nf'
        # if len(gt_boxes[nf])>0 and len(det_boxes[cls_ind][nf]):
        # get frame detections for class cls in nf
        frame_det_boxes = np.copy(det_boxes[0][nf])
        frame_gt_boxes = gt_boxes[nf][0]
        frame_gt_labels = gt_boxes[nf][1]
        # iou = compute_iou(frame_gt_boxes, box) # compute IOU between remaining gt boxes
        gt_count += frame_det_boxes.shape[0]
        # check if there are dection for class cls in nf frame
        if frame_det_boxes.shape[0] > 0 and frame_gt_boxes.shape[0] > 0:
            # sort in descending order
            sorted_ids = np.argsort(-frame_det_boxes[:, -1])
            for k in sorted_ids:  # start from best scoring detection of cls to end
                box = frame_det_boxes[k, :4]  # detection bounfing box
                # score = frame_det_boxes[k,-1] # detection score
                # we can only find a postive detection
                if frame_gt_boxes.shape[0] > 0:
                    # compute IOU between remaining gt boxes
                    iou = compute_iou(frame_gt_boxes, box)
                    # and detection boxes
                    maxid = np.argmax(iou)  # get the max IOU window gt index
                    # check is max IOU is greater than detection threshold
                    if iou[maxid] >= iou_thresh:
                        true_labels = filter_labels(frame_gt_labels, maxid)
                        recall_count += 1
                        for cc in range(num_c):
                             #ispositive = False
                            if cc in true_labels:
                                # set current detection index (det_count) ispositive = True
                                istp_all[det_count, cc] = 1
                                # print('here')
                            # if det_count == 0:
                            #     pdb.set_trace()
                            scores[det_count, cc] = frame_det_boxes[k, cc+4]

                        # remove assigned gt box
                        frame_gt_boxes = np.delete(frame_gt_boxes, maxid, 0)
                        frame_gt_labels = np.delete(frame_gt_labels, maxid, 0)

                        det_count += 1
                else:
                    break

    gt_count = max(1, gt_count)
    print_str = '\n\nRecalled {:d} out of {:d} ground truhs Accuracy is {:0.2f} % \n\n'.format(
        int(recall_count), int(gt_count), 100.0*recall_count/gt_count)
    scores = scores[:det_count, :]
    istp = istp_all[:det_count, :]
    # pdb.set_trace()
    for cls_ind in range(num_c):
        cls_scores = scores[:, cls_ind]
        argsort_scores = np.argsort(-cls_scores)  # sort in descending order
        # reorder istp's on score sorting
        istp = istp_all[argsort_scores, cls_ind].copy()
        fp = np.cumsum(istp == 0)  # get false positives
        tp = np.cumsum(istp == 1)  # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        num_postives = max(1, np.sum(istp))
        recall = tp / num_postives  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # pdb.set_trace()
        # compute average precision using voc2007 metric
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        # print(cls_ind,classes[cls_ind], cls_ap)
        ap_str = str(classes[cls_ind]) + ' : ' + str(num_postives) + \
            ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)
    ap_strs[-1] += print_str
    print('mean ap ', np.mean(np.asarray(ap_all)))
    return np.mean(ap_all), ap_all, ap_strs


def evaluate(gts, dets, all_classes, iou_thresh=0.5):
    # np.mean(ap_all), ap_all, ap_strs
    aps, aps_all, ap_strs = [], [], []
    for nlt in range(len(all_classes)):
        # if nlt<3:
        a, b, c = evaluate_detections(
            gts[nlt], dets[nlt], all_classes[nlt], iou_thresh)
        # else: # evlauate locations
        # a, b, c = evaluate_locations(gts[nlt], dets[nlt], all_classes[nlt], iou_thresh)
        aps.append(a)
        aps_all.append(b)
        ap_strs.append(c)
    return aps, aps_all, ap_strs


def evaluate_ego(gts, dets, classes):
    
    ap_strs = []
    num_frames = gts.shape[0]
    print('Evaluating for ', num_frames, 'frames')
    ap_all = np.zeros(len(classes), dtype=np.float32)

    # num_frames = len(gt_boxes)
    # loop over each class 'cls'
    for cls_ind, class_name in enumerate(classes):
        scores = dets[:, cls_ind]
        istp = np.zeros_like(gts)
        istp[gts == cls_ind] = 1
        det_count = num_frames
        num_postives = np.sum(istp)
        if num_postives < 1:
            num_postives = 1
        argsort_scores = np.argsort(-scores)  # sort in descending order
        istp = istp[argsort_scores]  # reorder istp's on score sorting
        fp = np.cumsum(istp == 0)  # get false positives
        tp = np.cumsum(istp == 1)  # get  true positives
        fp = fp.astype(np.float64)
        tp = tp.astype(np.float64)
        recall = tp / float(num_postives)  # compute recall
        # compute precision
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # compute average precision using voc2007 metric
        cls_ap = voc_ap(recall, precision)
        ap_all[cls_ind] = cls_ap
        # print(cls_ind,classes[cls_ind], cls_ap)
        ap_str = class_name + ' : ' + \
            str(num_postives) + ' : ' + str(det_count) + ' : ' + str(cls_ap)
        ap_strs.append(ap_str)

    return [np.mean(ap_all)], [ap_all], [ap_strs]


def save_detection_framewise(det_boxes, image_ids, det_save_dir):
    # det_save_dir = '/mnt/mars-beta/gur-workspace/use-ssd-data/UCF101/detections/RGB-01-{:06d}/'.format(iteration)
    print('Saving detections to', det_save_dir)
    num_images = len(image_ids)
    for idx in range(num_images):
        img_id = image_ids[idx]
        save_path = det_save_dir+img_id[:-5]
        if not os.path.isdir(save_path):
            os.system('mkdir -p '+save_path)
        fid = open(det_save_dir+img_id+'.txt', 'w')
        for cls_ind in range(len(det_boxes)):
            frame_det_boxes = det_boxes[cls_ind][idx]
            for d in range(len(frame_det_boxes)):
                line = str(cls_ind+1)
                for k in range(5):
                    line += ' {:f}'.format(frame_det_boxes[d, k])
                line += '\n'
                fid.write(line)
        fid.close()


def get_gt_tubes(final_annots, subset, label_type):
    """Get video list form ground truth videos used in subset 
    and their ground truth tubes """

    video_list = []
    tubes = {}
    for videoname in final_annots['db']:
        if is_part_of_subsets(final_annots['db'][videoname]['split_ids'], [subset]):
            video_list.append(videoname)
            tubes[videoname] = get_filtered_tubes(
                label_type+'_tubes', final_annots, videoname)

    return video_list, tubes


def get_det_class_tubes(tubes, cl_id):
    class_tubes = []
    for video, video_tubes in tubes.items():
        for tube in video_tubes:
            if tube['label_id'] == cl_id:
                scores, boxes = tube['scores'], tube['boxes']
                frames, label_id  = tube['frames'], tube['label_id']
                class_tubes.append([video, make_det_tube(scores, boxes, frames, label_id)])
    return class_tubes


def get_gt_class_tubes(tubes, cl_id):
    class_tubes = {}
    for video, video_tubes in tubes.items():
        class_tubes[video] = []
        for tube in video_tubes:
            if tube['label_id'] == cl_id:
                class_tubes[video].append(tube)
    return class_tubes


def evaluate_tubes(anno_file, det_file, classes, label_type, subset='val_3', iou_thresh=0.2):

    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    with open(det_file, 'r') as fff:
        detections = json.load(fff)


    ap_all = []
    ap_strs = []
    sap = 0.0
    video_list, gt_tubes = get_gt_tubes(final_annots, subset, label_type)
    det_tubes = {}
    
    for videoname in video_list:
        det_tubes[videoname] = detections[label_type][videoname]

    for cl_id, class_name in enumerate(classes):

        class_dets = get_det_class_tubes(det_tubes, cl_id)
        class_gts = get_gt_class_tubes(gt_tubes, cl_id)

        pr = np.empty((len(class_dets) + 1, 2), dtype=np.float32)
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0

        fn = max(1, sum([len(class_gts[video])
                         for video in class_gts]))  # false negatives
        num_postives = fn
        fp = 0  # false positives
        tp = 0  # true positives

        inv_det_scores = np.asarray([-det[1]['score'] for det in class_dets])
        indexs = np.argsort(inv_det_scores)
        for count, det_id in enumerate(indexs):
            is_positive = False
            detection = class_dets[det_id]
            video, det_tube = detection
            if len(class_gts[video]) > 0:
                ious = np.asarray([get_tube_3Diou(det_tube, gt_tube)
                                   for gt_tube in class_gts[video]])
                max_iou_id = np.argmax(ious)
                if ious[max_iou_id] >= iou_thresh:
                    is_positive = True
                    del class_gts[video][max_iou_id]

            if is_positive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[count+1, 0] = float(tp) / float(tp + fp)
            pr[count+1, 1] = float(tp) / float(tp + fn)

        class_ap = float(100*pr_to_ap(pr))
        sap += class_ap
        ap_all.append(class_ap)
        # print(cls_ind,classes[cls_ind], cls_ap)
        ap_str = class_name + ' : ' + str(num_postives) + \
            ' : ' + str(count) + ' : ' + str(class_ap)
        ap_strs.append(ap_str)
    mAP = sap/len(classes)
    ap_strs.append('\nMean AP:: {:0.2f}'.format(mAP))
    return mAP, ap_all, ap_strs
