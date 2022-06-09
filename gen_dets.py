
""" 

 Testing 

"""

import os
import time, json
import datetime
import numpy as np
import torch
import pdb
import pickle
import copy
import torch.utils.data as data_utils
from modules.evaluation import evaluate_frames
from modules.box_utils import decode, nms
from data import custum_collate
from modules import utils
import modules.evaluation as evaluate
from modules.utils import make_joint_probs_from_marginals
logger = utils.get_logger(__name__)

def gen_dets(args, net, val_dataset):
    
    net.eval()
    val_data_loader = data_utils.DataLoader(val_dataset, int(args.TEST_BATCH_SIZE), num_workers=args.NUM_WORKERS,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)
    for epoch in args.EVAL_EPOCHS:
        args.det_itr = epoch
        logger.info('Testing at ' + str(epoch))
        
        args.det_save_dir = os.path.join(args.SAVE_ROOT, "detections-{it:02d}-{sq:02d}-{n:d}/".format(it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS)))
        logger.info('detection saving dir is :: '+args.det_save_dir)
        
        is_all_done = True
        if os.path.isdir(args.det_save_dir):
            for vid, videoname in enumerate(val_dataset.video_list):
                save_dir = '{:s}/{}'.format(args.det_save_dir, videoname)
                if os.path.isdir(save_dir):
                    numf = val_dataset.numf_list[vid]
                    dets_list = [d for d in os.listdir(save_dir) if d.endswith('.pkl')]
                    if numf != len(dets_list):
                        is_all_done = False
                        print('Not done', save_dir, numf, len(dets_list))
                        break 
                else:
                    is_all_done = False
                    break
        else:
            is_all_done = False
            os.makedirs(args.det_save_dir)
        
        if is_all_done:
            print('All done! skipping detection')
            continue
        
        args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(epoch)
        net.load_state_dict(torch.load(args.MODEL_PATH))
        
        logger.info('Finished loading model %d !' % epoch )
        
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        
        net.eval() # switch net to evaluation mode        
        mAP, _, ap_strs = perform_detection(args, net, val_data_loader, val_dataset, epoch)
        label_types = [args.label_types[0]] + ['ego_action']
        for nlt in range(len(label_types)):
            for ap_str in ap_strs[nlt]:
                logger.info(ap_str)
        ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
        logger.info(ptr_str)

        torch.cuda.synchronize()
        logger.info('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))


def perform_detection(args, net,  val_data_loader, val_dataset, iteration):

    """Test a network on a video database."""

    num_images = len(val_dataset)    
    print_time = True
    val_step = 50
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()

    ego_pds = []
    ego_gts = []

    det_boxes = []
    gt_boxes_all = []

    for nlt in range(1):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        gt_boxes_all.append([])
    
    nlt = 0
    processed_videos = []
    with torch.no_grad():
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            batch_size = images.size(0)
            
            images = images.cuda(0, non_blocking=True)
            decoded_boxes, confidence, ego_preds = net(images)
            ego_preds = activation(ego_preds).cpu().numpy()
            ego_labels = ego_labels.numpy()
            confidence = activation(confidence)
            seq_len = ego_preds.shape[1]
            
            if val_itr == 5:
                os.system("nvidia-smi")

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                logger.info('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                index = img_indexs[b]
                annot_info = val_dataset.ids[index]
                if args.DATASET != 'ava':
                    video_id, frame_num, step_size = annot_info
                else:
                    video_id, frame_num, step_size, keyframe = annot_info
                    startf = frame_num
                    temp_startf = frame_num
                    frame_num = keyframe-1

                videoname = val_dataset.video_list[video_id]
                save_dir = '{:s}/{}'.format(args.det_save_dir, videoname)
                store_last = False
                if videoname not in processed_videos:
                    processed_videos.append(videoname)
                    store_last = True

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                count += 1
                
                
                
                for si in range(seq_len):
                    if args.DATASET == 'ava' and startf != keyframe:
                        startf += step_size
                        continue
                    
                    if ego_labels[b,si]>-1:
                        ego_pds.append(ego_preds[b,si,:])
                        ego_gts.append(ego_labels[b,si])
                    
                    gt_boxes_batch = gt_boxes[b, si, :batch_counts[b, si],:].numpy()
                    gt_labels_batch =  gt_targets[b, si, :batch_counts[b, si]].numpy()
                    decoded_boxes_batch = decoded_boxes[b,si]
                    frame_gt = utils.get_individual_labels(gt_boxes_batch, gt_labels_batch[:,:1])
                    gt_boxes_all[0].append(frame_gt)
                    confidence_batch = confidence[b,si]
                    scores = confidence_batch[:, 0].squeeze().clone()
                    cls_dets, save_data = utils.filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidence_batch)
                    det_boxes[0][0].append(cls_dets)
                    # print('number of samples', batch_counts[b, si].sum())
                    # pdb.set_trace()
                    save_name = '{:s}/{:05d}.pkl'.format(save_dir, frame_num+1)
                    frame_num += step_size
                    save_data = {'ego':ego_preds[b,si,:], 'main':save_data}
                    
                    if si<seq_len-args.skip_ending or store_last:
                        with open(save_name,'wb') as ff:
                            pickle.dump(save_data, ff)
                    
                    if args.DATASET == 'ava':
                        startf += step_size

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('NMS stuff Time {:0.3f}'.format(te - tf))

    mAP, ap_all, ap_strs = evaluate.evaluate(gt_boxes_all, det_boxes, args.all_classes, iou_thresh=args.IOU_THRESH)
    mAP_ego, ap_all_ego, ap_strs_ego = evaluate.evaluate_ego(np.asarray(ego_gts), np.asarray(ego_pds),  args.ego_classes)
    return mAP + [mAP_ego], ap_all + [ap_all_ego], ap_strs + [ap_strs_ego]



def gather_framelevel_detection(args, val_dataset):
    
    detections = {}
    for l, ltype in enumerate(args.label_types):
        detections[ltype] = {}
    
    if args.DATASET == 'road':
        detections['av_actions'] = {}
    else:
        detections['frame_actions'] = {}
    numv = len(val_dataset.video_list)

    for vid, videoname in enumerate(val_dataset.video_list):       
        vid_dir = os.path.join(args.det_save_dir, videoname)
        frames_list = os.listdir(vid_dir)
        for frame_name in frames_list:
            if not frame_name.endswith('.pkl'):
                continue
            save_name = os.path.join(vid_dir, frame_name)
            with open(save_name,'rb') as ff:
                dets = pickle.load(ff)
            frame_name = frame_name.rstrip('.pkl')
            # detections[videoname+frame_name] = {}
            if args.DATASET == 'road':
                detections['av_actions'][videoname+frame_name] = dets['ego']
            else:
                detections['frame_actions'][videoname+frame_name] = dets['ego']
            frame_dets = dets['main']
            
            if args.JOINT_4M_MARGINALS:
                frame_dets = make_joint_probs_from_marginals(frame_dets, val_dataset.childs, args.num_classes_list)
            
            start_id = 4
            for l, ltype in enumerate(args.label_types):
                numc = args.num_classes_list[l]
                ldets = get_ltype_dets(frame_dets, start_id, numc, ltype, args)
                detections[ltype][videoname+frame_name] = ldets
                start_id += numc

        logger.info('[{}/{}] Done for {}'.format(vid, numv, videoname))
        # break
    logger.info('Dumping detection in ' + args.det_file_name)
    with open(args.det_file_name, 'wb') as f:
            pickle.dump(detections, f)
    logger.info('Done dumping')


def get_ltype_dets(frame_dets, start_id, numc, ltype, args):
    dets = []
    for cid in range(numc):
        if frame_dets.shape[0]>0:
            boxes = frame_dets[:, :4].copy()
            scores = frame_dets[:, start_id+cid].copy()
            pickn = boxes.shape[0]
            if args.CLASSWISE_NMS:
                cls_dets = utils.filter_detections(args, torch.from_numpy(scores), torch.from_numpy(boxes))
            elif pickn<= args.TOPK+15:
                cls_dets = np.hstack((boxes[:pickn,:], scores[:pickn, np.newaxis]))
                if not args.JOINT_4M_MARGINALS:
                    cls_dets = cls_dets[scores>args.CONF_THRESH,:]
            else:
                sorted_ind = np.argsort(-scores)
                sorted_ind = sorted_ind[:args.TOPK+15]
                cls_dets = np.hstack((boxes[sorted_ind,:], scores[sorted_ind, np.newaxis]))
                scores = scores[sorted_ind]
                if not args.JOINT_4M_MARGINALS:
                    cls_dets = cls_dets[scores>args.CONF_THRESH,:]
        else:
            cls_dets = np.asarray([])
        dets.append(cls_dets)
    return dets


def eval_framewise_dets(args, val_dataset):
    for epoch in args.EVAL_EPOCHS:
        log_file = open("{pt:s}/frame-level-resutls-{it:06d}-{sq:02d}-{n:d}.log".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS)), "a", 10)
        args.det_save_dir = os.path.join(args.SAVE_ROOT, "detections-{it:02d}-{sq:02d}-{n:d}/".format(it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS)))
        args.det_file_name = "{pt:s}/frame-level-dets-{it:02d}-{sq:02d}-{n:d}.pkl".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS))
        result_file = "{pt:s}/frame-ap-results-{it:02d}-{sq:02d}-{n:d}.json".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN,n=int(100*args.GEN_NMS))
        
        if args.JOINT_4M_MARGINALS:
            log_file = open("{pt:s}/frame-level-resutls-{it:06d}-{sq:02d}-{n:d}-j4m.log".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS)), "a", 10)
            args.det_file_name = "{pt:s}/frame-level-dets-{it:02d}-{sq:02d}-{n:d}-j4m.pkl".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS))
            result_file = "{pt:s}/frame-ap-results-{it:02d}-{sq:02d}-{n:d}-j4m.json".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN,n=int(100*args.GEN_NMS))
        
        doeval = True
        if not os.path.isfile(args.det_file_name):
            logger.info('Gathering detection for ' + args.det_file_name)
            gather_framelevel_detection(args, val_dataset)
            logger.info('Done Gathering detections')
            doeval = True
        else:
            logger.info('Detection will be loaded: ' + args.det_file_name)
        
        if args.DATASET == 'road':
            label_types =  args.label_types + ['av_actions']
        elif args.DATASET == 'ucf24':
            label_types = args.label_types + ['frame_actions']
        else:
            label_types = args.label_types


        if doeval or not os.path.isfile(result_file):
            results = {}
            
            for subset in args.SUBSETS:
                if len(subset)<2:
                    continue

                sresults = evaluate_frames(val_dataset.anno_file, args.det_file_name, subset, iou_thresh=0.5, dataset=args.DATASET)
                for _, label_type in enumerate(label_types):
                    name = subset + ' & ' + label_type
                    rstr = '\n\nResults for ' + name + '\n'
                    logger.info(rstr)
                    log_file.write(rstr+'\n')
                    results[name] = {'mAP': sresults[label_type]['mAP'], 'APs': sresults[label_type]['ap_all']}
                    for ap_str in sresults[label_type]['ap_strs']:
                        logger.info(ap_str)
                        log_file.write(ap_str+'\n')
                        
                with open(result_file, 'w') as f:
                    json.dump(results, f)
        