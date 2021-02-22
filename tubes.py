
""" 
 Build tubes and evaluate
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
from modules import utils
from modules.evaluation import evaluate_tubes
from modules.box_utils import decode, nms
from data import custum_collate, get_gt_video_list
from modules.tube_helper import nms3dt
import modules.gen_agent_paths as gen_paths #update_agent_paths, copy_live_to_dead,
from modules.tube_helper import trim_tubes
logger = utils.get_logger(__name__)

def build_eval_tubes(args, val_dataset):
    for epoch in args.EVAL_EPOCHS:
        args.det_itr = epoch
        logger.info('Building tubes at ' + str(epoch))
        log_file = open("{pt:s}/tubeing-{it:02d}-{sq:02d}.log".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN), "w", 10)
        
        args.det_save_dir = args.det_save_dir = os.path.join(args.SAVE_ROOT, "detections-{it:02d}-{sq:02d}-{n:d}/".format(it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS)))
        args.tube_save_dir = "{pt:s}/tubes-{it:02d}-{sq:02d}-{n:d}-{tk:d}-{s:s}-{io:d}-{jp:d}/".format(pt=args.SAVE_ROOT, it=epoch,  
                            sq=args.TEST_SEQ_LEN,  n=int(100*args.GEN_NMS), tk=args.TOPK, s=args.PATHS_COST_TYPE,
                            io=int(args.PATHS_IOUTH*100), jp=args.PATHS_JUMP_GAP)
        tube_file = args.tube_save_dir+ 'tubes_{}_{:d}.pkl'.format(args.TRIM_METHOD, int(args.TUBES_ALPHA*10))
        if args.JOINT_4M_MARGINALS:
            tube_file = args.tube_save_dir+ 'tubes_{}_{:d}-j4m.pkl'.format(args.TRIM_METHOD, int(args.TUBES_ALPHA*10))
            log_file = open("{pt:s}/tubeing-{it:02d}-{sq:02d}-j4m.log".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN), "w", 10)
        
        if not os.path.isdir(args.tube_save_dir):
            os.makedirs(args.tube_save_dir)
        assert os.path.isdir(args.det_save_dir), args.det_save_dir + ' detection directory does not exists '
        
        log_file.write(args.exp_name + '\n')

        tt0 = time.perf_counter()
        log_file.write('Building tubes......\n')
        
        
        paths = perform_building(args, val_dataset.video_list, epoch)
        childs = []
        if args.JOINT_4M_MARGINALS:
            childs = val_dataset.childs
        make_tubes(args, paths, val_dataset.video_list, childs, tube_file)

        # torch.cuda.synchronize()
        logger.info('Computation time {:0.2f}'.format(time.perf_counter() - tt0))

        # result_file = args.SAVE_ROOT + '/video-map-results.json'
        results = {}
        table = '\n|class'
        map_line = ['|mAP |' for _ in range(len(args.SUBSETS)*len(args.label_types[1:]))]
        metric_types = ['stiou'] #['tiou','siou','stiou']
        for metric_type in metric_types:
            for TUBES_EVAL_THRESH in args.TUBES_EVAL_THRESHS:
                table += '|{:s} {:0.02f}'.format(metric_type, TUBES_EVAL_THRESH)
                
                result_file = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}.json".format(tm=args.TRIM_METHOD, a=int(args.TUBES_ALPHA*10), pt=args.tube_save_dir, th=int(TUBES_EVAL_THRESH*100), m=metric_type)
                if args.JOINT_4M_MARGINALS:
                    result_file = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}-j4m.json".format(tm=args.TRIM_METHOD, a=int(args.TUBES_ALPHA*10), pt=args.tube_save_dir, th=int(TUBES_EVAL_THRESH*100), m=metric_type)
                
                mcount =0 
                for subset in args.SUBSETS:
                    if len(subset)<2:
                        continue
                    sresults = evaluate_tubes(val_dataset.anno_file, tube_file, dataset=args.DATASET, subset=subset, iou_thresh=TUBES_EVAL_THRESH, metric_type=metric_type)
                    for _, label_type in enumerate(args.label_types[1:]):
                        name = subset + ' & ' + label_type
                        rstr = '\n\nResults for {:s} @ {:0.02f} {:s}\n'.format(name, TUBES_EVAL_THRESH, metric_type)
                        logger.info(rstr)
                        log_file.write(rstr+'\n')
                        results[name] = {'mAP': sresults[label_type]['mAP'], 'APs': sresults[label_type]['ap_all'],
                                        'mR':sresults[label_type]['mR'], 'Recalls': sresults[label_type]['recalls'],
                                        'ap_strs': sresults[label_type]['ap_strs']}
                        map_line[mcount] += '{:0.1f}/{:0.01f}|'.format(sresults[label_type]['mAP'],sresults[label_type]['mR'])
                        mcount += 1
                        for ap_str in sresults[label_type]['ap_strs']:
                            logger.info(ap_str)
                            log_file.write(ap_str+'\n')
                            
                with open(result_file, 'w') as f:
                    json.dump(results, f)
        mcount = 0
        for subset in args.SUBSETS:
            if len(subset)<2:
                continue
            for nlt, label_type in enumerate(args.label_types[1:]):
                name = subset + ' & ' + label_type
                print(args.label_types, len(args.all_classes))
                table += '|\n'
                table += '|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n' + map_line[mcount] + '\n'
                mcount += 1
                for c, cls in enumerate(args.all_classes[nlt+1]):
                    table += '|{:s}'.format(cls)
                    for metric_type in metric_types:
                        for TUBES_EVAL_THRESH in args.TUBES_EVAL_THRESHS:
                            result_file = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}.json".format(tm=args.TRIM_METHOD, a=int(args.TUBES_ALPHA*10), pt=args.tube_save_dir, th=int(TUBES_EVAL_THRESH*100), m=metric_type)
                            if args.JOINT_4M_MARGINALS:
                                result_file = "{pt:s}/video-ap-results-{tm:s}-{a:d}-{th:d}-{m:s}-j4m.json".format(tm=args.TRIM_METHOD, a=int(args.TUBES_ALPHA*10), pt=args.tube_save_dir, th=int(TUBES_EVAL_THRESH*100), m=metric_type)
                            with open(result_file, 'r') as f:
                                results = json.load(f)
                            table += '|{:0.01f}/{:0.01f}'.format(results[name]['APs'][c],results[name]['Recalls'][c])
                    table += '|\n'
                logger.info(table) 
        log_file.close()

def perform_building(args, video_list, epoch):

    """Build agent-level tube or called paths"""

    all_paths = {}
    for videoname in video_list:
        total_dets = 0
        video_dir = args.det_save_dir + videoname + '/'
        assert os.path.isdir(
            video_dir), 'Detection should exist @ ' + video_dir
        if args.DATASET == 'ucf24':
            dirname = args.tube_save_dir + videoname.split('/')[0]
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        
        agent_path_save_name = args.tube_save_dir + videoname + '_paths.pkl'.format()
        
        if args.COMPUTE_PATHS or not os.path.isfile(agent_path_save_name):
            frames_names = os.listdir(video_dir)
            frame_ids = [int(fn.split('.')[0]) for fn in frames_names if fn.endswith('.pkl')]
            num_classes_to_use = args.num_classes_list[0] + args.num_classes_list[1]
            t1 = time.perf_counter()
            live_paths = []
            dead_paths = []
            for frame_num in sorted(frame_ids):
                save_name = '{:s}/{:05d}.pkl'.format(video_dir, frame_num)
                with open(save_name, 'rb') as f:
                    det_boxes = pickle.load(f)
                
                det_boxes = det_boxes['main']
                pickn = min(args.TOPK, det_boxes.shape[0])
                
                det_boxes = det_boxes[:args.TOPK,:]
                det_boxes = det_boxes[det_boxes[:,4]>args.CONF_THRESH,:]

                num_dets = 0
                if det_boxes.shape[0]>0:
                    frame = {}
                    frame['boxes'] = det_boxes[:,:4]
                    frame['scores'] = det_boxes[:,4]
                    frame['allScores'] = det_boxes[:,4:]
                    num_dets = det_boxes.shape[0]

                    live_paths, dead_paths = gen_paths.update_agent_paths(live_paths, 
                                            dead_paths, frame, num_classes_to_use, 
                                            iouth=args.PATHS_IOUTH, time_stamp=frame_num, 
                                            costtype=args.PATHS_COST_TYPE,
                                            jumpgap=args.PATHS_JUMP_GAP,
                                            min_len=args.PATHS_MIN_LEN)
                    
                    total_dets += num_dets

                if frame_num % 600 == 0 and frame_num>1:
                    logger.info('Time taken at fn {:d}, num dets {:d}, num live_paths {:d} time {:0.02f}'.format(frame_num, num_dets, len(live_paths), time.perf_counter() - t1))
                    t1 = time.perf_counter()
            
            paths = gen_paths.copy_live_to_dead(live_paths, dead_paths,args.PATHS_MIN_LEN)
            paths = gen_paths.fill_gaps(
                paths, min_len_with_gaps=args.PATHS_MIN_LEN, 
                minscore=args.PATHS_MINSCORE)
            ## dump agent paths to disk
            with open(agent_path_save_name,'wb') as f:
                pickle.dump(paths, f)
        else:
            with open(agent_path_save_name, 'rb') as f:
                paths = pickle.load(f)
        all_paths[videoname] = paths
    
    return all_paths

def apply_labelwise_nms(all_tubes):
    labelwise_tubes = {}
    for tube in all_tubes:
        label = 'l'+str(tube['label_id'])
        if label not in labelwise_tubes:
            labelwise_tubes[label] = [tube]
        else:
            labelwise_tubes[label].append(tube)
    det_tubes = []
    for label, ltubes in labelwise_tubes.items():
        ltubes = nms3dt(ltubes)
        for tube in ltubes:
            det_tubes.append(tube)
    return det_tubes

def make_tubes(args, paths, video_list, childs, tube_file):
    """Make tubes from paths and dump in tube_file"""
    if args.COMPUTE_TUBES or not os.path.isfile(tube_file):
        # logger.info('building agent tubes')
        detection_tubes = {}
        for ltype in args.label_types[1:]:
            detection_tubes[ltype] = {}
        for vid, videoname in enumerate(video_list):
            start_id = 1
            for nlt, ltype in  enumerate(args.label_types[1:]):
                logger.info('building tubes for '+ ltype)
                # print(args.num_classes_list, args.label_types)
                numc = args.num_classes_list[nlt+1]
                all_tubes = trim_tubes(start_id, numc, copy.deepcopy(paths[videoname]), childs, args.num_classes_list, topk=args.TUBES_TOPK, alpha=args.TUBES_ALPHA, min_len=args.TUBES_MINLEN, trim_method=args.TRIM_METHOD)
                # det_tubes = apply_labelwise_nms(all_tubes)
                detection_tubes[ltype][videoname] = all_tubes
                start_id += numc
                logger.info(str(vid+1) + '/'+ str(len(video_list)) + ' '+ str(len(detection_tubes[ltype][videoname])) +' tubes built for '+ ltype+' '+ videoname)

        with open(tube_file, 'wb') as f:
            pickle.dump(detection_tubes, f)
