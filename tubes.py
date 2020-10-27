
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
import torch.utils.data as data_utils
from modules import utils
from modules.evaluation import evaluate_tubes
from modules.box_utils import decode, nms
from data import custum_collate, get_gt_video_list
import modules.gen_agent_paths as gen_paths #update_agent_paths, copy_live_to_dead,
from modules.tube_helper import trim_tubes
logger = utils.get_logger(__name__)

def build_eval_tubes(args, val_dataset):
    for epoch in args.EVAL_EPOCHS:
        args.det_itr = epoch
        logger.info('Building tubes at ' + str(epoch))
        log_file = open("{pt:s}/tubeing-{it:02d}-{sq:02d}.log".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN), "w", 10)
        args.det_save_dir = "{pt:s}/detections-{it:02d}-{sq:02d}/".format(pt=args.SAVE_ROOT, it=epoch,  sq=args.TEST_SEQ_LEN)
        args.tube_save_dir = "{pt:s}/tubes-{it:02d}-{sq:02d}/".format(pt=args.SAVE_ROOT, it=epoch,  sq=args.TEST_SEQ_LEN)
        if not os.path.isdir(args.tube_save_dir):
            os.makedirs(args.tube_save_dir)
        assert os.path.isdir(args.det_save_dir), args.det_save_dir + ' detection directory does not exists '
        
        log_file.write(args.exp_name + '\n')

        tt0 = time.perf_counter()
        log_file.write('Building tubes......\n')
        
        tube_file = args.tube_save_dir+ 'tubes.json'

        # video_list = get_gt_video_list(val_dataset.anno_file, args.SUBSETS)

        # if args.COMPUTE_TUBES:
        paths = perform_building(args, val_dataset.video_list, epoch)
        make_tubes(args, paths, val_dataset.video_list, tube_file)

        # torch.cuda.synchronize()
        logger.info('Computation time {:0.2f}'.format(time.perf_counter() - tt0))

        # result_file = args.SAVE_ROOT + '/video-map-results.json'
        result_file = "{pt:s}/video-ap-results-{it:02d}-{sq:02d}.json".format(pt=args.SAVE_ROOT, it=epoch, sq=args.TEST_SEQ_LEN)
        
        results = {}
        for subset in args.SUBSETS:
            if len(subset)<2:
                continue
            sresults = evaluate_tubes(val_dataset.anno_file, tube_file, dataset=args.DATASET, subset=subset, iou_thresh=args.TUBES_EVAL_THRESH)
            for _, label_type in enumerate(args.label_types[1:]):
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

        logger.info('Results are saved at '+ result_file)
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
        agent_path_save_name = args.tube_save_dir + videoname + '_agent_paths.pkl'
        if args.COMPUTE_PATHS or not os.path.isfile(agent_path_save_name):
            frames_names = os.listdir(video_dir)
            frame_ids = [int(fn.split('.')[0]) for fn in frames_names if fn.endswith('.pkl')]
            num_classes_to_use = args.num_classes_list[0] + args.num_classes_list[1]
            t1 = time.perf_counter()
            live_paths = []
            dead_paths = []
            for frame_num in sorted(frame_ids):
                save_name = '{:s}/{:08d}.pkl'.format(video_dir, frame_num)
                with open(save_name, 'rb') as f:
                    det_boxes = pickle.load(f)
                
                det_boxes = det_boxes['main']
                det_boxes = utils.filter_detections_for_tubing(args, torch.from_numpy(det_boxes[:,4]), torch.from_numpy(det_boxes[:,:4]), torch.from_numpy(det_boxes[:,4:]))
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
                                            jumpgap=args.PATHS_JUMP_GAP)
                    
                    total_dets += num_dets

                if frame_num % 600 == 0 and frame_num>1:
                    logger.info('Time taken at fn {:d}, num dets {:d}, num live_paths {:d} time {:0.02f}'.format(frame_num, num_dets, len(live_paths), time.perf_counter() - t1))
                    t1 = time.perf_counter()
            
            paths = gen_paths.copy_live_to_dead(live_paths, dead_paths)
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


def make_tubes(args, paths, video_list, tube_file):
    """Make tubes from paths and dump in tube_file"""
    if args.COMPUTE_TUBES or not os.path.isfile(tube_file):
        # logger.info('building agent tubes')
        detection_tubes = {}
        for ltype in args.label_types[1:]:
            detection_tubes[ltype] = {}
        for vid, videoname in enumerate(video_list):
            start_id = 1
            for nlt, ltype in  enumerate(args.label_types[1:]):
                # logger.info('building tubes for '+ ltype)
                numc = args.num_classes_list[nlt+1]
                detection_tubes[ltype][videoname] = trim_tubes(
                    start_id, numc, paths[videoname], topk=args.TUBES_TOPK, alpha=args.TUBES_ALPHA, min_len=args.TUBES_MINLEN)
                start_id += numc
                logger.info(str(vid+1) + '/'+ str(len(video_list)) + ' '+ str(len(detection_tubes[ltype][videoname])) +' tubes built for '+ ltype+' '+ videoname)

        with open(tube_file, 'w') as f:
                json.dump(detection_tubes, f)
