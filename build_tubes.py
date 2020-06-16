
""" 

 Build tubes and evaluate

"""

import os
import time, json
import datetime
import numpy as np
import torch
import pdb
import pickle as pickle
import torch.utils.data as data_utils
from modules import utils
from modules.evaluation import evaluate_tubes
from modules.box_utils import decode, nms
from data import custum_collate
import modules.gen_agent_paths as gen_paths #update_agent_paths, copy_live_to_dead,
from modules.tube_helper import trim_tubes

def build_tubes(args, val_dataset):
    for iteration in args.eval_iters:
        args.det_itr = iteration
        print('Testing at ', iteration)
        log_file = open("{pt:s}/tubeing-{it:06d}.log".format(pt=args.save_root, it=iteration, date=datetime.datetime.now()), "w", 10)
        args.det_save_dir = "{pt:s}/detections-{it:06d}/".format(pt=args.save_root, it=iteration)
        assert os.path.isdir(args.det_save_dir), args.det_save_dir + ' detection directory does not exists '
        
        log_file.write(args.exp_name + '\n')
        
        # args.model_path = args.save_root + 'model_{:06d}.pth'.format(iteration)
        # log_file.write(args.model_path+'\n')
        # net.load_state_dict(torch.load(args.model_path))
        # print('Finished loading model %d !' % iteration )
        # torch.cuda.synchronize()

        tt0 = time.perf_counter()
        log_file.write('Building tubes......\n')
        
        # net.eval() # switch net to evaluation mode        
        # mAP, ap_all, ap_strs = perform_test(args, net, val_data_loader, val_dataset, iteration)
        
        tube_file = args.save_root + 'tubes.json'

        
        paths = perform_building(args, val_dataset, iteration, tube_file)
        
        if args.compute_tubes:
            make_tubes(args, paths, tube_file)
        
        torch.cuda.synchronize()
        print('Computation time {:0.2f}'.format(time.perf_counter() - tt0))

        result_file = args.save_root + '/video-map-results.json'
        results = {}
        for nlt, label_type in enumerate(args.label_types) :
            for subset in args.test_subsets:
                mAP, aps, ap_strs = evaluate_tubes(
                    val_dataset.anno_file, tube_file, args.all_classes[nlt], label_type, subset, iou_thresh=0.2)
                name = subset + '-' + label_type
                rstr = '\n\nResults for ' + name + '\n\n'
                print(rstr)
                log_file.write(rstr+'\n')
                results[name] = {'mAP': mAP, 'APs': aps}
                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str+'\n')
        log_file.close()
        # return  aps, aps_all, ap_strs
        with open(result_file, 'w') as f:
            json.dump(results, f)


def perform_building(args, video_list, iteration, tube_file):

    """Test a FPN network on an image database."""
    
    # opts
    jumpgap = 2
    iouth=0.2
    costtype='scoreiou'
    args.do_3d_plot = True
    min_len = 4
    
    for videoname in video_list:
        total_dets = 0
        video_dir = args.det_save_dir + videoname + '/'
        assert os.path.isdir(
            video_dir), 'Detection should exist @ ' + video_dir

        frames_names = os.listdir(video_dir)
        frame_ids = [int(fn.split('.')[0]) for fn in frames_names if fn.endswith('.npy')]
        num_agent = args.num_agent
        t1 = time.perf_counter()
        
        agent_path_save_name = args.det_save_dir + videoname + '_agent_paths.pkl'
        
        if args.compute_paths or not os.path.isfile(agent_path_save_name):
            live_paths = []
            dead_paths = []
            for frame_num in sorted(frame_ids):
                
                save_name = '{:s}/{:08d}.npy'.format(video_dir, frame_num)
                det_boxes = np.load(save_name)
                num_dets = 0
                if det_boxes.shape[0]>0:
                    frame = {}
                    frame['boxes'] = det_boxes[:,:4]
                    frame['scores'] = det_boxes[:,4]
                    frame['allScores'] = det_boxes[:,5:]
                    num_dets = det_boxes.shape[0]

                    live_paths, dead_paths = gen_paths.update_agent_paths(live_paths, 
                                            dead_paths, frame, num_agent, 
                                            iouth=iouth, time_stamp=frame_num, costtype=costtype, 
                                            jumpgap=jumpgap)
                    
                    total_dets += num_dets

                if frame_num % 600 == 0 and frame_num>1:
                    print('Time taken at fn ', frame_num, num_dets, len(live_paths), time.perf_counter() - t1)
                    t1 = time.perf_counter()
            
            paths = gen_paths.copy_live_to_dead(live_paths, dead_paths)
            paths = gen_paths.fill_gaps(paths, min_len_with_gaps=min_len*1.6)
            ## dump agent paths to disk
            with open(agent_path_save_name,'wb') as f:
                pickle.dump(paths, f)
        else:
            with open(agent_path_save_name, 'rb') as f:
                paths = pickle.load(f)

    return paths


def make_tubes(args, paths, tube_file, min_len=4):
    
    detection_tubes = {}
    for ltype in args.label_types:
        detection_tubes[ltype] = {}
    print('building agent tubes')
    start_id = 1
    agrs_vars = vars(args)
    for ltype in args.label_types:
            print('building tubes for ', ltype)
            numc = agrs_vars['num_' + ltype]
            detection_tubes[ltype][videoname] = trim_tubes(
                start_id, numc, paths, topk=3, score_thresh=0.1, alpha=3, min_len=min_len)
            start_id += numc
            print(len(detection_tubes[ltype][videoname]), ltype+' tubes built')

    with open(tube_file, 'w') as f:
            json.dump(detection_tubes, f)

# if __name__ == '__main__':
    # main()
