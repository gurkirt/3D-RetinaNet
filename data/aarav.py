
"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import json, os
import torch
import pdb, time
import torch.utils as tutils
import pickle
from .transforms import get_clip_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =   True
from PIL import Image, ImageDraw
from modules.tube_helper import make_gt_tube
import random as random
from modules import utils 

logger = utils.get_logger(__name__)

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids


def get_gt_video_list(anno_file, SUBSETS):
    """Get video list form ground truth videos used in subset 
    and their ground truth tubes """

    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    video_list = []
    for videoname in final_annots['db']:
        if is_part_of_subsets(final_annots['db'][videoname]['split_ids'], SUBSETS):
            video_list.append(videoname)

    return video_list


def get_filtered_tubes(label_key, final_annots, videoname):
    
    key_tubes = final_annots['db'][videoname][label_key]
    all_labels = final_annots['all_'+label_key.replace('tubes','labels')]
    labels = final_annots[label_key.replace('tubes','labels')]
    filtered_tubes = []
    for _ , tube in key_tubes.items():
        label_id = tube['label_id']
        label = all_labels[label_id]
        if label in labels:
            new_label_id = labels.index(label)
            # temp_tube = GtTube(new_label_id)
            frames = []
            boxes = []
            if 'annos' in tube.keys():
                for fn, anno_id in tube['annos'].items():
                    frames.append(int(fn))
                    anno = final_annots['db'][videoname]['frames'][fn]['annos'][anno_id]
                    box = anno['box'].copy()
                    box[0] *= 800
                    box[2] *= 800
                    box[3] *= 600
                    box[1] *= 600
                    boxes.append(box)
            else:
                for fn in tube['frames']:
                    frames.append(int(fn))

            temp_tube = make_gt_tube(frames, boxes, new_label_id)
            filtered_tubes.append(temp_tube)
            
    return filtered_tubes


def get_video_tubes(final_annots, videoname):
    
    tubes = {}
    for key in final_annots['db'][videoname].keys():
        if key.endswith('tubes'):
            filtered_tubes = get_filtered_tubes(key, final_annots, videoname)
            tubes[key] = filtered_tubes
    
    return tubes


def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    
    return is_it


class Read(tutils.data.Dataset):
    """
    AARAV Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                skip_step=1, full_test=False):

        self.DATASET = args.DATASET
        self.SUBSETS = args.SUBSETS
        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        self.MULIT_SCALE = args.MULIT_SCALE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.full_test = full_test
        self.skip_step = skip_step
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.DATA_ROOT + args.DATASET + '/'
        self.anno_file  = self.root + 'annots_12fps_full_v1.0.json'
        self._imgpath = os.path.join(self.root, self.input_type)
        # self.image_sets = image_sets
        self.transform = transform
        self.ids = list()
        self._make_lists()        
        self.num_label_types = len(self.label_types)

        
    def __len__(self):
        return self.BATCH_SIZE*500


    def _make_lists(self): #anno_file, SUBSETS=['train_3'], skip_step=1, full_test=False):

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        
        num_label_type = 5
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(final_annots[name+'_labels'])))
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = final_annots['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros((len(final_annots[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)

        video_list = []
        tubes = {}
        frame_level_list = []
        for vid, videoname in enumerate(sorted(database.keys())):
            
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = database[videoname]['numf']
            video_list.append([videoname,numf])
            
            frames = database[videoname]['frames']
            numf = database[videoname]['numf']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                # if frame_num % self.skip_step != 0:
                #     continue
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    # frame = {'annos':{}}
                    # if frame_id in frames:
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1280 and height==960, (width, height, box)

                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  
        
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        self.label_types = ['agentness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.video_list = video_list
        self.num_videos = len(video_list)
        self.print_str = ptrstr
        
    
    def __getitem__(self, index):
        video_id = random.randint(0,self.num_videos-1)
        [videoname, numf] = self.video_list[video_id]
        step_size = random.randint(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP)
        max_start_frame = numf-step_size*self.SEQ_LEN-2
        start_frame = random.randint(0,max_start_frame)
        images = []
        frame_num = start_frame
        ego_labels = np.zeros(self.SEQ_LEN)-1
        all_boxes = []
        labels = []
        ego_labels = []
        mask = np.zeros(self.SEQ_LEN, dtype=np.int)
        indexs = []
        for i in range(self.SEQ_LEN):
            img_name = self._imgpath + '/{:s}/{:08d}.jpg'.format(videoname, frame_num+1)
            img = Image.open(img_name).convert('RGB')
            images.append(img)
            frame_num += step_size
            if self.frame_level_list[video_id][frame_num]['labeled']:
                mask[i] = 1
                all_boxes.append(self.frame_level_list[video_id][frame_num]['boxes'].copy())
                labels.append(self.frame_level_list[video_id][frame_num]['labels'].copy())
                ego_labels.append(self.frame_level_list[video_id][frame_num]['ego_label'])
            else:
                all_boxes.append(np.asarray([]))
                labels.append(np.asarray([]))
                ego_labels.append(-1)
            indexs.append([video_id, frame_num])

        clip = self.transform(images)
        height, width = clip.shape[-2:]
        wh = [height, width]
        
        for bb, boxes in enumerate(all_boxes):
            if boxes.shape[0]>0:
                if boxes[0,0]>1:
                    print(bb, videoname)
                    pdb.set_trace()
                boxes[:, 0] *= width # width x1
                boxes[:, 2] *= width # width x2
                boxes[:, 1] *= height # height y1
                boxes[:, 3] *= height # height y2

        return clip, all_boxes, labels, ego_labels, indexs, wh, self.num_classes


def custum_collate(batch):
    
    images = []
    boxes = []
    targets = []
    ego_targets = []
    image_ids = []
    whs = []
    
    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        targets.append(sample[2])
        ego_targets.append(torch.LongTensor(sample[3]))
        image_ids.append(torch.LongTensor(sample[4]))
        whs.append(torch.LongTensor(sample[5]))
        num_classes = sample[6]
        
    counts = []
    max_len = -1
    seq_len = len(boxes[0])
    for bs_ in boxes:
        temp_counts = []
        for bs in bs_:
            max_len = max(max_len, bs.shape[0])
            temp_counts.append(bs.shape[0])
        assert seq_len == len(temp_counts)
        counts.append(temp_counts)
    counts = np.asarray(counts, dtype=np.int)
    new_boxes = torch.zeros(len(boxes), seq_len, max_len, 4)
    new_targets = torch.zeros([len(boxes), seq_len, max_len, num_classes])
    for c1, bs_ in enumerate(boxes):
        for c2, bs in enumerate(bs_):
            if counts[c1,c2]>0:
                assert bs.shape[0]>0, 'bs'+str(bs)
                new_boxes[c1, c2, :counts[c1,c2], :] = torch.from_numpy(bs)
                targets_temp = targets[c1][c2]
                assert targets_temp.shape[0] == bs.shape[0], 'num of labels and boxes should be same'
                new_targets[c1, c2, :counts[c1,c2], :] = torch.from_numpy(targets_temp)

    # images = torch.stack(images, 0)
    images = get_clip_list_resized(images)
    return images, new_boxes, new_targets, torch.stack(ego_targets,0), \
            torch.LongTensor(counts), torch.stack(image_ids,0), torch.stack(whs,0)
    
