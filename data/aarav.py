
"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import json, os
import torch
import pdb, time
import torch.utils.data as data
import pickle
from .transforms import get_image_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =   True
from PIL import Image, ImageDraw
from modules.tube_helper import make_gt_tube


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


def get_filtered_tubes(label_key, final_annots, videoname):
    
    key_tubes = final_annots['db'][videoname][label_key]
    all_labels = final_annots['all_'+label_key.replace('tubes','labels')]
    labels = final_annots[label_key.replace('tubes','labels')]
    # print(key, len(all_labels), len(labels))
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
                    # print(fn, anno_id)
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
        # print(key)
        if key.endswith('tubes'):
            filtered_tubes = get_filtered_tubes(key, final_annots, videoname)
            tubes[key] = filtered_tubes
    
    return tubes


def is_part_of_subsets(split_ids, subsets):
    
    is_it = False
    for subset in subsets:
        if subset in split_ids:
            is_it = True
    
    return is_it

def make_lists(anno_file, subsets=['train_3'], skip_step=1, full_test=False):

    # imagesDir = rootpath + imgtype + '/'
    idlist = []

    with open(anno_file,'r') as fff:
        final_annots = json.load(fff)
    
    database = final_annots['db']
    
    all_agent_labels = final_annots['all_agent_labels']
    all_action_labels = final_annots['all_action_labels']
    all_loc_labels = final_annots['all_loc_labels']
    all_duplex_labels = final_annots['all_duplex_labels']
    all_triplet_labels = final_annots['all_triplet_labels']
    
    agent_labels = final_annots['agent_labels']
    action_labels = final_annots['action_labels']
    loc_labels = final_annots['loc_labels']
    duplex_labels = final_annots['duplex_labels']
    triplet_labels = final_annots['triplet_labels']
    duplex_childs = final_annots['duplex_childs']
    triplet_childs = final_annots['triplet_childs']
    num_label_type = 5
    

    print('Len of duplex labels : all :: ', len(duplex_labels), ':', len(all_duplex_labels))
    print('Len of triplet labels : all :: ', len(triplet_labels), ':', len(all_triplet_labels))
    print(agent_labels, all_agent_labels, len(agent_labels), len(all_agent_labels))
    print('IDS to use', duplex_labels)
    counts = np.zeros((len(triplet_labels), num_label_type), dtype=np.int32)

    video_list = []
    frames_counted = 0
    tubes = {}

    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        # actidx = database[videoname]['label']
        numf = database[videoname]['numf']
        tubes[videoname] = {}
        # print(subsets)
        if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], subsets):
            continue
        # print('we got here', videoname)
        # pdb.set_trace()
        if full_test:
            tubes[videoname] = get_video_tubes(final_annots, videoname)

        frames = database[videoname]['frames']

        frame_nums = [int(f) for f in frames.keys()]
    
        if full_test:
            frame_nums = [f+1 for f in range(numf)]
        
        for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
            if frame_num % skip_step != 0:
                continue
            frame_id = str(frame_num)
            if full_test or (frame_id in frames.keys() and frames[frame_id]['annotated']>0 and len(frames[frame_id]['annos'])>0):
                # print(frame_num)
                frames_counted += 1
                # image_name = imagesDir + videoname+'/{:08d}.jpg'.format(int(frame_num))
                # assert os.path.isfile(image_name), 'Image does not exist'+image_name
                frame = {'annos':{}}
                if frame_id in frames:
                    frame = frames[frame_id]
                
                if 'annos' not in frame.keys():
                    frame = {'annos':{}}
                
                width, height = 1280, 960
                maxnum = 1
                all_boxes = []
                all_labels = []
                frame_annos = frame['annos']
                for key in frame_annos:
                    width, height = frame['width'], frame['height']
                    anno = frame_annos[key]
                    box = anno['box']
                    
                    assert box[0]<box[2] and box[1]<box[3], box
                    assert width==1280 and height==960, (width, height, box)
                    assert 0<=box[3]<=1.01, box

                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1,max(0,box[bi]))
                    
                    all_boxes.append(box)
                    agent_ids = filter_labels(anno['agent_ids'], all_agent_labels, agent_labels)
                    # assert agent_ids[0]<10, 'agent id ' + str(agent_ids[0])
                    action_ids = filter_labels(anno['action_ids'], all_action_labels, action_labels)
                    duplex_ids = filter_labels(anno['duplex_ids'], all_duplex_labels, duplex_labels)
                    triplet_ids = filter_labels(anno['triplet_ids'], all_triplet_labels, triplet_labels)
                    loc_ids = filter_labels(anno['loc_ids'], all_loc_labels, loc_labels)
                    # print(agent_ids,action_ids)
                    maxnum = max(maxnum, len(action_ids))
                    maxnum = max(maxnum, len(duplex_ids))
                    maxnum = max(maxnum, len(triplet_ids))
                    maxnum = max(maxnum, len(loc_ids))
                    box_labels = [agent_ids, action_ids, duplex_ids, triplet_ids, loc_ids]
                    all_labels.append(box_labels)

                    # for box_labels in all_labels:
                    for k, bls in enumerate(box_labels):
                        for l in bls:
                            counts[l, k] += 1     
                
                idlist.append([vid, int(frame_num), all_labels, 
                                    np.asarray(all_boxes), [width, height], maxnum])
                
    ptrstr = ''
    types = ['agents', 'actions', 'duplexes', 'triplets', 'locations']
    for k, labels in enumerate([agent_labels, action_labels, duplex_labels, triplet_labels, loc_labels]):
        for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
            ptrstr += '-'.join(subsets) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                            counts[c,k] , c, cls_)

    ptrstr += 'idlistlen' + str(len(idlist)) 
    ptrstr += ' frames_counted ' + str(frames_counted)
    
    # print(ptrstr)
    list2r = [[idlist, video_list, ptrstr, agent_labels], [action_labels, duplex_labels, triplet_labels, loc_labels, types]]
    return list2r, tubes



class Read(data.Dataset):
    """UCF24 Action Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                anno_transform=None, skip_step=1, full_test=False):

        self.dataset = args.dataset
        self.subsets, = args.subsets,
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.data_root + args.dataset + '/'
        self.anno_file  = self.root + 'annots_12fps_v1.0.json'
        self._imgpath = os.path.join(self.root, self.input_type)
        # self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        list4r, self.tubes = make_lists(self.anno_file, subsets=args.subsets, skip_step=skip_step, full_test=full_test)
        idlist, self.video_list, self.print_str, self.agent = list4r[0]
        self.action, self.duplex, self.triplet, self.loc, self.label_types  = list4r[1]
        self.num_label_type = len(self.label_types)
        
        self.ids = idlist
        #print('spacify correct subset ')
    
    
    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, index):
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        img_name = self._imgpath + '/{:s}/{:08d}.jpg'.format(videoname, frame_num)
        labels = annot_info[2]
        boxes1 = annot_info[3]
        # maxnum = annot_info[5]
        maxnum = annot_info[5]
        img = Image.open(img_name).convert('RGB')
        orig_w, orig_h = img.size
        
        if False:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            for ib in range(boxes1.shape[0]):
                bb = boxes1[ib, :].copy()
                # print(bb)
                bb[0] *= orig_w
                bb[1] *= orig_h
                bb[2] *= orig_w
                bb[3] *= orig_h
                # print(bb)
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], fill=None, outline="red")

            img.convert('RGB').save('images/{:s}-{:08d}.jpg'.format(videoname, frame_num))
        
        
        # print(img.size, wh)
        img1 = self.transform(img)
        _, height, width = img1.shape


        wh = [width, height, orig_w, orig_h]
        # print('wh', wh)
        if orig_w>1280 and width>1280:
            print('we have problem', width, height, orig_w, orig_h) 
        
        boxes = boxes1.copy()
        if boxes.shape[-1]>0:
            boxes[:, 0] *= width # width x1
            boxes[:, 2] *= width # width x2
            boxes[:, 1] *= height # height y1
            boxes[:, 3] *= height # height y2

        # print(wh)
        # print(index, 'images/{:s}/{:08d}.jpg'.format(videoname, frame_num), wh, boxes, boxes1)
        return img1, boxes, labels, index, wh, self.num_label_type, maxnum

def custum_collate(batch):
    targets = []
    images = []
    image_ids = []
    whs = []

    maxnum = 1
    boxes = []
    for sample in batch:
        images.append(sample[0])
        boxes.append(torch.FloatTensor(sample[1]))
        targets.append(sample[2])
        image_ids.append(sample[3])
        whs.append(sample[4])
        num_label_type = sample[5]
        maxnum = max(maxnum, sample[6])
    
    counts = []
    max_len = -1
    for target in boxes:
        max_len = max(max_len, target.shape[0])
        counts.append(target.shape[0])
    # new_targets = torch.zeros(len(targets), num_label_type, maxnum)
    # pdb.set_trace()
    new_boxes = torch.zeros(len(boxes), max_len, 4)
    for cc, target in enumerate(boxes):
        if boxes[cc].shape[-1]>0:
            new_boxes[cc, :counts[cc], :] = boxes[cc]

    new_targets = torch.zeros([len(boxes), max_len, num_label_type, maxnum], dtype=torch.int64) - 1

    for t, target in enumerate(targets):
        for cc in range(counts[t]):
            for nlt in range(num_label_type):
                temp = targets[t][cc][nlt] 
                # print(new_targets.shape, nlt, temp)
                if len(temp)>0:
                    new_targets[t, cc, nlt, :len(temp)] = torch.LongTensor(temp)
        
    images = get_image_list_resized(images)
    # print(image_ids)
    return images, torch.LongTensor(counts), new_boxes, new_targets, torch.LongTensor(image_ids), whs

