
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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def make_lists(rootpath, imgtype='rgb-images', fulltest=False):

    imagesDir = rootpath + imgtype + '/'
    trainlist = []
    testlist = []

    testvideos = ['2014-06-26-09-31-18_stereo_centre_02']
    

    with open(rootpath + 'read_annots_v1.0.json','r') as fff:
        final_annots = json.load(fff)
    database = final_annots['db']
    agent_labels = final_annots['agent_labels']
    agent_action_labels = final_annots['agent_action_labels']
    action_labels = final_annots['action_labels']
    loc_labels = final_annots['loc_labels']
    num_label_type = 4

    train_counts = np.zeros((len(agent_action_labels), 4), dtype=np.int32)
    test_counts = np.zeros((len(agent_action_labels), 4), dtype=np.int32)

    video_list = []
    frames_counted = 0
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        # actidx = database[videoname]['label']
        numf = database[videoname]['numf']
        lastf = numf

        istrain = True
        if videoname in testvideos:
            istrain = False

        frames = database[videoname]['frames']

        frame_ids = [f for f in frames.keys()]
        ua
        if fulltest:
            frame_ids = [str(f+1) for f in range(numf)]
        
        for frame_num in frame_ids: #loop from start to last possible frame which can make a legit sequence

            if str(frame_num) in frames.keys() and frames[str(frame_num)]['annotated']>0 and len(frames[str(frame_num)]['annos'])>0:
                # print(frame_num)
                frames_counted += 1
                image_name = imagesDir + videoname+'/{:08d}.jpg'.format(int(frame_num))
                assert os.path.isfile(image_name), 'Image does not exist'+image_name
                frame = frames[str(frame_num)]
                width, height = frame['width'], frame['height']
                # img = Image.open(image_name).convert('RGB')
                # orig_w, orig_h = img.size
                # assert orig_w==1280 and orig_h == 960, (orig_w, orig_h)
                # print(width, height)
                maxnum = 1
                all_boxes = []
                all_labels = []
                frame_annos = frame['annos']
                for idx, key in enumerate(frame_annos.keys()):
                    anno = frame_annos[key]
                    box = anno['box']
                    
                    assert box[0]<box[2] and box[1]<box[3], box
                    assert width==1280 and height==960, (width, height, box)
                    assert 0<=box[3]<=1.01, box

                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1,max(0,box[bi]))
                    
                    all_boxes.append(box)
                
                    maxnum = max(maxnum, len(anno['action_ids']))
                    maxnum = max(maxnum, len(anno['act_agt_ids']))
                    maxnum = max(maxnum, len(anno['loc_label_ids']))
                    
                    all_labels.append([[anno['agent_id']], anno['action_ids'], 
                                    anno['act_agt_ids'], anno['loc_label_ids']])

                for box_labels in all_labels:
                    for k, bls in enumerate(box_labels):
                        for l in bls:
                            # pdb.set_trace()
                            if istrain:
                                train_counts[l, k] += 1
                            else:
                                test_counts[l, k] += 1
                if istrain:
                    trainlist.append([vid, int(frame_num), all_labels, 
                                    np.asarray(all_boxes), [width, height], maxnum])
                else:
                    testlist.append([vid, int(frame_num), all_labels, 
                                    np.asarray(all_boxes), [width, height], maxnum])
                    
            elif fulltest and not istrain: #if test video with no ground truth and fulltest is trues
                testlist.append([vid, int(frame_num), 
                [[-1] for _ in range(num_label_type)], np.zeros((1,4)), [width, height], 1])
    
    ptrstr = ''
    types = ['agents', 'actions', 'loctaions', 'agent_actions']
    for k, labels in enumerate([agent_labels, action_labels, agent_action_labels, loc_labels]):
        for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
            ptrstr += 'train {:05d} test {:05d} label: ind={:02d} name:{:s}\n'.format(
                train_counts[c,k], test_counts[c,k] , c, cls_)

    ptrstr += 'Trainlistlen' + str(len(trainlist)) + ' testlist ' 
    ptrstr += str(len(testlist)) + ' frames_counted ' + str(frames_counted)
    
    # print(ptrstr)
    list2r = [[trainlist, testlist, video_list, ptrstr, agent_labels], [action_labels, agent_action_labels, loc_labels, types]]
    return list2r



class Read(data.Dataset):
    """UCF24 Action Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                anno_transform=None, full_test=False):

        self.dataset = args.dataset
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.data_root + args.dataset + '/'
        self._imgpath = os.path.join(self.root, self.input_type)
        # self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        list4r = make_lists(self.root, self.input_type, fulltest=full_test)
        trainlist, testlist, self.video_list, self.print_str, self.agents = list4r[0]
        self.actions, self.agtacts, self.locations, self.label_types  = list4r[1]
        self.num_label_type = len(self.label_types)
        if self.train:
            self.ids = trainlist
        else:
            self.ids = testlist
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
            for ib in range(boxes.shape[0]):
                bb = boxes[ib,:].copy()
                # print(bb)
                bb[0] *= orig_w
                bb[1] *= orig_h
                bb[2] *= orig_w
                bb[3] *= orig_h
                # print(bb)
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], fill=None, outline="red")
            
            # del draw
            # img1.show()
            img.convert('RGB').save('images/{:s}-{:08d}.jpg'.format(videoname, frame_num))
        
            
            # pdb.set_trace()
        
        # print(img.size, wh)
        img1 = self.transform(img)
        _, height, width = img1.shape

        wh = [width, height, orig_w, orig_h]
        
        if orig_w>1280 and width>1280:
            print('we have problem', width, height, orig_w, orig_h) 
        
        boxes = boxes1.copy()
        boxes[:, 0] *= width # width x1
        boxes[:, 2] *= width # width x2
        boxes[:, 1] *= height # height y1
        boxes[:, 3] *= height # height y2

        # print(wh)
        print(index, 'images/{:s}-{:08d}.jpg'.format(videoname, frame_num), wh, boxes, boxes1)
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
    new_boxes = torch.zeros(len(boxes), max_len, boxes[0].shape[1])
    for cc, target in enumerate(boxes):
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

