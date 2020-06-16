import numpy as np
import pdb


def make_det_tube(scores, boxes, frames, label_id):
    tube = {}
    tube['label_id'] =label_id
    tube['scores'] = np.asarray(scores)
    tube['boxes'] = np.asarray(boxes)
    tube['score'] = np.mean(scores)
    tube['frames'] = np.asarray(frames)
    # assert tube['frames'].shape[0] == tube['boxes'].shape[0], 'must be equal'
    return tube

def get_nonnp_det_tube(scores, boxes, start, end, label_id):
    tube = {}
    tube['label_id'] =label_id
    tube['scores'] = scores.tolist()
    tube['boxes'] = boxes.tolist()
    tube['score'] = float(np.mean(scores))
    tube['frames'] = [i for i in range(start, end)]
    assert len(tube['frames']) == len(tube['boxes']), 'must be equal'

    return tube

def make_gt_tube(frames, boxes, label_id):
    frames = np.asarray(frames)
    indexs = np.argsort(frames)
    frames = frames[indexs]
    boxes = np.asarray(boxes)
    if boxes.shape[0]>0:
        boxes = boxes[indexs,:]
    tube = {}
    tube['frames'] = frames
    tube['boxes'] = boxes
    tube['label_id'] = label_id
    return tube

def trim_tubes(start_id, numc, paths, topk=3, score_thresh=0.1, alpha=3, min_len=4):
    """ Trim the paths into tubes using DP"""
    tubes = []
    for path in paths:
        allScores = path['allScores'][:,start_id:start_id+numc]
        path_start_frame = path['foundAt'][0]
        if allScores.shape[0]<min_len:
            continue
        # topk_classes = get_topk_classes(allScores, topk)
        score_mat = np.transpose(allScores.copy())
        for _ in range(topk):
            (segments, _) = dpEMmax(score_mat, alpha)
            # print(segments)
            labels, starts, ends = getLabels(segments)
            # print(labels, starts, ends)
            for i in range(len(labels)):
                if ends[i] - starts[i] >= min_len:
                    scores = score_mat[labels[i], starts[i]:ends[i]+1]
                    boxes = path['boxes'][starts[i]:ends[i]+1, :]
                    start = starts[i] + path_start_frame
                    end = ends[i] + path_start_frame + 1
                    tube = get_nonnp_det_tube(scores, boxes, int(start), int(end), int(labels[i]))
                    tubes.append(tube)
                    score_mat[labels[i], starts[i]:ends[i]+1] = 0.0
            
    return tubes

def getLabels(segments, cls=1):
    starts = np.zeros(len(segments), dtype='int32')
    ends = np.zeros(len(segments), dtype='int32')
    labels = np.zeros(len(segments), dtype='int32')
    fl = 0
    i=0
    starts[i]=0
    fl = segments[0]
    labels[i] =  segments[0]
#    print segments[0]
#    pdb.set_trace()
    for ii in range(len(segments)):
        if abs(segments[ii] -fl)>0:
            ends[i]=ii-1
            fl = segments[ii]
            i+=1
            starts[i]=ii
            labels[i] = fl
    ends[i] = len(segments)-1
    return labels[:i+1],starts[:i+1],ends[:i+1]

def get_topk_classes(allScores, topk):
    scores = np.sum(allScores, axis=0)
    sorted_classes =np.argsort(-scores)
    return sorted_classes[:topk]


def dpEMmax(M, alpha=3):
    (r,c) = np.shape(M)
    D = np.zeros((r, c+1)) # add an extra column
    D[:,0] = 1 # % put the maximum cost
    D[:, 1:(c+1)] = M
    phi = np.zeros((r,c))
    for j in range(1,c):
        for i in range(r):
            v1 = np.ones(r)*alpha
            v1[i] = 0
            values= D[:, j-1] - v1
            tb = np.argmax(values)
            dmax = max(values)
            D[i,j] = D[i,j]+dmax
            phi[i,j] = tb

    q = c-1
    values= D[:, c-1]
    p = np.argmax(values)
    i = p
    j = q 
    ps = np.zeros(c)
    ps[q] = p
    while j>0:
        tb = phi[i,j]
        j = int(j-1)
        q = j
        ps[q] = tb
        i = int(tb)
    
    D = D[:,1:]
    return (ps,D)


def intersect(box_a, box_b):
        # A = box_a.size(0)
        B = box_b.shape[0]
        inters = np.zeros(B)
        for b in range(B):
            max_x = min(box_a[2], box_b[b, 2])
            max_y = min(box_a[3], box_b[b, 3])
            min_x = max(box_a[0], box_b[b, 0])
            min_y = max(box_a[1], box_b[b, 1])
            inters[b] = (max_x-min_x)*(max_y-min_y)
        return inters


def bbox_overlaps(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    B = box_b.shape[0]
    ious = np.zeros(B)
    for b in range(B):
        if inter[b]>0:
            area_b = (box_b[b,2] - box_b[b,0]) * (box_b[b,3] - box_b[b,1])
            union = area_a + area_b - inter[b]
            ious[b] = inter[b]/union
    return ious


def get_tube_3Diou(tube_a, tube_b , spatial_only=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(tube_a['frames'][0], tube_b['frames'][0])
    tmax = min(tube_a['frames'][-1], tube_b['frames'][-1])
    
    if tmax < tmin: return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(tube_a['frames'][-1], tube_b['frames']
                         [-1]) - min(tube_a['frames'][0], tube_b['frames'][0]) + 1

    # try:
    tube_a_boxes = tube_a['boxes'][int(np.where(tube_a['frames'] == tmin)[0][0]): int(
        np.where(tube_a['frames'] == tmax)[0][0]) + 1, :]
    tube_b_boxes = tube_b['boxes'][int(np.where(tube_b['frames'] == tmin)[0][0]): int(
        np.where(tube_b['frames'] == tmax)[0][0]) + 1, :]
    # except:
    #     pdb.set_trace()     print('something', tube_a_boxes, tube_b_boxes, iou)

    iou = iou3d(tube_a_boxes, tube_b_boxes)

    if spatial_only:
        return iou
    else:
        return  iou * temporal_inter / temporal_union


def iou3d(tube_a, tube_b):
    """Compute the IoU between two tubes with same temporal extent"""

    assert tube_a.shape[0] == tube_b.shape[0]
    # assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(tube_a,tube_b)

    return np.mean(ov / (area2d(tube_a) + area2d(tube_b) - ov) )


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]+1) * (b[:,3]-b[:,1]+1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2] + 1, b2[:,2] + 1)
    ymax = np.minimum(b1[:,3] + 1, b2[:,3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def nms3dt(tubes, overlap=0.5):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([get_tube_3Diou(tubes[ii][0], tubes[i][0])
                         for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]
