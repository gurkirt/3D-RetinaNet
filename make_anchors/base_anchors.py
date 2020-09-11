import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np

class anchorBox(object):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, aspect_ratios =[0.5, 1 / 1., 1.5],
                    scale_ratios = [1.,]):
        super(anchorBox, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.default_sizes= [0.01, 0.06, 0.2, 0.4, 0.85]
        self.anchor_boxes = len(self.aspect_ratios)*len(self.scale_ratios)
        self.ar = self.anchor_boxes
        self.num_anchors = self.ar
        
        print(self.scale_ratios, self.ar)

    def forward(self, grid_sizes):
        anchors = []
        for k, f in enumerate(grid_sizes):
            for i, j in product(range(f), repeat=2):
                f_k = 1 / f
                # unit center x,y
                cx = (j + 0.5) * f_k
                cy = (i + 0.5) * f_k
                s = self.default_sizes[k]
                s *= s
                for ar in self.aspect_ratios:  # w/h = ar
                    h = sqrt(s / ar)
                    w = ar * h
                    for sr in self.scale_ratios:  # scale
                        anchor_h = h * sr
                        anchor_w = w * sr
                        anchors.append([cx, cy, anchor_w, anchor_h])
                        print(cx, cy, anchor_w, anchor_h)
        output = torch.FloatTensor(anchors).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output