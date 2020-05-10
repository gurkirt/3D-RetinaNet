import torch
from torchvision.transforms import functional as F
import math


# modified from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/structures/image_list.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def get_image_list_resized(tensors):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    stride = 32
    max_size = list(max_size)
    max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
    max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).zero_()
    
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    # image_sizes = [im.shape[-2:] for im in tensors]
    return batched_imgs


# from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/data/transforms/transforms.py 
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        
        if self.min_size == self.max_size:
        
            return (self.min_size, self.max_size)
        
        else:
            w, h = image_size
            size = self.min_size
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


