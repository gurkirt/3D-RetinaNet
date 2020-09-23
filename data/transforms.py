import torch
from torchvision.transforms import functional as F
import math


# modified from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/structures/image_list.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def get_clip_list_resized(tensors):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    stride = 32
    max_size = list(max_size)
    max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
    max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).zero_()
    
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)

    # image_sizes = [im.shape[-2:] for im in tensors]
    return batched_imgs


# from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/data/transforms/transforms.py 
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.stride = 32
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
                oh = int(math.floor(oh / self.stride) * self.stride)
            else:
                oh = size
                ow = int(size * w / h)
                ow = int(math.floor(ow / self.stride) * self.stride)
            # print('owoh', size, ow, oh)

            return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class ResizeClip(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.stride = 32
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

    def __call__(self, clip):
        size = self.get_size(clip[0].size)
        clip = [F.resize(image, size) for image in clip]
        return clip


class ToTensorStack(object):
    
    """
    
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    """

    def __call__(self, clip):
        """
        Args:
            pic (PIL Images ): List of images to be converted to tensor and stack along time in dimension 1 not 0.
        Returns:
            Tensor: Converted clip into (C x T x H x W).
        """
        stacked_clip =  torch.stack([F.to_tensor(img) for img in clip], 1)
        # print('stacked_clip, shape', stacked_clip.shape)
        return stacked_clip

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            tensor (Tensor): Tensor image of size (C x T x H x W) to be normalized.

        Returns:
            Tensor: Normalized Tensor (C x T x H x W).
        """
        
        for i in range(len(self.mean)):
            clip[i] = (clip[i] - self.mean[i])/ self.std[i]
        # print('after norm ', clip.shape)
        return clip
