import random
import numbers
import collections
import numpy as np
import torch
from skimage.transform import resize


class ComposeEvents(object):
    """
    Composes several transforms together.
    Args:
        transforms (list of Transform objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __getitem__(self, item):
        return self.transforms[item]

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class GroupCenterCropEvents(object):
    def __init__(self, size):
        self.worker = CenterCropEvents(size)

    def __call__(self, img_group):
        return np.stack([self.worker(img) for img in img_group], axis=0)


class GroupMultiScaleCropEvents(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):

        # fixed size then padding is added
        c = img_group[0].shape[0]
        im_size = (img_group[0].shape[1], img_group[0].shape[2])
        # pad_img_group = [self.padding(img, im_size[0], im_size[1]) for img in img_group]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img[:, offset_h:offset_h + crop_h, offset_w:offset_w + crop_w] for img in img_group]
        ret_img_group = [resize(img, (c, self.input_size[0], self.input_size[1]), anti_aliasing=True)
                         for img in crop_img_group]
        return np.stack(ret_img_group, axis=0)

    def _sample_crop_size(self, im_size):
        image_h, image_w = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    @staticmethod
    def padding(array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desired width
        :return: padded array
        """

        h = array.shape[1]
        w = array.shape[2]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        return np.pad(array, pad_width=((0, 0), (a, aa), (b, bb)), mode='constant')


class ToTensorEvents(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (C x H x W) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    NOTICE: here C is equal to the input channel x number of frames for the action
    """

    def __init__(self):
        pass

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        img = torch.from_numpy(pic).contiguous()
        return img

    """
    import pdb; pdb.set_trace()
    list o nome variabili per vederle 
    n per andare avanti riga per riga
    """

    def randomize_parameters(self):
        pass

class RandomInvertVoxels(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Invert voxel values from positive to negative and viceversa
    """
    def __init__(self):
        self.p = 0.5

    def __call__(self, img_group):
        v = random.random()
        if v < self.p:
            img_group = [img * (-1) for img in img_group]
            return img_group
        else:
            return img_group


    """
    import pdb; pdb.set_trace()
    list o nome variabili per vederle 
    n per andare avanti riga per riga
    """

    def randomize_parameters(self):
        pass

class InvertVoxels(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Invert voxel values from positive to negative and viceversa
    """
    def __init__(self):
        pass

    def __call__(self, img_group):
        return img_group * (-1)

    """
    import pdb; pdb.set_trace()
    list o nome variabili per vederle 
    n per andare avanti riga per riga
    """

    def randomize_parameters(self):
        pass


class NormalizeGroupEvents(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean and std over all x channels
    will normalize each channel of the numpy.ndarray, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for all channels respecitvely.
        std (sequence): Sequence of standard deviations for all channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_group):
        """
        Args:
            image_group (numpy.ndarray): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        return np.stack([img.sub(self.mean).div(self.std) for img in image_group], axis=0)

    def randomize_parameters(self):
        pass


class GroupRandomHorizontalFlipEvents(object):
    """Randomly horizontally flips the given numpy with a probability 0.5
    """

    def __init__(self):
        self.p = 0.5

    def __call__(self, img_group):
        v = random.random()
        if v < self.p:
            ret = [np.flip(img, axis=1) for img in img_group]
            return np.stack(ret, axis=0)
        else:
            return img_group


class StackEvents(object):
    def __init__(self, args, input_mean, input_std, range):
        self.args = args
        self.input_mean = input_mean
        self.input_std = input_std
        self.range = range

    def __call__(self, pic):
        # img_size is n_frames_per_clip x C x H x W
        pic_size = pic.shape

        if self.args.channels_events == 1:
            pic = np.mean(pic, axis=1)

        img_npy = pic.clip(-0.5, 0.5)

        if self.args.normalize_events:
            # rescale into the range

            img_npy = (img_npy - img_npy.min())/(img_npy.max() - img_npy.min()) * (self.range[1] - self.range[0]) + self.range[0]
            # normalize according to the pretrained
            if self.args.channels_events != 1:
                img_npy = np.stack([(img - np.tile(self.input_mean[:, np.newaxis, np.newaxis], (1, img.shape[1], img.shape[2])))
                                / np.tile(self.input_std[:, np.newaxis, np.newaxis], (1, img.shape[1], img.shape[2])) for img in img_npy], axis=0).astype(np.float32)
            else:
                img_npy = np.stack(
                    [(img - self.input_mean)
                     / self.input_std for img in
                     img_npy], axis=0).astype(np.float32)
        return np.reshape(img_npy, (-1, pic_size[-2], pic_size[-1]))


# Transformations on single image
class ScaleEvents(object):
    """Rescale the input numpy to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (ndarray): Image to be scaled.
        Returns:
            ndarray: Rescaled image.
        """
        c, h, w = img.shape
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if h < w:
                ow = self.size
                oh = int(self.size * h / w)
                img = resize(img, (c, oh, ow), anti_aliasing=True)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                img = resize(img, (c, oh, ow), anti_aliasing=True)
        else:
            img = resize(img, (c, self.size[0], self.size[1]), anti_aliasing=True)

        return img

    def randomize_parameters(self):
        pass


class CenterCropEvents(object):
    """Crops the given numpy image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy): Image to be cropped.
        Returns:
            numpy image: Cropped image.
        """
        c, h, w = img.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[:, y1:y1 + th, x1:x1 + tw]

    def randomize_parameters(self):
        pass


class RandomCropEvents(object):
    """Randomly crop the given numpy image.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.top = 0
        self.left = 0

    def __call__(self, img):
        """
        Returns:
            numpy: Cropped numpy image.
        """
        c, h, w = img.shape
        th, tw = self.size

        self.top = random.randint(0, h - th)
        self.left = random.randint(0, w - tw)

        img = img[:, self.top: self.top + th, self.left:self.left + th]
        return img

    def randomize_parameters(self):
        pass


class RandomHorizontalFlipEvents(object):
    """Horizontally flip the given numpy randomly with a probability"""

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (ndarray): Image to be flipped.
        Returns:
            ndarray: Randomly flipped image.
        """
        rand = random.random()
        if rand < self.p:
            img = np.flip(img, axis=1)
        return img

    def randomize_parameters(self):
        self.p = random.random()
