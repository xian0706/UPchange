from albumentations import DualTransform, to_tuple, CLAHE, ShiftScaleRotate, OpticalDistortion
from albumentations.augmentations.functional import _brightness_contrast_adjust_uint, clahe
import random
import numpy as np
import cv2

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


class TemporalSwap(DualTransform):
    def apply_to_masks(self, masks, **params):
        masks = masks[::-1]
        return masks

    def apply(self, img, **params):
        img = img[:, :, (3, 4, 5, 0, 1, 2)]
        return img


class ChannelShift(DualTransform):
    """Randomly shift values for each channel of the input RGB image.
        reference RGBshift
        Image types:
            uint8
        """

    def __init__(self, channel_shift_limit=20, always_apply=False, p=0.5):
        super(ChannelShift, self).__init__(always_apply, p)
        self.c_shift_limit = to_tuple(channel_shift_limit)

    def apply_to_masks(self, masks, **params):
        return masks

    def apply(self, img, **params):
        _, _, C = img.shape
        result_img = np.empty_like(img)
        for c in range(0, C):
            shift = random.randint(self.c_shift_limit[0], self.c_shift_limit[1])
            result_img[..., c] = img[..., c] + shift

        return result_img

    def get_transform_init_args_names(self):
        return ("c_shift_limit")


class RandomBrightnessContrastv2(DualTransform):
    """Randomly change brightness and contrast of the input image.
        reference RandomBrightnessContrast
        Image types:
            uint8
        """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        super(RandomBrightnessContrastv2, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply_to_masks(self, masks, **params):
        return masks

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        result_img = np.empty_like(img)

        img1 = img[:, :, :3]
        img2 = img[:, :, 3:]
        result_img[:, :, :3] = _brightness_contrast_adjust_uint(img1, alpha, beta, self.brightness_by_max)
        result_img[:, :, 3:] = _brightness_contrast_adjust_uint(img2, alpha, beta, self.brightness_by_max)


        return result_img

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class CLAHEV2(DualTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.
    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (0, clip_limit). Default: (0, debug).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CLAHEV2, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply_to_masks(self, masks, **params):
        return masks

    def apply(self, img, clip_limit=2, **params):
        result_img = np.empty_like(img)

        img1 = img[:, :, :3]
        img2 = img[:, :, 3:]
        result_img[:, :, :3] = clahe(img1, clip_limit, self.tile_grid_size)
        result_img[:, :, 3:] = clahe(img2, clip_limit, self.tile_grid_size)

        return result_img

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


class OpticalDistortionV2(DualTransform):

    def __init__(
            self,
            distort_limit=0.05,
            shift_limit=0.05,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
            mask_value=None,
            always_apply=False,
            p=0.5,
    ):
        super(OpticalDistortionV2, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        result_img = np.empty_like(img)

        img1 = img[:, :, :3]
        img2 = img[:, :, 3:]
        result_img[:, :, :3] = optical_distortion(img1, k, dx, dy, interpolation, self.border_mode, self.value)
        result_img[:, :, 3:] = optical_distortion(img2, k, dx, dy, interpolation, self.border_mode, self.value)
        return result_img

    def apply_to_masks(self, masks, k=0, dx=0, dy=0, **params):
        result_img = []
        img1 = masks[0]
        img2 = masks[1]
        result_img.append(optical_distortion(img1, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value))
        result_img.append(optical_distortion(img2, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value))
        return result_img

    def get_params(self):
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }

    def get_transform_init_args_names(self):
        return ("distort_limit", "shift_limit", "interpolation", "border_mode", "value", "mask_value")
