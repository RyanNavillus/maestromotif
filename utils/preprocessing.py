# Code for preprocessing
import cv2
import os

import numpy as np
import torch

from PIL import Image, ImageFont, ImageDraw
from numba import njit

from rl_baseline.obs_wrappers import COLORS, BlstatsWrapper, initialize_char_array, tile_characters_to_image
from utils.forked_pdb import ForkedPdb

class GPT5BaselineTransform:
    def __init__(self, font_size=9, crop_size=12, rescale_font_size=(6, 6), data_keys={}):
        self.char_array = initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size
        self.data_keys = data_keys

        # Render only crop region
        self.half_crop_size = crop_size // 2
        self.output_height_chars = crop_size
        self.output_width_chars = crop_size

        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width
        )

    def __call__(self, minibatch):
        
        # Dataset would look like this. BS is batch size, L is the length of a sequence

        batch = {}
        if 'tty' in self.data_keys:
            chars = minibatch['tty_chars'] # Size (2, L, 24, 80)
            colors = minibatch['tty_colors'] # Size (2, L, 24, 80)
            tty_shape = chars.shape[-2:]
            if self.crop_size:
                # Center around player
                center_xs = bl_stats[:, 0].astype(int)
                center_ys = bl_stats[:, 1].astype(int)
                offset_hs = center_ys - self.half_crop_size
                offset_ws = center_xs - self.half_crop_size

            cropped_views = np.zeros((np.prod(seq_shape), *self.chw_image_shape), dtype=np.uint8)
            for i, (char, color, offset_h, offset_w) in enumerate(zip(chars.reshape(-1, *tty_shape), colors.reshape(-1, *tty_shape), offset_hs, offset_ws)):
                tile_characters_to_image(
                    out_image=cropped_views[i],
                    chars=char,
                    colors=color,
                    output_height_chars=self.output_height_chars,
                    output_width_chars=self.output_width_chars,
                    char_array=self.char_array,
                    offset_h=offset_h,
                    offset_w=offset_w
                )
            cropped_views = cropped_views.reshape(seq_shape + cropped_views.shape[1:])
            batch["obs"] = cropped_views

        if 'blstats' in self.data_keys:
            bl_stats = minibatch['blstats'].astype(np.int32) # Size (2, L, 27)
            seq_shape = bl_stats.shape[:-1]

            bl_stats = bl_stats.reshape((-1, bl_stats.shape[-1]))
            bl_shape = min(bl_stats.shape[-1], BlstatsWrapper.BLSTAT_NORMALIZATION_STATS.shape[-1])

            norm_bl_stats = (bl_stats * BlstatsWrapper.BLSTAT_NORMALIZATION_STATS[:bl_shape]).astype(np.float32)
            np.clip(
                norm_bl_stats,
                BlstatsWrapper.BLSTAT_CLIP_RANGE[0],
                BlstatsWrapper.BLSTAT_CLIP_RANGE[1],
                out=norm_bl_stats
            ) 
            norm_bl_stats = norm_bl_stats.reshape(seq_shape + norm_bl_stats.shape[1:])
            batch["norm_blstats"] = norm_bl_stats
        
        message_strs = []
        for msg in minibatch['message']:
            msg = msg[0]
            message_str = bytes(msg[msg != 0]).decode("utf-8")
            message_strs.append(message_str)

        batch["message_strs"] = np.expand_dims(np.array(message_strs), axis=1)

        batch["message"] = minibatch['message'] # Size (2, L, 256)

        # Pass through all the keys that are not used in the transform
        for key, value in minibatch.items():
            if key not in ['blstats', 'message']:
                batch[key] = value
        return batch


class ToTensorDict(object):
    """Convert a dict of ndarrays in a dict of Tensors or Numpy arrays."""
    def __call__(self, sample):
        tensor_dict = {}
        for key, value in sample.items():
            if key =='idx':
                tensor_dict[key] = value
            elif np.issubdtype(value.dtype, np.number) or value.dtype == bool:
                tensor_dict[key] = torch.from_numpy(np.array(value))
            else:
                tensor_dict[key] = np.array(value)
        return tensor_dict


class DictAsAttributes:
    def __init__(self, data_dict):
        self.__dict__['_data_dict'] = data_dict

    def __getattr__(self, key):
        if key in self._data_dict:
            return self._data_dict[key]
        else:
            raise AttributeError(f"'DictAsAttributes' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data_dict[key] = value

    def __delattr__(self, key):
        del self._data_dict[key]