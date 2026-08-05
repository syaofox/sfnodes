import copy

import numpy as np
from PIL import Image, ImageChops

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "add",
    "subtract",
    "difference",
    "darker",
    "lighter",
    "color_burn",
    "color_dodge",
    "linear_burn",
    "linear_dodge",
    "overlay",
    "soft_light",
    "hard_light",
    "vivid_light",
    "pin_light",
    "linear_light",
    "hard_mix",
]


def _to_float_array(image):
    return np.asarray(image).astype(np.float64) / 255.0


def _to_image(array):
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))


def _blend_color_burn(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    img = 1 - (1 - layer) / (bg + 0.001)
    return _to_image(np.clip(img, 0.0, 1.0))


def _blend_color_dodge(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    img = layer / (1.0 - bg + 0.001)
    return _to_image(np.clip(img, 0.0, 1.0))


def _blend_linear_burn(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    return _to_image(np.clip(bg + layer - 1, 0.0, None))


def _blend_linear_dodge(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    return _to_image(np.clip(bg + layer, None, 1.0))


def _blend_overlay(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask = layer < 0.5
    img = 2 * bg * layer * mask + (1 - mask) * (1 - 2 * (1 - bg) * (1 - layer))
    return _to_image(img)


def _blend_soft_light(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask = bg < 0.5
    t1 = (2 * bg - 1) * (layer - layer * layer) + layer
    t2 = (2 * bg - 1) * (np.sqrt(layer) - layer) + layer
    return _to_image(t1 * mask + t2 * (1 - mask))


def _blend_hard_light(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask = bg < 0.5
    t1 = 2 * bg * layer
    t2 = 1 - 2 * (1 - bg) * (1 - layer)
    return _to_image(t1 * mask + t2 * (1 - mask))


def _blend_vivid_light(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask = bg < 0.5
    t1 = np.clip(1 - (1 - layer) / (2 * bg + 0.001), 0.0, None)
    t2 = np.clip(layer / (2 * (1 - bg) + 0.001), None, 1.0)
    return _to_image(t1 * mask + t2 * (1 - mask))


def _blend_pin_light(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask_1 = layer < (bg * 2 - 1)
    mask_2 = layer > 2 * bg
    t1 = 2 * bg - 1
    t2 = layer
    t3 = 2 * bg
    return _to_image(t1 * mask_1 + t2 * (1 - mask_1) * (1 - mask_2) + t3 * mask_2)


def _blend_linear_light(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    return _to_image(np.clip(layer + bg * 2 - 1, 0.0, 1.0))


def _blend_hard_mix(background_image, layer_image):
    bg = _to_float_array(background_image)
    layer = _to_float_array(layer_image)
    mask = bg + layer > 1
    return _to_image(mask.astype(np.float64))


def chop_image(background_image, layer_image, blend_mode, opacity):
    ret_image = background_image
    if blend_mode == "normal":
        ret_image = copy.deepcopy(layer_image)
    elif blend_mode == "multiply":
        ret_image = ImageChops.multiply(background_image, layer_image)
    elif blend_mode == "screen":
        ret_image = ImageChops.screen(background_image, layer_image)
    elif blend_mode == "add":
        ret_image = ImageChops.add(background_image, layer_image, 1, 0)
    elif blend_mode == "subtract":
        ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
    elif blend_mode == "difference":
        ret_image = ImageChops.difference(background_image, layer_image)
    elif blend_mode == "darker":
        ret_image = ImageChops.darker(background_image, layer_image)
    elif blend_mode == "lighter":
        ret_image = ImageChops.lighter(background_image, layer_image)
    elif blend_mode == "color_burn":
        ret_image = _blend_color_burn(background_image, layer_image)
    elif blend_mode == "color_dodge":
        ret_image = _blend_color_dodge(background_image, layer_image)
    elif blend_mode == "linear_burn":
        ret_image = _blend_linear_burn(background_image, layer_image)
    elif blend_mode == "linear_dodge":
        ret_image = _blend_linear_dodge(background_image, layer_image)
    elif blend_mode == "overlay":
        ret_image = _blend_overlay(background_image, layer_image)
    elif blend_mode == "soft_light":
        ret_image = _blend_soft_light(background_image, layer_image)
    elif blend_mode == "hard_light":
        ret_image = _blend_hard_light(background_image, layer_image)
    elif blend_mode == "vivid_light":
        ret_image = _blend_vivid_light(background_image, layer_image)
    elif blend_mode == "pin_light":
        ret_image = _blend_pin_light(background_image, layer_image)
    elif blend_mode == "linear_light":
        ret_image = _blend_linear_light(background_image, layer_image)
    elif blend_mode == "hard_mix":
        ret_image = _blend_hard_mix(background_image, layer_image)
    if opacity == 0:
        ret_image = background_image
    elif opacity < 100:
        alpha = 1.0 - float(opacity) / 100
        ret_image = Image.blend(ret_image, background_image, alpha)
    return ret_image
