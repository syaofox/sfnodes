import hashlib

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from comfy.utils import common_upscale
from typing import Any, Callable, Dict, Iterable, List

def mask2tensor(mask):
    image = (
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        .movedim(1, -1)
        .expand(-1, -1, -1, 3)
    )
    return image


def tensor2mask(image, channel="red"):
    channels = ["red", "green", "blue", "alpha"]
    mask = image[:, :, :, channels.index(channel)]
    return mask


def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert("RGB"))


def tensor2images(tensor: torch.Tensor) -> List[Image.Image]:
    images = []
    for i in range(tensor.shape[0]):
        image = tensor[i].cpu().numpy()
        image = (image.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        images.append(Image.fromarray(image))
    return images

def images2tensor(images: Iterable[Image.Image]) -> torch.Tensor:
    tensor_list = []
    for image in images:
        image = np.array(image).astype(np.float32) / 255.0
        tensor_list.append(torch.from_numpy(image).unsqueeze(0))
    return torch.cat(tensor_list, dim=0)

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    # return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2hex(image):
    return hashlib.sha256(
        np.array(tensor2pil(image)).astype(np.uint16).tobytes()
    ).hexdigest()


def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np).unsqueeze(0)
    return mask


def mask2pil(mask):
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = (mask.cpu().numpy() * 255.0).astype("uint8")
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def pil2np(image):
    return np.array(image).astype(np.uint8)


def np2pil(image):
    return Image.fromarray(image)


def tensor2np(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert("RGB"))


def np2tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)


def np2mask(image):
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def mask2np(image):
    return np.array(image.squeeze(0).cpu())


def image_posterize(image, threshold):
    image = image.mean(dim=3, keepdim=True)
    image = (image > threshold).float()
    image = image.repeat(1, 1, 1, 3)

    return image


def rescale_image(image, width, height):
    samples = image.movedim(-1, 1)
    resized = common_upscale(samples, width, height, "lanczos", "disabled")
    return resized.movedim(1, -1)



def contrast_adaptive_sharpening(image, amount):
    img = T.functional.pad(image, (1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = torch.nan_to_num(output)
    output = output.clamp(0, 1)

    return output