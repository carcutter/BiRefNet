import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image


def path_to_image(path, size=(1024, 1024), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if image is None:
        # cv2.imread returns None for a missing/corrupt/zero-byte file. Fail with the offending
        # path instead of a cryptic crash inside cv2.resize/cvtColor further down. The dataset
        # filters missing pairs at construction, so reaching here means a present-but-unreadable
        # file (e.g. a truncated download) — surface it loudly rather than train on garbage.
        raise FileNotFoundError('cv2 could not read image (missing or corrupt): {}'.format(path))
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image


def path_to_binary_mask_from_colors(path, size=None, fg_colors=((255, 0, 0), (0, 255, 0))):
    """Load an RGB color-coded segmentation mask and binarize: pixels matching any
    color in `fg_colors` become foreground (255), all others become background (0).

    Thresholding happens at native resolution before resize, so edge antialiasing in
    a linear resize won't blur class colors across the threshold.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fg = np.zeros(rgb.shape[:2], dtype=bool)
    for color in fg_colors:
        fg |= np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
    bin_mask = (fg.astype(np.uint8) * 255)
    if size:
        bin_mask = cv2.resize(bin_mask, size, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(bin_mask).convert('L')


def path_to_binary_mask_auto(path, size=None, fg_colors=((255, 0, 0), (0, 255, 0)),
                             chroma_tol=20, chroma_frac=0.005):
    """Adaptive per-mask binarization for datasets that mix two mask conventions:
      - **Colour-coded mask** (contains coloured, i.e. non-grey, pixels): apply the colour rule —
        pixels matching any colour in `fg_colors` (red/green) are foreground, everything else bg.
      - **Black-and-white mask** (only greyscale, no colour): apply luminance — white/bright pixels
        are foreground.

    A pixel is 'chromatic' when its channel spread max(R,G,B)-min(R,G,B) > `chroma_tol`; the mask is
    treated as colour-coded when chromatic pixels exceed `chroma_frac` of the image (robust to a few
    stray/compression pixels). Thresholding is at native resolution before resize.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    spread = rgb.max(axis=-1).astype(np.int16) - rgb.min(axis=-1).astype(np.int16)
    is_coloured = float((spread > chroma_tol).mean()) > chroma_frac
    if is_coloured:
        fg = np.zeros(rgb.shape[:2], dtype=bool)
        for color in fg_colors:
            fg |= np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
    else:
        # Pure black/white mask: white (bright) is foreground.
        fg = rgb.max(axis=-1) > 127
    bin_mask = (fg.astype(np.uint8) * 255)
    if size:
        bin_mask = cv2.resize(bin_mask, size, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(bin_mask).convert('L')


def path_to_binary_mask_nonbg(path, size=None, bg_colors=((0, 0, 0),)):
    """Load an RGB color-coded segmentation mask and binarize by background subtraction:
    any pixel NOT in `bg_colors` becomes foreground (255). Robust to adding new class
    colors — the unified foreground is the union of every class.

    Thresholding happens at native resolution before resize so the linear resize cannot
    blur class colors across the threshold.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bg = np.zeros(rgb.shape[:2], dtype=bool)
    for color in bg_colors:
        bg |= np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
    bin_mask = ((~bg).astype(np.uint8) * 255)
    if size:
        bin_mask = cv2.resize(bin_mask, size, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(bin_mask).convert('L')



def path_to_window_blob_map(path, size=None, fg_colors=((255, 0, 0), (0, 255, 0)),
                            window_colors=((0, 0, 255),), ring_radius=12,
                            window_frac=0.5, max_area_frac=0.2, min_area=64):
    """Load an RGB color-coded mask and return an 'L' map (0/255) flagging foreground blobs
    that are ISOLATED INSIDE A WINDOW region.

    A pixel is 255 iff it belongs to a foreground (any `fg_colors`) connected component whose
    surrounding `ring_radius`-px ring is at least `window_frac` window (any `window_colors`).
    Components smaller than `min_area` px (noise) or larger than `max_area_frac` of the image
    (the main interior body — not an isolated blob) are skipped.

    Detection runs at native resolution (so blob topology and ring fractions are measured on the
    true mask); the result is resized to `size` (W, H) with NEAREST so the 0/255 region stays crisp.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    fg = np.zeros((H, W), dtype=bool)
    for c in fg_colors:
        fg |= np.all(rgb == np.array(c, dtype=np.uint8), axis=-1)
    win = np.zeros((H, W), dtype=bool)
    for c in window_colors:
        win |= np.all(rgb == np.array(c, dtype=np.uint8), axis=-1)

    out = np.zeros((H, W), dtype=np.uint8)
    if fg.any() and win.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        k = 2 * int(ring_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        area_cap = max_area_frac * H * W
        for lab in range(1, n):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < min_area or area > area_cap:
                continue
            comp = (labels == lab).astype(np.uint8)
            ring = (cv2.dilate(comp, kernel) > 0) & (comp == 0)
            rs = int(ring.sum())
            if rs == 0:
                continue
            if (ring & win).sum() / rs >= window_frac:
                out[labels == lab] = 255

    if size:
        out = cv2.resize(out, size, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(out).convert('L')


def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BiRefNet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
