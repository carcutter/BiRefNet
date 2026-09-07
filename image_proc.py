import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import torch
from torchvision import transforms


## CPU version refinement
def FB_blur_fusion_foreground_estimator_cpu(image, FG, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FGA = cv2.blur(FG * alpha, (r, r))
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    FG = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG = np.clip(FG, 0, 1)
    return FG, blurred_B


def FB_blur_fusion_foreground_estimator_cpu_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    FG, blur_B = FB_blur_fusion_foreground_estimator_cpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_cpu(image, FG, blur_B, alpha, r=6)[0]


## GPU version refinement
def mean_blur(x, kernel_size):
    """
    equivalent to cv.blur
    x:  [B, C, H, W]
    """
    if kernel_size % 2 == 0:
        pad_l = kernel_size // 2 - 1
        pad_r = kernel_size // 2
        pad_t = kernel_size // 2 - 1
        pad_b = kernel_size // 2
    else:
        pad_l = pad_r = pad_t = pad_b = kernel_size // 2

    x_padded = torch.nn.functional.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='replicate')

    return torch.nn.functional.avg_pool2d(x_padded, kernel_size=(kernel_size, kernel_size), stride=1, count_include_pad=False)

def FB_blur_fusion_foreground_estimator_gpu(image, FG, B, alpha, r=90):
    as_dtype = lambda x, dtype: x.to(dtype) if x.dtype != dtype else x

    input_dtype = image.dtype
    # convert image to float to avoid overflow
    image = as_dtype(image, torch.float32)
    FG = as_dtype(FG, torch.float32)
    B = as_dtype(B, torch.float32)
    alpha = as_dtype(alpha, torch.float32)

    blurred_alpha = mean_blur(alpha, kernel_size=r)

    blurred_FGA = mean_blur(FG * alpha, kernel_size=r)
    blurred_FG = blurred_FGA / (blurred_alpha + 1e-5)

    blurred_B1A = mean_blur(B * (1 - alpha), kernel_size=r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    FG_output = blurred_FG + alpha * (image - alpha * blurred_FG - (1 - alpha) * blurred_B)
    FG_output = torch.clamp(FG_output, 0, 1)

    return as_dtype(FG_output, input_dtype), as_dtype(blurred_B, input_dtype)


def FB_blur_fusion_foreground_estimator_gpu_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/ZhengPeng7/BiRefNet/issues/226#issuecomment-3016433728
    FG, blur_B = FB_blur_fusion_foreground_estimator_gpu(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator_gpu(image, FG, blur_B, alpha, r=6)[0]


def refine_foreground(image, mask, r=90, device='cuda'):
    """both image and mask are in range of [0, 1]"""
    if mask.size != image.size:
        mask = mask.resize(image.size)

    if device == 'cuda':
        image = transforms.functional.to_tensor(image).float().cuda()
        mask = transforms.functional.to_tensor(mask).float().cuda()
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        estimated_foreground = FB_blur_fusion_foreground_estimator_gpu_2(image, mask, r=r)
        
        estimated_foreground = estimated_foreground.squeeze()
        estimated_foreground = (estimated_foreground.mul(255.0)).to(torch.uint8)
        estimated_foreground = estimated_foreground.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
    else:
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        estimated_foreground = FB_blur_fusion_foreground_estimator_cpu_2(image, mask, r=r)
        estimated_foreground = (estimated_foreground * 255.0).astype(np.uint8)

    estimated_foreground = Image.fromarray(np.ascontiguousarray(estimated_foreground))

    return estimated_foreground


def preproc(image, label, preproc_methods=['flip'], weight=None):
    # `weight` is an optional extra label-aligned PIL map (e.g. the window-blob loss-weight map).
    # It rides the same GEOMETRIC ops as the label (flip/crop/rotate) so it stays pixel-aligned;
    # the image-only ops (enhance/blur) leave it untouched. Returns (image, label) when weight is
    # None, else (image, label, weight) — so existing 2-arg callers are unaffected.
    if 'flip' in preproc_methods:
        image, label, weight = cv_random_flip(image, label, weight)
    if 'crop' in preproc_methods:
        image, label, weight = random_crop(image, label, weight)
    if 'rotate' in preproc_methods:
        image, label, weight = random_rotate(image, label, weight)
    if 'letterbox' in preproc_methods:
        image, label, weight = random_letterbox(image, label, weight)
    if 'enhance' in preproc_methods:
        image = color_enhance(image)
    if 'blur' in preproc_methods:
        image = random_blur(image)
    if weight is None:
        return image, label
    return image, label, weight


def cv_random_flip(img, label, weight=None):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if weight is not None:
            weight = weight.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, weight


def random_crop(image, label, weight=None, prob=0.3, crop_frac=0.5):
    """Fixed half-size zoom crop. With probability `prob`, take a random-position window whose size
    is `crop_frac` of each side (default 0.5 ⇒ half width and half height, i.e. a quarter of the
    area / a 2× zoom), then resize it back to the original size. Resizing back is required: the
    batch is fixed-resolution (config.size), so a raw crop of a different size would break collation.

    Geometric — image, label and (optional) weight map are cropped from the same box and resized
    together. Image uses bilinear; label/weight use nearest to keep the mask crisp.
    """
    if random.random() >= prob:
        return image, label, weight
    W, H = image.size
    cw = max(1, int(round(crop_frac * W)))
    ch = max(1, int(round(crop_frac * H)))
    x0 = random.randint(0, W - cw)
    y0 = random.randint(0, H - ch)
    box = (x0, y0, x0 + cw, y0 + ch)
    image = image.crop(box).resize((W, H), Image.BILINEAR)
    label = label.crop(box).resize((W, H), Image.NEAREST)
    if weight is not None:
        weight = weight.crop(box).resize((W, H), Image.NEAREST)
    return image, label, weight


def random_letterbox(image, label, weight=None, prob=0.3, max_frac=0.3):
    """Letterbox pad (top/bottom only). With probability `prob`, shrink the content vertically and
    add horizontal bars so the frame keeps its original size — WHITE bars on the image, BLACK bars
    on the mask (and 0 on the weight map). Simulates letterboxed / matted footage where the subject
    sits in a central horizontal band. Only top/bottom borders are added; full width is preserved.

    The total border height is a random fraction (up to `max_frac`) of H, split randomly between top
    and bottom (so the band may be centred or offset). Content is resized into the remaining height,
    then pasted onto the padded canvas. Geometric: image (bilinear) + label/weight (nearest) stay
    aligned. Output size is unchanged so fixed-resolution batch collation is preserved.
    """
    if random.random() >= prob:
        return image, label, weight
    W, H = image.size
    total = min(int(round(random.uniform(0.05, max_frac) * H)), H - 1)   # top+bottom border height
    if total <= 0:
        return image, label, weight
    content_h = H - total
    top = random.randint(0, total)                                       # random split ⇒ centred or offset
    white = (255,) * len(image.getbands())

    out_img = Image.new(image.mode, (W, H), white)
    out_img.paste(image.resize((W, content_h), Image.BILINEAR), (0, top))
    out_lab = Image.new(label.mode, (W, H), 0)                           # black = background
    out_lab.paste(label.resize((W, content_h), Image.NEAREST), (0, top))
    image, label = out_img, out_lab
    if weight is not None:
        out_w = Image.new(weight.mode, (W, H), 0)                        # no upweighting in the bars
        out_w.paste(weight.resize((W, content_h), Image.NEAREST), (0, top))
        weight = out_w
    return image, label, weight


def random_rotate(image, label, weight=None, angle=15):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        if weight is not None:
            # NEAREST keeps the weight map a clean 0/255 region (no interpolated grey at edges).
            weight = weight.rotate(random_angle, Image.NEAREST)
    return image, label, weight


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def random_gaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def random_blur(image, prob=0.15, radius_range=(0.1, 0.5)):
    # Light Gaussian blur to mimic mild defocus / soft optics. Applied with `prob` probability
    # only to the image (not the label). Radius kept low-intensity so edges stay learnable.
    if random.random() < prob:
        radius = random.uniform(*radius_range)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image
