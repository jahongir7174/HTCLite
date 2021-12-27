import math
import random
from copy import deepcopy
from os.path import basename

import cv2
import mmcv
import numpy
from PIL import Image, ImageOps, ImageEnhance
from mmdet.datasets.pipelines.transforms import PIPELINES


def resize(image, image_size):
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)
    if ratio != 1:
        shape = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    return image, image.shape[:2]


def xy2wh(x):
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * x[:, 0] + pad_w  # top left x
    y[:, 1] = h * x[:, 1] + pad_h  # top left y
    return y


def whn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def mask2box(mask, w, h):
    x, y = mask.T
    inside = (x >= 0) & (y >= 0) & (x <= w) & (y <= h)
    x, y, = x[inside], y[inside]
    if any(x):
        return numpy.array([x.min(), y.min(), x.max(), y.max()]), x, y
    else:
        return numpy.zeros((1, 4)), x, y


def box_ioa(box1, box2, eps=1E-7):
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    area1 = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0)
    area2 = (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0)
    inter_area = area1 * area2

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def masks2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xy2wh(numpy.array(boxes))


def resample_masks(masks, n=1000):
    for i, s in enumerate(masks):
        x = numpy.linspace(0, len(s) - 1, n)
        xp = numpy.arange(len(s))
        mask = [numpy.interp(x, xp, s[:, i]) for i in range(2)]
        masks[i] = numpy.concatenate(mask).reshape(2, -1).T
    return masks


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = numpy.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def copy_paste(image, boxes, masks, p=0.):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    n = len(masks)
    if p and n:
        h, w, c = image.shape
        img = numpy.zeros(image.shape, numpy.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = boxes[j], masks[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = box_ioa(box, boxes[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                boxes = numpy.concatenate((boxes, [[l[0], *box]]), 0)
                masks.append(numpy.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(img, [masks[j].astype(numpy.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=image, src2=img)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        image[i] = result[i]

    return image, boxes, masks


def random_perspective(image, boxes=(), masks=(),
                       degrees=0, translate=.1, scale=.5,
                       shear=0, perspective=0., border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    c_gain = numpy.eye(3)
    c_gain[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    c_gain[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    p_gain = numpy.eye(3)
    p_gain[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    p_gain[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    r_gain = numpy.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    r_gain[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    s_gain = numpy.eye(3)
    s_gain[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    s_gain[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    t_gain = numpy.eye(3)
    t_gain[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w  # x translation (pixels)
    t_gain[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h  # y translation (pixels)

    # Combined rotation matrix
    matrix = t_gain @ s_gain @ r_gain @ p_gain @ c_gain  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        if perspective:
            image = cv2.warpPerspective(image, matrix, dsize=(w, h), borderValue=(0, 0, 0))
        else:  # affine
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    n = len(boxes)
    if n:
        new_masks = []
        new_boxes = numpy.zeros((n, 4))
        for i, mask in enumerate(resample_masks(masks)):
            xy = numpy.ones((len(mask), 3))
            xy[:, :2] = mask
            xy = xy @ matrix.T
            xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

            # clip
            new_boxes[i], x, y = mask2box(xy, w, h)
            new_masks.append([x, y])

        # filter candidates
        candidates = box_candidates(boxes[:, 1:5].T * s, new_boxes.T, area_thr=0.01)
        boxes = boxes[candidates]
        boxes[:, 1:5] = new_boxes[candidates]
        masks = []
        for candidate, new_mask in zip(candidates, new_masks):
            if candidate:
                masks.append(new_mask)
    return image, boxes, masks


def mosaic(self, index):
    boxes4, masks4 = [], []
    size = numpy.random.choice(self.image_sizes)
    border = [-size // 2, -size // 2]
    indexes4 = [index] + random.choices(range(self.num_samples), k=3)
    yc, xc = [int(random.uniform(-x, 2 * size + x)) for x in border]
    numpy.random.shuffle(indexes4)
    results4 = [deepcopy(self.dataset[index]) for index in indexes4]
    filename = results4[0]['filename']
    shapes = [x['img_shape'][:2] for x in results4]
    image4 = numpy.full((2 * size, 2 * size, 3), 0, numpy.uint8)

    for i, (results, shape) in enumerate(zip(results4, shapes)):
        image, (h, w) = resize(results['img'], size)

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        masks = []
        label = numpy.array(results['ann_info']['labels'])
        for mask in results['ann_info']['masks']:
            mask = [j for i in mask for j in i]
            mask = numpy.array(mask).reshape(-1, 2)
            masks.append(mask / numpy.array([shape[1], shape[0]]))
        masks = [x for x in masks]
        try:
            boxes = (label.reshape(-1, 1), masks2boxes(masks))
            boxes = numpy.concatenate(boxes, axis=1)
        except IndexError:
            return None
        if boxes.size:
            boxes[:, 1:] = whn2xy(boxes[:, 1:], w, h, pad_w, pad_h)
            masks = [xyn2xy(x, w, h, pad_w, pad_h) for x in masks]
        boxes4.append(boxes)
        masks4.extend(masks)
    # concatenate & clip
    boxes4 = numpy.concatenate(boxes4, 0)
    for i, box4 in enumerate(boxes4[:, 1:]):
        if i % 2 == 0:
            numpy.clip(box4, 0, 2 * size, out=box4)
        else:
            numpy.clip(box4, 0, 2 * size, out=box4)
    for mask4 in masks4:
        numpy.clip(mask4[:, 0:1], 0, 2 * size, out=mask4[:, 0:1])
        numpy.clip(mask4[:, 1:2], 0, 2 * size, out=mask4[:, 1:2])
    image4, boxes4, masks4 = copy_paste(image4, boxes4, masks4, p=0.0)
    image4, boxes4, masks4 = random_perspective(image4, boxes4, masks4, border=border)

    label = []
    boxes = []
    masks = []
    for box4, mask4 in zip(boxes4, masks4):
        mask = []
        for x, y in zip(mask4[0], mask4[1]):
            mask.append(x)
            mask.append(y)
        masks.append([mask])
        label.append(box4[0])
        boxes.append(box4[1:5])
    if len(boxes) and len(label) and len(masks):
        label = numpy.array(label, dtype=numpy.int64)
        boxes = numpy.array(boxes, dtype=numpy.float32)
        return dict(filename=filename, image=image4, label=label, boxes=boxes, masks=masks)
    else:
        return None


def mix_up(self, index1, index2):
    r = numpy.random.beta(32.0, 32.0)
    data1 = mosaic(self, index1)
    data2 = mosaic(self, index2)
    if data1 is not None and data2 is not None:
        image1 = data1['image']
        label1 = data1['label']
        boxes1 = data1['boxes']
        masks1 = data1['masks']

        image2 = data2['image']
        label2 = data2['label']
        boxes2 = data2['boxes']
        masks2 = data2['masks']
        image = (image1 * r + image2 * (1 - r)).astype(numpy.uint8)
        boxes = numpy.concatenate((boxes1, boxes2), 0)
        label = numpy.concatenate((label1, label2), 0)
        masks1.extend(masks2)
        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes, masks=masks1)
    if data1 is None and data2 is not None:
        image = data2['image']
        label = data2['label']
        boxes = data2['boxes']
        masks = data2['masks']
        return dict(filename=data2['filename'], image=image, label=label, boxes=boxes, masks=masks)
    if data1 is not None and data2 is None:
        image = data1['image']
        label = data1['label']
        boxes = data1['boxes']
        masks = data1['masks']
        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes, masks=masks)
    return None


def process(self, data):
    image = data['image']
    label = data['label']
    boxes = data['boxes']
    masks = data['masks']

    results = dict()
    results['filename'] = data['filename']

    results['ann_info'] = {'labels': label, 'bboxes': boxes, 'masks': masks}
    results['img_info'] = {'height': image.shape[0], 'width': image.shape[1]}
    results['img_fields'] = ['img']
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['ori_filename'] = basename(data['filename'])
    results['img'] = image
    results['img_shape'] = image.shape
    results['ori_shape'] = image.shape
    return self.pipeline(results)


def box2field():
    bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
    bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}
    return bbox2label, bbox2mask


def invert(image, _):
    return ImageOps.invert(image)


def equalize(image, _):
    return ImageOps.equalize(image)


def solar1(image, magnitude):
    return ImageOps.solarize(image, int((magnitude / 10.) * 256))


def solar2(image, magnitude):
    return ImageOps.solarize(image, 256 - int((magnitude / 10.) * 256))


def solar3(image, magnitude):
    lut = []
    for i in range(256):
        if i < 128:
            lut.append(min(255, i + int((magnitude / 10.) * 110)))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        return image


def poster1(image, magnitude):
    magnitude = int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster2(image, magnitude):
    magnitude = 4 - int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster3(image, magnitude):
    magnitude = int((magnitude / 10.) * 4) + 4
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def contrast1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    if random.random() > 0.5:
        magnitude *= -1
    return ImageEnhance.Contrast(image).enhance(magnitude)


def contrast2(image, magnitude):
    return ImageEnhance.Contrast(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def contrast3(image, _):
    return ImageOps.autocontrast(image)


def color1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    if random.random() > 0.5:
        magnitude *= -1
    return ImageEnhance.Color(image).enhance(magnitude)


def color2(image, magnitude):
    return ImageEnhance.Color(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def brightness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    if random.random() > 0.5:
        magnitude *= -1
    return ImageEnhance.Brightness(image).enhance(magnitude)


def brightness2(image, magnitude):
    return ImageEnhance.Brightness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def sharpness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    if random.random() > 0.5:
        magnitude *= -1
    return ImageEnhance.Sharpness(image).enhance(magnitude)


def sharpness2(image, magnitude):
    return ImageEnhance.Sharpness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


class Shear:
    def __init__(self, min_val=0.0003, max_val=0.3):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _shear_image(results, magnitude, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(img,
                                       magnitude,
                                       direction)
            results[key] = img_sheared.astype(img.dtype)

    @staticmethod
    def _shear_boxes(results, magnitude, direction):
        h, w, c = results['img_shape']
        if direction == 'horizontal':
            shear_matrix = numpy.stack([[1, magnitude], [0, 1]]).astype(numpy.float32)  # [2, 2]
        else:
            shear_matrix = numpy.stack([[1, 0], [magnitude, 1]]).astype(numpy.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose((2, 1, 0)).astype(numpy.float32)  # [nb_box, 2, 4]
            new_coords = numpy.matmul(shear_matrix[None, :, :], coordinates)  # [nb_box, 2, 4]
            min_x = numpy.min(new_coords[:, 0, :], axis=-1)
            min_y = numpy.min(new_coords[:, 1, :], axis=-1)
            max_x = numpy.max(new_coords[:, 0, :], axis=-1)
            max_y = numpy.max(new_coords[:, 1, :], axis=-1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _shear_masks(results, magnitude, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        magnitude = numpy.random.uniform(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            magnitude *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._shear_image(results, magnitude, direction)
        self._shear_boxes(results, magnitude, direction)
        self._shear_masks(results, magnitude, direction)
        self._filter_invalid(results)
        return results


class Rotate:
    def __init__(self, min_val=1, max_val=90):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _rotate_image(results, angle, center, scale):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(img, angle, center, scale)
            results[key] = img_rotated.astype(img.dtype)

    @staticmethod
    def _rotate_boxes(results, rotate_matrix):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])
            coordinates = numpy.concatenate((coordinates,
                                             numpy.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                                            axis=1)
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = numpy.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x = numpy.min(rotated_coords[:, :, 0], axis=1)
            min_y = numpy.min(rotated_coords[:, :, 1], axis=1)
            max_x = numpy.max(rotated_coords[:, :, 0], axis=1)
            max_y = numpy.max(rotated_coords[:, :, 1], axis=1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _rotate_masks(results, angle, center, scale):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, 0)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        h, w = results['img'].shape[:2]
        angle = numpy.random.randint(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            angle *= -1
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        self._rotate_image(results, angle, center, 1)
        self._rotate_boxes(results, matrix)
        self._rotate_masks(results, angle, center, 1)
        self._filter_invalid(results)
        return results


class Translate:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    @staticmethod
    def _translate_image(results, offset, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(img, offset, direction).astype(img.dtype)

    @staticmethod
    def _translate_boxes(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            if direction == 'horizontal':
                min_x = numpy.maximum(0, min_x + offset)
                max_x = numpy.minimum(w, max_x + offset)
            elif direction == 'vertical':
                min_y = numpy.maximum(0, min_y + offset)
                max_y = numpy.minimum(h, max_y + offset)

            results[key] = numpy.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    @staticmethod
    def _translate_masks(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, 0)

    @staticmethod
    def _filter_invalid(results):
        bbox2label, bbox2mask = box2field()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > 0) & (bbox_h > 0)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]
        return results

    def __call__(self, results):
        h, w, c = results['img_shape']
        offset = numpy.random.randint(1, int(self.ratio * min(h, w)))
        if numpy.random.rand() > 0.5:
            offset *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._translate_image(results, offset, direction)
        self._translate_boxes(results, offset, direction)
        self._translate_masks(results, offset, direction)
        self._filter_invalid(results)
        return results


@PIPELINES.register_module()
class RandomHSV:
    def __init__(self, h=5, s=30, v=30):
        self.h = h
        self.s = s
        self.v = v

    def __call__(self, results):
        img = results['img']
        hsv_gains = numpy.random.uniform(-1, 1, 3) * [self.h, self.s, self.v]
        # random selection of h, s, v
        hsv_gains *= numpy.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(numpy.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(numpy.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = numpy.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = numpy.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        results['img'] = img
        return results


@PIPELINES.register_module()
class RandomAugment:
    def __init__(self):
        self.transform1 = RandomHSV()
        self.transform2 = [color1, color2, solar1, solar2, solar3, invert,
                           poster1, poster2, poster3, equalize, contrast1,
                           contrast2, contrast3, sharpness1, sharpness2,
                           brightness1, brightness2]
        self.transform3 = [Shear(), Rotate(), Translate()]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            results = self.transform1(results)
        else:
            image = results['img']
            image = Image.fromarray(image[:, :, ::-1])
            for transform in numpy.random.choice(self.transform2, 2):
                magnitude = min(10., max(0., random.gauss(9, 0.5)))
                image = transform(image, magnitude)
            results['img'] = numpy.array(image)[:, :, ::-1]
        for transform in numpy.random.choice(self.transform3, 1):
            results = transform(results)
        return results


@PIPELINES.register_module()
class GridDropout:
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.5):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def __call__(self, results):
        img = results['img']
        if numpy.random.rand() > self.prob:
            return results
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = numpy.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = numpy.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = numpy.ones((hh, ww), numpy.float32)
        st_h = numpy.random.randint(d)
        st_w = numpy.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = numpy.random.randint(self.rotate)
        mask = Image.fromarray(numpy.uint8(mask))
        mask = mask.rotate(r)
        mask = numpy.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(numpy.float32)
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(img)
        mask = numpy.expand_dims(mask, 2).repeat(3, axis=2)
        if self.offset:
            offset = 2 * (numpy.random.rand(h, w) - 0.5)
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        results['img'] = img
        return results


@PIPELINES.register_module()
class TSCopyPaste:
    def __init__(self, data_dir='../Dataset/Ins2021'):
        self.data_dir = data_dir

        self.crop_image_dir = 'c_images'
        self.crop_label_dir = 'c_labels'

        import glob
        self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')
        self.src_names_1 = glob.glob(f'{data_dir}/{self.crop_image_dir}/1/*.png')

        with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
            self.labels = {}
            for line in f.readlines():
                line = line.rstrip().split(' ')
                self.labels[line[0]] = line[1:]
        with open(f'{data_dir}/{self.crop_label_dir}/1.txt') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                self.labels[line[0]] = line[1:]

    def paste(self, img):
        gt_label = []
        gt_masks = []
        gt_boxes = []

        dst_h, dst_w = img.shape[:2]
        num_0 = numpy.random.randint(5, 15)
        num_1 = numpy.random.randint(5, 15)
        y_c_list = numpy.random.randint(dst_h // 2 - 256, dst_h // 2 + 256, num_0 + num_1)
        x_c_list = numpy.random.randint(256, dst_w - 256, num_0 + num_1)
        src_names = numpy.random.choice(self.src_names_0, num_0).tolist()
        src_names.extend(numpy.random.choice(self.src_names_1, num_1).tolist())

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)
        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            dst_poly = []
            for p in poly_list[i]:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(img.shape, img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
            gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            mask = mask_list[i].copy()
            img[dst_mask > 0] = 0
            img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
            if 'human' in basename(src_name_list[i]):
                gt_label.append(0)
            else:
                gt_label.append(1)
            dst_point = []
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        img = results['img']
        img, label, boxes, masks = self.paste(img)

        masks.extend(results['ann_info']['masks'])
        label.extend(results['ann_info']['labels'].tolist())
        boxes.extend(results['ann_info']['bboxes'].tolist())

        results['img'] = img
        results['ann_info']['labels'] = numpy.array(label, numpy.int64)
        results['ann_info']['bboxes'] = numpy.array(boxes, numpy.float32)
        results['ann_info']['masks'] = masks
        return results
