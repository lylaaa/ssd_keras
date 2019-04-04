"""
Various geometric image transformations for 2D object detection, both deterministic
and probabilistic.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import cv2
import random

from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator


class Resize:
    """
    Resize images to a specified height and width in pixels.
    """

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid OpenCV interpolation mode.
                For example, integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria after the
                transformation. Refer to the `BoxFilter` documentation for details. If `None`, the validity of the
                bounding boxes is not checked.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        img_height, img_width = image.shape[:2]
        xmin = self.labels_format.index('xmin')
        ymin = self.labels_format.index('ymin')
        xmax = self.labels_format.index('xmax')
        ymax = self.labels_format.index('ymax')
        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)
        if return_inverter:
            # Adam
            def inverter(new_labels):
                old_labels = np.copy(new_labels)
                old_labels[:, [ymin, ymax]] = np.round(
                    old_labels[:, [ymin, ymax]] * (img_height / self.out_height), decimals=0)
                old_labels[:, [xmin, xmax]] = np.round(
                    old_labels[:, [xmin, xmax]] * (img_width / self.out_width), decimals=0)
                return old_labels
        else:
            inverter = None
        if labels is None:
            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if self.box_filter is not None:
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)
            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels


class ResizeRandomInterp:
    """
    Resize images to a specified height and width in pixels using a randomly selected interpolation mode.
    """

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=(cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4),
                 box_filter=None,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers that represent valid OpenCV
                interpolation modes. For example, integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)


class Flip:
    """
    Flips images horizontally or vertically.
    """

    def __init__(self,
                 dim='horizontal',
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along the vertical axis.
                If 'vertically', images will be flipped vertically, i.e. along the horizontal axis.
            labels_format (list or tuple, optional): A list or tuple that defines which index in the last axis of the
                labels of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        """
        if dim not in {'horizontal', 'vertical'}:
            raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):
        img_height, img_width = image.shape[:2]
        xmin = self.labels_format.index('xmin')
        ymin = self.labels_format.index('ymin')
        xmax = self.labels_format.index('xmax')
        ymax = self.labels_format.index('ymax')
        if self.dim == 'horizontal':
            # 左右翻转
            image = image[:, ::-1]
            if labels is None:
                return image, labels
            else:
                labels = np.copy(labels)
                # Adam
                labels[:, [xmin, xmax]] = img_width - 1 - labels[:, [xmax, xmin]]
                return image, labels
        else:
            # 上下翻转
            image = image[::-1]
            if labels is None:
                return image, labels
            else:
                labels = np.copy(labels)
                # Adam
                labels[:, [ymin, ymax]] = img_height - 1 - labels[:, [ymax, ymin]]
                return image, labels


class RandomFlip:
    """
    Randomly flips images horizontally or vertically. The randomness only refers
    to whether or not the image will be flipped.
    """

    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along the vertical axis.
                If 'horizontal', images will be flipped vertically, i.e. along the horizontal axis.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the
                labels of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p < self.prob:
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        else:
            return image, labels


class Translate:
    """
    Translates images horizontally and/or vertically.
    """

    def __init__(self,
                 dy,
                 dx,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            dy (float): The fraction of the image height by which to translate images along the vertical axis.
                Positive values translate images downwards, negative values translate images upwards.
            dx (float): The fraction of the image width by which to translate images along the horizontal axis.
                Positive values translate images to the right, negative values translate images to the left.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the image after the
                translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):
        img_height, img_width = image.shape[:2]

        # Compute the translation matrix.
        dy_abs = int(round(img_height * self.dy_rel))
        dx_abs = int(round(img_width * self.dx_rel))
        matrix = np.float32([[1, 0, dx_abs],
                             [0, 1, dy_abs]])

        # Translate the image.
        image = cv2.warpAffine(image,
                               M=matrix,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image, labels
        else:
            xmin = self.labels_format.index('xmin')
            ymin = self.labels_format.index('ymin')
            xmax = self.labels_format.index('xmax')
            ymax = self.labels_format.index('ymax')
            labels = np.copy(labels)

            # Translate the box coordinates to the translated image's coordinate system.
            labels[:, [xmin, xmax]] += dx_abs
            labels[:, [ymin, ymax]] += dy_abs

            # Compute all valid boxes for this patch.
            if self.box_filter is not None:
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=img_height - 1)
                labels[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=img_width - 1)

            return image, labels


class RandomTranslate:
    """
    Randomly translates images horizontally and/or vertically.
    """

    def __init__(self,
                 dy_minmax=(0.03, 0.3),
                 dx_minmax=(0.03, 0.3),
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0, 0, 0),
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            dy_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that determines the minimum
                and maximum relative translation of images along the vertical axis both upward and downward. That is,
                images will be randomly translated by at least `min` and at most `max` either upward or downward.
                For example, if `dy_minmax == (0.05,0.3)`, an image of size `(100,100)` will be translated by at least 5
                and at most 30 pixels either upward or downward. The translation direction is chosen randomly.
            dx_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that determines the minimum
                and maximum relative translation of images along the horizontal axis both to the left and right.
                That is, images will be randomly translated by at least `min` and at most `max` either left or right.
                For example, if `dx_minmax == (0.05,0.3)`, an image of size `(100,100)` will be translated by at least 5
                and at most 30 pixels either left or right. The translation direction is chosen randomly.
            prob (float, optional): `(1 - prob)` determines the probability with which the original, unaltered image is
                returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the image after the
                translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a translated image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maximum number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-int list/tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """

        if dy_minmax[0] > dy_minmax[1]:
            raise ValueError("It must be `dy_minmax[0] <= dy_minmax[1]`.")
        if dx_minmax[0] > dx_minmax[1]:
            raise ValueError("It must be `dx_minmax[0] <= dx_minmax[1]`.")
        if dy_minmax[0] < 0 or dx_minmax[0] < 0:
            raise ValueError("It must be `dy_minmax[0] >= 0` and `dx_minmax[0] >= 0`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.dy_minmax = dy_minmax
        self.dx_minmax = dx_minmax
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.translate = Translate(dy=0,
                                   dx=0,
                                   clip_boxes=self.clip_boxes,
                                   box_filter=self.box_filter,
                                   background=self.background,
                                   labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p < self.prob:
            img_height, img_width = image.shape[:2]
            xmin = self.labels_format.index('xmin')
            ymin = self.labels_format.index('ymin')
            xmax = self.labels_format.index('xmax')
            ymax = self.labels_format.index('ymax')

            # Override the preset labels format.
            if self.image_validator is not None:
                self.image_validator.labels_format = self.labels_format
            self.translate.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):
                # Pick the relative amount by which to translate.
                dy_rel = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
                dx_rel = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
                # Pick the direction in which to translate.
                dy_rel = np.random.choice([-dy_rel, dy_rel])
                dx_rel = np.random.choice([-dx_rel, dx_rel])
                self.translate.dy_rel = dy_rel
                self.translate.dx_rel = dx_rel

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or we do but we have no image_validator,
                    # we will accept any outcome as valid.
                    return self.translate(image, labels)
                else:
                    # Translate the box coordinates to the translated image's coordinate system.
                    new_labels = labels.copy()
                    new_labels[:, [ymin, ymax]] += int(round(img_height * dy_rel))
                    new_labels[:, [xmin, xmax]] += int(round(img_width * dx_rel))

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.translate(image, labels)
            # If all attempts failed, return the unaltered input image.
            return image, labels
        return image, labels


class Scale:
    """
    Scales images, i.e. zooms in or out.
    """

    def __init__(self,
                 factor,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0, 0, 0),
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            factor (float): The fraction of the image size by which to scale images. Must be positive.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details.
                If `None`, the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """

        if factor <= 0:
            raise ValueError("It must be `factor > 0`.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.factor = factor
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):
        img_height, img_width = image.shape[:2]
        # Compute the rotation matrix.
        rotation_matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                  angle=0,
                                                  scale=self.factor)
        # Scale the image.
        image = cv2.warpAffine(image,
                               M=rotation_matrix,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)
        if labels is None:
            return image, labels
        else:
            xmin = self.labels_format.index('xmin')
            ymin = self.labels_format.index('ymin')
            xmax = self.labels_format.index('xmax')
            ymax = self.labels_format.index('ymax')

            labels = np.copy(labels)
            # Scale the bounding boxes accordingly.
            # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
            # 这波操作要比 mask_rcnn 中对原图进行 resize 后的操作要更方便一点
            # shape 为 (3, num_gt_boxes)
            top_lefts = np.array([labels[:, xmin], labels[:, ymin], np.ones(labels.shape[0])])
            # shape 为 (3, num_gt_boxes)
            bottom_rights = np.array([labels[:, xmax], labels[:, ymax], np.ones(labels.shape[0])])
            # dot 为 (2,3).(3,num_gt_boxes) = (2,num_gt_boxes) 再转置成 (num_gt_boxes, 2)
            new_top_lefts = (np.dot(rotation_matrix, top_lefts)).T
            new_bottom_rights = (np.dot(rotation_matrix, bottom_rights)).T
            labels[:, [xmin, ymin]] = np.round(new_top_lefts, decimals=0).astype(np.int)
            labels[:, [xmax, ymax]] = np.round(new_bottom_rights, decimals=0).astype(np.int)

            # Compute all valid boxes for this patch.
            if self.box_filter is not None:
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)
            if self.clip_boxes:
                labels[:, [ymin, ymax]] = np.clip(labels[:, [ymin, ymax]], a_min=0, a_max=img_height - 1)
                labels[:, [xmin, xmax]] = np.clip(labels[:, [xmin, xmax]], a_min=0, a_max=img_width - 1)
            return image, labels


class RandomScale:
    """
    Randomly scales images.
    """

    def __init__(self,
                 min_factor=0.5,
                 max_factor=1.5,
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0, 0, 0),
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            min_factor (float, optional): The minimum fraction of the image size by which to scale images.
                Must be positive.
            max_factor (float, optional): The maximum fraction of the image size by which to scale images.
                Must be positive.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a scaled image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maximal number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple at least contains the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """

        if not (0 < min_factor <= max_factor):
            raise ValueError("It must be `0 < min_factor <= max_factor`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.scale = Scale(factor=1.0,
                           clip_boxes=self.clip_boxes,
                           box_filter=self.box_filter,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        """

        Arguments:
            image:
            labels: (np.array, optional)
        Returns:
        """
        p = np.random.uniform(0, 1)
        if p < self.prob:
            img_height, img_width = image.shape[:2]
            xmin = self.labels_format.index('xmin')
            ymin = self.labels_format.index('ymin')
            xmax = self.labels_format.index('xmax')
            ymax = self.labels_format.index('ymax')

            # Override the preset labels format.
            if self.image_validator is not None:
                self.image_validator.labels_format = self.labels_format
            self.scale.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick a scaling factor.
                factor = np.random.uniform(self.min_factor, self.max_factor)
                self.scale.factor = factor

                # Adam
                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or image_validator, we will accept any outcome as valid.
                    return self.scale(image, labels)
                else:
                    # Scale the bounding boxes accordingly.
                    # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
                    top_lefts = np.array([labels[:, xmin], labels[:, ymin], np.ones(labels.shape[0])])
                    bottom_rights = np.array([labels[:, xmax], labels[:, ymax], np.ones(labels.shape[0])])
                    # Compute the rotation matrix.
                    rotation_matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                              angle=0,
                                                              scale=factor)
                    new_top_lefts = (np.dot(rotation_matrix, top_lefts)).T
                    new_bottom_rights = (np.dot(rotation_matrix, bottom_rights)).T
                    new_labels = labels.copy()
                    new_labels[:, [xmin, ymin]] = np.around(new_top_lefts, decimals=0).astype(np.int)
                    new_labels[:, [xmax, ymax]] = np.around(new_bottom_rights, decimals=0).astype(np.int)

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.scale(image, labels)
            # If all attempts failed, return the unaltered input image.
            return image, labels
        return image, labels


class Rotate:
    """
    Rotates images counter-clockwise by 90, 180, or 270 degrees.
    """

    def __init__(self,
                 angle,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            angle (int): The angle in degrees by which to rotate the images counter-clockwise.
                Only 90, 180, and 270 are valid values.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """

        if angle not in {90, 180, 270}:
            raise ValueError("`angle` must be in the set {90, 180, 270}.")
        self.angle = angle
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Compute the rotation matrix.
        rotation_matrix = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                  angle=self.angle,
                                                  scale=1)

        # Get the sine and cosine from the rotation matrix.
        # M 为  [[cos_theta, sin_theta, (1 - cos_theta)*center_x - sin_theta*center_y],
        #       [-sin_theta, cos_theta, sin_theta*center_x + (1 - cos_theta)*center_y]]
        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])
        # Compute the new bounding dimensions of the image.
        img_width_new = int(img_height * sin_angle + img_width * cos_angle)
        img_height_new = int(img_height * cos_angle + img_width * sin_angle)
        # Adjust the rotation matrix to take into account the translation.
        # UNCLEAR: It works but why?
        rotation_matrix[1, 2] += (img_height_new - img_height) / 2
        rotation_matrix[0, 2] += (img_width_new - img_width) / 2
        # Rotate the image.
        image = cv2.warpAffine(image,
                               M=rotation_matrix,
                               dsize=(img_width_new, img_height_new))
        if labels is None:
            return image, labels
        else:
            xmin = self.labels_format.index('xmin')
            ymin = self.labels_format.index('ymin')
            xmax = self.labels_format.index('xmax')
            ymax = self.labels_format.index('ymax')

            labels = np.copy(labels)
            # Rotate the bounding boxes accordingly.
            # Transform two opposite corner points of the rectangular boxes using the rotation matrix `rotation_matrix`.
            top_lefts = np.array([labels[:, xmin], labels[:, ymin], np.ones(labels.shape[0])])
            bottom_rights = np.array([labels[:, xmax], labels[:, ymax], np.ones(labels.shape[0])])
            new_top_lefts = (np.dot(rotation_matrix, top_lefts)).T
            new_bottom_rights = (np.dot(rotation_matrix, bottom_rights)).T
            labels[:, [xmin, ymin]] = np.round(new_top_lefts, decimals=0).astype(np.int)
            labels[:, [xmax, ymax]] = np.round(new_bottom_rights, decimals=0).astype(np.int)

            if self.angle == 90:
                # ymin and ymax were switched by the rotation.
                # 左上角 (xmin, ymin) 变成了左下角 (xmin_new, ymax_new)
                # 右下角 (xmax, ymax) 变成了右上角 (xmax_new, ymin_new)
                # 所以 xmin = xmin_new, ymin_new = y_max, xmax=xmax_new, ymax_new = y_min
                labels[:, [ymax, ymin]] = labels[:, [ymin, ymax]]
            elif self.angle == 180:
                # ymin and ymax were switched by the rotation, and also xmin and xmax were switched.
                # 左上角 (xmin, ymin) 变成了右下角 (xmax_new, ymax_new)
                # 右下角 (xmax, ymax) 变成了左上角 (xmin_new, ymin_new)
                # 所以 xmin = xmax_new, ymin = ymax_new, xmax = xmin_new, ymax = ymin_new
                labels[:, [ymax, ymin]] = labels[:, [ymin, ymax]]
                labels[:, [xmax, xmin]] = labels[:, [xmin, xmax]]
            elif self.angle == 270:
                # xmin and xmax were switched by the rotation.
                # 左上角 (xmin, ymin) 变成了右上角 (xmax_new, ymin_new)
                # 右下角 (xmax, ymax) 变成了左下角 (xmin_new, ymax_new)
                # 素有 xmin = xmax_new, ymin = ymin_new, xmax = xmin_new, ymax = ymax_new
                labels[:, [xmax, xmin]] = labels[:, [xmin, xmax]]
            return image, labels


class RandomRotate:
    """
    Randomly rotates images counter-clockwise.
    """

    def __init__(self,
                 angles=(90, 180, 270),
                 prob=0.5,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):
        """
        Arguments:
            angles (list): The list of angles in degrees from which one is randomly selected to rotate
                the images counter-clockwise. Only 90, 180, and 270 are valid values.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'xmin', 'ymin', 'xmax', and 'ymax'.
        """
        for angle in angles:
            if angle not in {90, 180, 270}:
                raise ValueError("`angles` can only contain the values 90, 180, and 270.")
        self.angles = angles
        self.prob = prob
        self.labels_format = labels_format
        self.rotate = Rotate(angle=90, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0, 1)
        if p < self.prob:
            # Pick a rotation angle.
            self.rotate.angle = random.choice(self.angles)
            self.rotate.labels_format = self.labels_format
            return self.rotate(image, labels)
        return image, labels
