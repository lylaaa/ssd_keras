"""
An encoder that converts ground truth annotations to SSD-compatible training targets.

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

from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi


class SSDInputEncoder:
    """
    Transforms ground truth labels for object detection in images (2D bounding box coordinates and class labels) to
    the format required for training an SSD model.

    In the process of encoding the ground truth labels, a template of anchor boxes is being built, which are
    subsequently matched to the ground truth boxes via an intersection-over-union threshold criterion.
    """

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=(0.5, 1.0, 2.0),
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 offsets=None,
                 clip_boxes=False,
                 variances=(0.1, 0.1, 0.2, 0.2),
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 background_id=0):
        """
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            predictor_sizes (list): A list of 2-int tuples of the format `(height, width)` containing the output heights
                and widths of the convolutional predictor layers.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images.
                Note that you should set the scaling factors such that the resulting anchor box sizes correspond to
                 the sizes of the objects you are trying to detect. Must be >0.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the largest
                will be linearly interpolated.
                Note that the second to last of the linearly interpolated scaling factors will actually be the scaling
                 factor for the last predictor layer, while the last scaling factor is used for the second box for
                 aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
                Note that you should set the scaling factors such that the resulting anchor box sizes correspond to the
                 sizes of the objects you are trying to detect. Must be greater than or equal to `min_scale`.
            scales (list/tuple, optional): A list of floats >0 containing scaling factors per convolutional predictor
                layer. This list must be one element longer than the number of predictor layers.
                The first `k` elements are the scaling factors for the `k` predictor layers, while the last element is
                used for the second box for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
                Note this additional last scaling factor must be passed either way, even if it is not being used.
                If a list is passed, this argument overrides `min_scale` and `max_scale`. All scaling factors must be
                greater than zero.
                Note that you should set the scaling factors such that the resulting anchor box sizes correspond to
                 the sizes of the objects you are trying to detect.
            aspect_ratios_global (list/tuple, optional): The list/tuple of aspect ratios for which anchor boxes are to
                be generated. This list is valid for all prediction layers.
                Note that you should set the aspect ratios such that the resulting anchor box shapes roughly correspond
                 to the shapes of the objects you are trying to detect.
            aspect_ratios_per_layer (list, optional): A nested list containing one aspect ratio list for each prediction
                layer. If a list is passed, it overrides `aspect_ratios_global`.
                Note that you should set the aspect ratios such that the resulting anchor box shapes very roughly
                 correspond to the shapes of the objects you are trying to detect.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored
                otherwise. If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            steps (list, optional): `None` or a list with as many elements as there are predictor layers.
                The elements can be either ints/floats or tuples of two ints/floats. These numbers represent for each
                predictor layer how many pixels apart the anchor box center points should be vertically and horizontally
                along the spatial grid over the image. If the list contains ints/floats,
                then that value will be used for both spatial dimensions.
                If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
                If no steps are provided, then they will be computed such that the anchor box center points will form an
                equidistant grid within the image dimensions.
            offsets (list, optional): `None` or a list with as many elements as there are predictor layers.
                The elements can be either floats or tuples of two floats. These numbers represent for each predictor
                layer how many pixels from the top and left borders of the image the top-most and left-most anchor box
                center points should be as a fraction of `steps`.
                The last bit is important: The offsets are not absolute pixel values, but fractions of the step size
                specified in the `steps` argument. If the list contains floats, then that value will be used for both
                spatial dimensions. If the list contains tuples of two floats, then they represent
                `(vertical_offset, horizontal_offset)`.
                If no offsets are provided, then they will default to 0.5 of the step size.
            clip_boxes (bool, optional): If `True`, limits the anchor box coordinates to stay within image boundaries.
            # UNCLEAR: 作用是什么, 控制 loss 吗?
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided
                by its respective variance value.
            matching_type (str, optional): Can be either 'multi' or 'bipartite'.
                In 'bipartite' mode, each ground truth box will be matched only to the one anchor box with the highest
                IoU overlap.
                In 'multi' mode, in addition to the aforementioned(上述提及的) bipartite matching, all anchor boxes with
                an IoU overlap greater than or equal to the `pos_iou_threshold` will be matched to a given ground truth
                box.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be met in
                order to match a given ground truth box to a given anchor box.
            neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity of an anchor box
                with any ground truth box to be labeled a negative (i.e. background) box. If an anchor box is neither a
                positive, nor a negative box, it will be ignored during training.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes. Can be 'include',
                'exclude', or 'half'.
                If 'include', the border pixels belong to the boxes.
                If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong to the boxes, but not the
                other.
            coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the
                input format of the ground truth labels).
                Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(x_min, x_max, y_min, y_max)`,
                or 'corners' for the format `(x_min, y_min, x_max, y_max)`.
            normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute target coordinates, the encoder will scale all coordinates to be
                within [0,1].
                This way learning becomes independent of the input image size.
            background_id (int, optional): Determines which class ID is for the background class.
        """

        ##################################################################################
        # check and set parameters' value
        ##################################################################################
        if not (isinstance(img_height, int) and isinstance(img_width, int)):
            raise ValueError('`img_height` and `img_width` must be float')
        elif not (img_height > 0 and img_width > 0):
            raise ValueError('`img_height` and `img_width` must be greater than 0')
        else:
            self.img_height = img_height
            self.img_width = img_width

        if not (isinstance(n_classes, int) and n_classes > 0):
            raise ValueError('`n_classes` must be a positive int')
        else:
            # +1 for background class
            self.n_classes = n_classes + 1

        if not (isinstance(predictor_sizes, list) and predictor_sizes):
            raise ValueError("`predictor_sizes must be a list and not empty")
        else:
            for predictor_size in predictor_sizes:
                if not (isinstance(predictor_size, tuple) and len(predictor_size) == 2):
                    raise ValueError("Element of `predictor_sizes` must be a 2-int tuple")
            else:
                predictor_sizes = np.array(predictor_sizes)
                # predictor_sizes[:, 0] 表示的是 height, predictor_sizes[:, 1] 表示的是 width.
                self.predictor_sizes = predictor_sizes

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
        elif scales:
            if not isinstance(scales, (list, tuple)):
                raise ValueError("It must be either `scales` is None, a list or a tuple")
            elif len(scales) != predictor_sizes.shape[0] + 1:
                raise ValueError(
                    "If `scales' is a list/tuple, it must meet len(scales) == len(predictor_sizes) + 1, "
                    "but len(scales) == {} and len(predictor_sizes) + 1 == {}".format(
                        len(scales), len(predictor_sizes) + 1))
            else:
                scales = np.array(scales)
                if np.any(scales <= 0):
                    raise ValueError(
                        "All values in `scales` must be greater than 0, but the passed list of scales is {}".format(
                            scales))
                else:
                    self.scales = scales
                    self.min_scale = min_scale
                    self.max_scale = max_scale
        else:
            # If no explicit list of scaling factors was passed, we need to
            # 1. make sure that `min_scale` and `max_scale` are valid values.
            # 2. compute the list of scaling factors from `min_scale` and `max_scale`
            # 如果 min_scale==max_scale, np.linspace(min_scale,max_scale,n) 返回 n 个数都为 min_scale/max_scale
            if not 0 < min_scale <= max_scale:
                raise ValueError(
                    "It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(
                        min_scale, max_scale))
            else:
                scales = np.linspace(min_scale, max_scale, len(predictor_sizes) + 1)
                self.scales = scales
                self.min_scale = min_scale
                self.max_scale = max_scale

        # aspect_ratios
        if aspect_ratios_per_layer is not None:
            if not isinstance(aspect_ratios_per_layer, (list, tuple)):
                raise ValueError("It must be either `aspect_ratios_per_layer` is None, a list or a tuple")
            elif len(aspect_ratios_per_layer) != predictor_sizes.shape[0]:
                raise ValueError(
                    "If `aspect_ratios_per_layer` is a list/tuple, it must meet "
                    "len(aspect_ratios_per_layer) == len(predictor_sizes), "
                    "but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(
                        len(aspect_ratios_per_layer), len(predictor_sizes)))
            else:
                for aspect_ratios in aspect_ratios_per_layer:
                    if not (isinstance(aspect_ratios, (list, tuple)) and aspect_ratios):
                        raise ValueError("All aspect ratios must be a list or tuple and not empty")
                    # NOTE 当 aspect_ratios 为 () 或 [], np.any(np.array(aspect_ratios)) <=0 为 False, 所以必须有上面的判断
                    elif np.any(np.array(aspect_ratios) <= 0):
                        raise ValueError("All aspect ratios must be greater than zero.")
                else:
                    # If aspect ratios are given per layer, we'll use those.
                    self.aspect_ratios = aspect_ratios_per_layer
                    # Compute the number of boxes per spatial location for each predictor layer.
                    # For example, if a predictor layer has three different aspect ratios, [1.0, 0.5, 2.0], and is
                    # supposed to predict two boxes of slightly different size for aspect ratio 1.0, then that predictor
                    # layer predicts a total of four boxes at every spatial location across the feature map.
                    self.n_boxes = []
                    for aspect_ratios in aspect_ratios_per_layer:
                        # Adam
                        if (1 in aspect_ratios) and two_boxes_for_ar1:
                            self.n_boxes.append(len(aspect_ratios) + 1)
                        else:
                            self.n_boxes.append(len(aspect_ratios))
        else:
            if aspect_ratios_global is None:
                raise ValueError(
                    "At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
            elif not (isinstance(aspect_ratios_global, (list, tuple)) and aspect_ratios_global):
                raise ValueError(
                    "`aspect_ratios_global` must be a list/tuple and not empty when `aspect_ratios_per_layer` is None")
            # NOTE 当 aspect_ratios_global 为 () 或 [], np.any(np.array(aspect_ratios)) <=0 为 False, 所以必须有上面的判断
            elif np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")
            else:
                # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
                # `aspect_ratios_global` for all predictor layers.
                self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
                if (1 in aspect_ratios_global) and two_boxes_for_ar1:
                    self.n_boxes = len(aspect_ratios_global) + 1
                else:
                    self.n_boxes = len(aspect_ratios_global)
                self.n_boxes = [self.n_boxes] * predictor_sizes.shape[0]

        # two_boxes_for_ar1
        if not (isinstance(two_boxes_for_ar1, bool)):
            raise ValueError('`two_boxes_for_ar1` must be bool')
        else:
            self.two_boxes_for_ar1 = two_boxes_for_ar1

        if steps is not None:
            if not (isinstance(steps, (list, tuple)) and (len(steps) == predictor_sizes.shape[0])):
                raise ValueError("You must provide one step value per predictor layer.")
            else:
                self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]

        if offsets is not None:
            if not (isinstance(offsets, (list, tuple)) and (len(offsets) == predictor_sizes.shape[0])):
                raise ValueError("You must provide one offset value per predictor layer.")
            else:
                self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]

        if not (isinstance(clip_boxes, bool)):
            raise ValueError('`clip_boxes` must be bool')
        else:
            self.clip_boxes = clip_boxes

        if not (isinstance(variances, (list, tuple)) and len(variances) == 4):
            # We need one variance value for each of the four box coordinates
            raise ValueError("4 variance values must be passed, but {} values were received.".format(len(variances)))
        else:
            variances = np.array(variances)
            if np.any(variances <= 0):
                raise ValueError("All variances must be >0, but the variances given are {}".format(variances))
            else:
                self.variances = variances

        if matching_type not in ('bipartite', 'multi'):
            raise ValueError("Unexpected value for `matching_type`. Supported values are 'bipartite', 'multi'.")
        else:
            self.matching_type = matching_type

        if not isinstance(pos_iou_threshold, float):
            raise ValueError('`pos_iou_threshold` must be float')
        else:
            self.pos_iou_threshold = pos_iou_threshold

        if not isinstance(neg_iou_limit, float):
            raise ValueError('`neg_iou_limit` must be float')
        else:
            self.neg_iou_limit = neg_iou_limit

        if border_pixels not in ('half', 'include', 'exclude'):
            raise ValueError(
                "Unexpected value for `border_pixels`. Supported values are 'half', 'include' and 'exclude'.")
        else:
            self.border_pixels = border_pixels

        if coords not in ('minmax', 'centroids', 'corners'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")
        else:
            self.coords = coords

        if not (isinstance(normalize_coords, bool)):
            raise ValueError('`clip_boxes` must be bool')
        else:
            self.normalize_coords = normalize_coords

        if not (isinstance(background_id, int) and background_id >= 0):
            raise ValueError('`background_id` must be >= 0')
        else:
            self.background_id = background_id

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################

        # We only have to do this once since the anchor boxes depend only on the model configuration, not on the input
        # data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.

        # This will store the anchor boxes for each predictor layer.
        self.boxes_list = []

        # The following lists just store diagnostic information. Sometimes it's handy to have the anchor boxes'
        # center points, heights, widths, etc. in a list.

        # Anchor box center points as `(cy, cx)` for each predictor layer
        self.centers_diag = []
        # Anchor box widths and heights for each predictor layer
        self.wh_list_diag = []
        # Horizontal and vertical distances between any two boxes for each predictor layer
        self.steps_diag = []
        # Offsets for each predictor layer
        self.offsets_diag = []

        # Iterate over all predictor layers and compute the anchor boxes for each one.
        for i in range(len(self.predictor_sizes)):
            # boxes 为 np.array, shape 为 (predictor_sizes[i][0], predictor_sizes[i][1], self.n_boxes[i], 4)
            # center 为 tuple, 有两个 np.array 类型的元素, 每个元素的 shape 为 (predictor_sizes[i][0], predictor_sizes[i][1])
            # wh 为 np.array, shape 为 (self.n_boxes, 2), 第一个元素表示 anchor 的 width, 第二个元素表示 anchor 的 height
            # step 为 tuple, 有两个 int/float 类型的元素, 第一个元素表示是竖直方向上两个 anchor 的中心点的距离, 第二个
            #  元素表示水平方向上两个 anchor 中心点的距离
            # offset 为 tuple, 有两个 float 类型的元素, 第一个元素表示最左上方的 anchor 的中心点 y 坐标(以 step[0] 的 fraction 表示),
            # 第二个元素表示 anchor 中心点 x 坐标(以 step[1] 的 fraction 表示)
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(
                feature_map_size=self.predictor_sizes[i],
                aspect_ratios=self.aspect_ratios[i],
                this_scale=self.scales[i],
                next_scale=self.scales[i + 1],
                this_steps=self.steps[i],
                this_offsets=self.offsets[i],
                n_boxes=self.n_boxes[i],
                diagnostics=True)
            self.boxes_list.append(boxes)
            self.centers_diag.append(center)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)

    def __call__(self, ground_truth_labels,
                 labels_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 diagnostics=False):
        """
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): (batch_size, num_gt_boxes, 5)
                A python list of length `batch_size` that contains one 2D Numpy array for each batch image.
                Each such array has `k` rows for the `k` ground truth bounding boxes belonging to the respective image,
                and the data for each ground truth bounding box has the format `(class_id, x_min, y_min, x_max, y_max)`
                (i.e. the 'corners' coordinate format), and `class_id` must be an integer greater than 0 for all boxes
                as class ID 0 is reserved for the background class.
            labels_format (list or tuple, optional): A list or tuple that defines what in the last axis of the labels
                of an image. The list or tuple contains at least the keywords 'x_min', 'y_min', 'x_max', and 'y_max'.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vectors in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        """

        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = labels_format.index('class_id')
        xmin = labels_format.index('xmin')
        ymin = labels_format.index('ymin')
        xmax = labels_format.index('xmax')
        ymax = labels_format.index('ymax')
        # Note 这里的 `ground_truth_labels` 是一个 list
        # 每一个元素是一个 np.array, 表示一个 batch_item 的 gt_boxes
        batch_size = len(ground_truth_labels)

        ##################################################################################
        # Generate the template for y_encoded.
        ##################################################################################
        # shape 为 (batch_size, total_num_boxes, num_classes + 12), 元素都为 0
        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)

        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################
        # Every anchor box that does not have a ground truth match and
        # for which the maximal IoU overlap with any ground truth box is less than or
        # equal to `neg_iou_limit` will be a negative (background) box.

        # All boxes are background boxes by default.
        y_encoded[:, :, self.background_id] = 1
        # An identity matrix that we'll use as one-hot class vectors
        class_vectors = np.eye(self.n_classes)

        # For each batch item...
        for i in range(batch_size):
            # If there is no ground truth for this batch item, there is nothing to match.
            # 这种情况应该只发生在 generator.keep_images_without_gt == True
            if ground_truth_labels[i].size == 0:
                continue
            # The labels for this batch item
            # (num_gt_boxes, 5)
            labels = ground_truth_labels[i].astype(np.float)

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:, [xmax]] - labels[:, [xmin]] <= 0) or np.any(labels[:, [ymax]] - labels[:, [ymin]] <= 0):
                # 这种情况应该只发生在 generator.degenerate_box_handling == 'warn'
                raise DegenerateBoxError(
                    "SSDInputEncoder detected degenerate ground truth bounding boxes "
                    "for batch item {} with bounding boxes {}, ".format(i, labels) +
                    "i.e. bounding boxes where x_max <= x_min and/or y_max <= y_min. "
                    "Degenerate ground truth bounding boxes will lead to NaN errors during the training.")

            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                # Normalize y_min and y_max relative to the image height
                labels[:, [ymin, ymax]] /= self.img_height
                # Normalize x_min and x_max relative to the image width
                labels[:, [xmin, xmax]] /= self.img_width

            # Maybe convert the box coordinate format.
            if self.coords == 'centroids':
                labels = convert_coordinates(labels,
                                             start_index=xmin,
                                             conversion='corners2centroids',
                                             border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels,
                                             start_index=xmin,
                                             conversion='corners2minmax')

            # The one-hot class IDs for the ground truth boxes of this batch item
            # class_vectors 的 shape 为 (num_classes, num_classes)
            # labels[:, class_id] 的 shape 为 (num_gt_boxes, )
            # classes_one_hot 的 shape 为 (num_gt_boxes, num_classes)
            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
            # The one-hot version of the labels for this batch item
            # shape 为 (num_gt_boxes, num_classes + 4)
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)

            ##################################################################################
            # Match anchors and gt_boxes
            ##################################################################################
            # 1. Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # labels[: [xmin, ymin, xmax, ymax]] 的 shape 为 (num_ground_truth_boxes, 4)
            # y_encoded[i, :, -12:-8] 的 shape 为 (num_anchor_boxes, 4)
            # similarities 的 shape 为 (num_ground_truth_boxes, num_anchor_boxes)
            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12:-8],
                               coords=self.coords,
                               mode='outer_product',
                               border_pixels=self.border_pixels)

            # 2: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            #   This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            # shape 为 (num_gt_boxes,), 每个元素表示与该 gt_box 有最大 iou 的 anchor_box 的 id
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            # 在每个对应的 anchor_box 上设置 label 值
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # 3: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #   ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #   such ground truth box.

            if self.matching_type == 'multi':
                # Get all matches that satisfy the IoU threshold.
                # matches[0] 表示所有满足条件的 anchor_box 对应的 gt_box 的 id
                # matches[1] 表示所有满足条件的 anchor_box 的 id
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # 4: Now after the matching is done, all negative (background) anchor boxes that have
            #   an IoU of `neg_iou_limit` or more with any ground truth box will be set to neutral,
            #   i.e. they will no longer be background boxes. These anchors are "too close" to a
            #   ground truth box to be valid background boxes.

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes_indices = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            # 那么 neutral_boxes 的 class_one_hot 全为 0, 不属于任何 class
            # 这样设置的话, 如果某个 anchor_box 和所有 gt_boxes 的最大 overlap 小于 threshold, 且这个值是该 gt_box 和所有
            # anchor_boxes 的最大 overlap, 那么该 anchor_box 仍然被认为是 positive, 但是我认为不算合理, 因为这是 anchor 没取好
            y_encoded[i, neutral_boxes_indices, self.background_id] = 0

        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################

        if self.coords == 'centroids':
            # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [-12, -11]] -= y_encoded[:, :, [-8, -7]]
            # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:, :, [-12, -11]] /= y_encoded[:, :, [-6, -5]] * y_encoded[:, :, [-4, -3]]
            # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= y_encoded[:, :, [-6, -5]]
            # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
            y_encoded[:, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encoded[:, :, [-2, -1]]
        elif self.coords == 'corners':
            # (gt - anchor) for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -10]] /= np.expand_dims(y_encoded[:, :, -6] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-11, -9]] /= np.expand_dims(y_encoded[:, :, -5] - y_encoded[:, :, -7], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all four coordinates,
            # where 'size' refers to w and h respectively
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]
        elif self.coords == 'minmax':
            # (gt - anchor) for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encoded[:, :, -8:-4]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(y_encoded[:, :, -7] - y_encoded[:, :, -8], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(y_encoded[:, :, -5] - y_encoded[:, :, -6], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all four coordinates,
            # where 'size' refers to w and h respectively
            y_encoded[:, :, -12:-8] /= y_encoded[:, :, -4:]

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box,
            # but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            # Keeping the anchor box coordinates means setting the offsets to zero.
            # 因为 y_encoded[:, :, -12:-8] 现在表示的都是 delta 值
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        n_boxes,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        """
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer of size
        `feature_map_size == [feature_map_height, feature_map_width]`.

        # 先算出 self.image_width, self.image_height 的较小值
        # 然后根据 this_scale 和 next_scale 算出各种 ap 的 anchor_boxes 的 width, height
        # 根据 this_steps 和 this_offsets 算出最左上方的 anchor_box 的中心点的 x, y 坐标
        # 推算出 (feature_map_size[0], feature_map_size[1]) 个 anchor_boxes 的中心点的坐标
        Arguments:
            feature_map_size (list/tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            n_boxes (int): An int, number of anchor boxes of one pixel in feature map
            this_steps (int or 2-int tuple): anchor 中心点右移一个位置和下移一个位置的距离
            this_offsets (float or 2-float tuple): 最左上方的 anchor 的中心点的相对于 feature_map 左上方点的偏移量
            diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.
        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension is determined by self.coords.
        """
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for aspect_ratio in aspect_ratios:
            if aspect_ratio == 1:
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                # aspect_ratio = box_width / box_height
                box_width = this_scale * size * np.sqrt(aspect_ratio)
                box_height = this_scale * size / np.sqrt(aspect_ratio)
                wh_list.append((box_width, box_height))
        wh_array = np.array(wh_list)
        assert len(wh_array) == n_boxes, \
            'incorrect number of anchor boxes, len(wh_array)={} and n_boxes={}'.format(wh_array, n_boxes)

        ##################################################################################
        # Compute the grid of box center points. They are identical for all aspect ratios.
        ##################################################################################
        # 1. Compute the step sizes
        # i.e. how far apart the anchor box center points will be vertically and horizontally.
        if this_steps is None:
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
            else:
                raise ValueError('`this_steps` must be one of 2-int list, 2-int tuple and int')

        # 2. Compute the offsets, i.e.
        # at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if this_offsets is None:
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, float):
                offset_height = this_offsets
                offset_width = this_offsets
            else:
                raise ValueError('`this_offsets` must be one of 2-float list, 2-float tuple and float')

        # 3. Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        # 第一个 anchor 的中心点往下移动 (feature_map_size[0] - 1) 个位置, 这样一共就有 feature_map_size[0] 个 位置
        # (feature_map_size[0], )
        cy = np.linspace(offset_height * step_height,
                         offset_height * step_height + (feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        # 第一个 anchor 的中心点往右移动 (feature_map_size[1] - 1) 个位置, 这样一共就有 feature_map_size[1] 个 位置
        # (feature_map_size[1], )
        cx = np.linspace(offset_width * step_width,
                         offset_width * step_width + (feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])
        # shape 为 (feature_map_size[1], feature_map_size[0])
        # meshgrid 的操作, cx_grid 相当于把 cx 作为一行复制 len(cy) 次, cy_grid 相当于把 cy 作为一列复制 len(cx) 次
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want further down
        # shape 为 (feature_map_size[1], feature_map_size[0], 1)
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))
        # Set cx
        # np.tile() 之后 shape 为 (feature_map_size[1], feature_map_size[0], n_boxes)
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))
        # Set cy
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))
        # Set w
        boxes_tensor[:, :, :, 2] = wh_array[:, 0]
        # Set h
        boxes_tensor[:, :, :, 3] = wh_array[:, 1]

        # Convert `(cx, cy, w, h)` to `(x_min, y_min, x_max, y_max)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back
        #  and forth.
        if self.coords == 'centroids':
            # Convert `(x_min, y_min, x_max, y_max)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor,
                                               start_index=0,
                                               conversion='corners2centroids',
                                               border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(x_min, y_min, x_max, y_max)` to `(x_min, x_max, y_min, y_max).
            boxes_tensor = convert_coordinates(boxes_tensor,
                                               start_index=0,
                                               conversion='corners2minmax',
                                               border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_array, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        """
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
         and the sub-functions it calls are identical to those performed inside the SSD model.
        This, of course, must be the case in order to preserve the spatial meaning of each box prediction, but
        it's useful to make yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that
        `y_encoded` has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        """
        # Tile the anchor boxes for each predictor layer across all batch items.
        batch_boxes = []
        # boxes_list 是一个 list
        # 每一个元素是一个 np.array, shape 为 (feature_map_height, feature_map_width, n_boxes, 4)
        # 表示一个 feature_map 上的所有的 anchor_box 的坐标(具体的值取决于 self.coords)
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting order of the tensor content
            # will be identical to the order obtained from the reshaping operation in our Keras model
            # (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            batch_boxes.append(boxes)

        # Concatenate the anchor tensors from the individual prediction layers to one.
        # shape 为 (batch_size, total_num_boxes, 4)
        boxes_tensor = np.concatenate(batch_boxes, axis=1)

        # Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        # It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        # contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        # Long live broadcasting
        variances_tensor += self.variances

        # Concatenate the classes, boxes and variances tensors to get our final template for y_encoded.
        # We also need another tensor of the shape of `boxes_tensor` as a space filler
        # so that `y_encoding_template` has the same shape as the SSD model output tensor.
        # The content of this tensor is irrelevant, we'll just use `boxes_tensor` a second time.
        # shape 为 (batch_size, total_num_boxes, num_classes + 4 + 4 + 4)
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template


class DegenerateBoxError(Exception):
    """
    An exception class to be raised if degenerate boxes are being detected.
    """
    pass
