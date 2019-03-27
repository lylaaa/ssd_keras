"""
A Keras port of the original Caffe SSD300 network.

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
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=([1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]),
            two_boxes_for_ar1=True,
            steps=(8, 16, 32, 64, 100, 300),
            offsets=None,
            clip_boxes=False,
            variances=(0.1, 0.1, 0.2, 0.2),
            coords='centroids',
            normalize_coords=True,
            subtract_mean=(123, 117, 104),
            divide_by_stddev=None,
            swap_channels=(2, 1, 0),
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    """
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture, as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor box layers.
    In case you're training the network, the parameters passed here must be the same as the ones used to set up
    `SSDBoxEncoder`. In case you're loading trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'.
            In 'training' mode, the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast'
            modes, the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering.
            The difference between latter two modes is that 'inference' follows the exact procedure of the original
            Caffe implementation, while 'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the largest will be
            linearly interpolated.
            Note that the second to last of the linearly interpolated scaling factors will actually be the scaling
            factor for the last predictor layer, while the last scaling factor is used for the second box for aspect
            ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
        scales (list/tuple, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list/tuple, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list/tuple, optional): A nested list/tuple containing one aspect ratio list/tuple for
            each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored
            otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1.
            The first will be generated using the scaling factor for the respective layer,
            the second one will be generated using geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers.
            The elements can be either ints/floats or tuples of two ints/floats.
            These numbers represent for each predictor layer how many pixels apart the anchor box center points should
            be vertically and horizontally along the spatial grid over the image.
            If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers.
            The elements can be either floats or tuples of two floats.
            These numbers represent for each predictor layer how many pixels from the top and left boarders of the image
            the top-most and left-most anchor box center points should be as a fraction of `steps`.
            Note the last bit is important: The offsets are not absolute pixel values, but fractions of the step size
             specified in the `steps` argument.
            If the list contains floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two floats, then they represent `(vertical_offset, horizontal_offset)`.
            If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input
            format of the ground truth labels).
            Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
            'minmax' for the format `(xmin, xmax, ymin, ymax)`,
            or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute
            coordinates, i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the
            input image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum
            suppression stage, while a larger value will result in a larger part of the selection process happening in
            the confidence thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than
            `iou_threshold` with a locally maximal box will be removed from the set of predictions for a given class,
            where 'maximal' refers to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS
            stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers.
            This isn't strictly necessary since you can always get their sizes easily via the Keras API,
            but it's convenient and less error-prone to get them this way.
            They are only relevant for training anyway (SSDBoxEncoder needs to know the spatial dimensions of the
            predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion of the output tensor shape
            for each convolutional predictor layer. During training, the generator function needs this in order to
            transform the ground truth labels into tensors of identical structure as the output tensors of the model,
            which is in turn needed for the cost function.

    References:
        https://arxiv.org/abs/1512.02325v5
    """

    # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_predictor_layers = 6
    # Make the internal name shorter.
    l2_reg = l2_regularization

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    # image_size
    if not (isinstance(image_size, (list, tuple)) and len(image_size) == 3):
        raise ValueError(
            "`image_size` must be a 3-int list/tuple"
            "that contains image_height, image_width, image_channels respectively")
    elif not (isinstance(image_size[0], int) and isinstance(image_size[1], int) and isinstance(image_size[2], int)):
        raise ValueError(
            "`image_size` must be a 3-int list/tuple"
            "that contains image_height, image_width, image_channels respectively")
    elif np.any(np.array(image_size) <= 0):
        raise ValueError("All elements of image_size must be greater than zero.")
    else:
        img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    if not (isinstance(n_classes, int) and n_classes > 0):
        raise ValueError('`n_classes` must be a positive int')
    else:
        # +1 for background class
        n_classes = n_classes + 1

    # mode
    if mode not in ('training', 'inference', 'inference_fast'):
        raise ValueError(
            "Unexpected value for `mode`. Supported values are 'training', 'inference' and 'inference_fast'.")

    # scales
    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    elif scales:
        if not isinstance(scales, (list, tuple)):
            raise ValueError("It must be either `scales` is None, a list or a tuple")
        elif len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, "
                             "but len(scales) == {}.".format(n_predictor_layers + 1, len(scales)))
        else:
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError(
                    "All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
    else:
        # If no explicit list of scaling factors was passed, we need to
        # 1. make sure that `min_scale` and `max_scale` are valid values
        # 2. compute the list of scaling factors from `min_scale` and `max_scale`
        if not (isinstance(min_scale, float) and isinstance(max_scale, float)):
            raise ValueError('`min_scale` and `max_scale` must be float')
        elif not 0 < min_scale <= max_scale:
            raise ValueError(
                "It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(
                    min_scale, max_scale))
        else:
            scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    # two_boxes_for_ar1
    if not (isinstance(two_boxes_for_ar1, bool)):
        raise ValueError('`two_boxes_for_ar1` must be bool')

    # aspect_ratio
    if aspect_ratios_per_layer is not None:
        if not isinstance(aspect_ratios_per_layer, (list, tuple)):
            raise ValueError("It must be either `aspect_ratios_per_layer` is None, a list or a tuple")
        elif len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "If `aspect_ratios_per_layer` is a list/tuple, it must meet "
                "len(aspect_ratios_per_layer) == n_predictor_layers, "
                "but len(aspect_ratios_per_layer) == {} and n_predictor_layers == {}".format(
                    len(aspect_ratios_per_layer), n_predictor_layers))
        for aspect_ratios in aspect_ratios_per_layer:
            if not (isinstance(aspect_ratios, (list, tuple)) and aspect_ratios):
                raise ValueError("All aspect ratios must be a list or tuple and not empty")
            # NOTE 当 aspect_ratios 为 () 或 [], np.any(np.array(aspect_ratios)) <=0 为 False, 所以必须有上面的判断
            elif np.any(np.array(aspect_ratios) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")
        else:
            # Compute the number of boxes to be predicted per cell for each predictor layer.
            # We need this so that we know how many channels the predictor layers need to have.
            n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) and two_boxes_for_ar1:
                    # +1 for the second box for aspect ratio 1
                    n_boxes.append(len(aspect_ratios) + 1)
                else:
                    n_boxes.append(len(aspect_ratios))
            # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
            aspect_ratios = aspect_ratios_per_layer
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
            # If aspect ratios are given per layer, we'll use those.
            aspect_ratios = [aspect_ratios_global] * n_predictor_layers
            # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor
            # layer
            if (1 in aspect_ratios_global) and two_boxes_for_ar1:
                n_boxes = len(aspect_ratios_global) + 1
            else:
                n_boxes = len(aspect_ratios_global)
            n_boxes = [n_boxes] * n_predictor_layers

    if steps is not None:
        if not (isinstance(steps, (list, tuple)) and (len(steps) == n_predictor_layers)):
            raise ValueError("You must provide at least one step value per predictor layer.")
    else:
        steps = [None] * n_predictor_layers

    if offsets is not None:
        if not (isinstance(offsets, (list, tuple)) and (len(offsets) == n_predictor_layers)):
            raise ValueError("You must provide at least one offset value per predictor layer.")
    else:
        offsets = [None] * n_predictor_layers

    if not (isinstance(clip_boxes, bool)):
        raise ValueError('`clip_boxes` must be bool')

    if not (isinstance(variances, (list, tuple)) and len(variances) == 4):
        # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be passed, but {} values were received.".format(len(variances)))
    else:
        if np.any(np.array(variances) <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if coords not in ('minmax', 'centroids', 'corners'):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    if not (isinstance(normalize_coords, bool)):
        raise ValueError('`normalize_coords` must be bool')

    if swap_channels is not None:
        if not (isinstance(swap_channels, (list, tuple)) and (len(swap_channels) in (3, 4))):
            raise ValueError("3 or 4 values must be passed if swap_channels is not None, but {} were received"
                             .format(len(swap_channels)))

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                 tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))
    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if subtract_mean is not None:
        x1 = Lambda(input_mean_normalization,
                    output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if divide_by_stddev is not None:
        x1 = Lambda(input_stddev_normalization,
                    output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap,
                    output_shape=(img_height, img_width, img_channels),
                    name='input_channel_swap')(x1)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv1_2')(conv1_1)
    # padding='same' 仍然会改变 feature map 的大小, 因为是否改变 feature_map 的大小是由 stride 决定的
    # 参考 https://www.imooc.com/article/73051
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    # dilation_rate 是指 kernel 中每个元素数的间隔, 如 (3,3) 的 kernel, dilation_rate 为 (6,6), 实际的 kernel_size 为 (13, 13)
    # 空洞卷积参考 https://www.zhihu.com/question/54149221
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg),
                 name='fc6')(pool5)
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg),
                 name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     name='conv9_2')(conv9_1)

    m = Model(input=x, output=conv9_2)
    m.summary()

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ############################################################################
    # Build the convolutional predictor layers on top of the base network
    ############################################################################

    # We precidt `n_classes` confidence values for each box,
    # hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg),
                                    name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg),
                           name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg),
                               name='conv9_2_mbox_conf')(conv9_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg),
                                   name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg),
                          name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg),
                              name='conv9_2_mbox_loc')(conv9_2)

    # Generate the anchor boxes
    # (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                             this_scale=scales[0],
                                             next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1,
                                             this_steps=steps[0],
                                             this_offsets=offsets[0],
                                             clip_boxes=clip_boxes,
                                             variances=variances,
                                             coords=coords,
                                             normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                    this_scale=scales[1],
                                    next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    this_steps=steps[1],
                                    this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    coords=coords,
                                    normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                        this_scale=scales[2],
                                        next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        this_steps=steps[2],
                                        this_offsets=offsets[2],
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                        this_scale=scales[3],
                                        next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        this_steps=steps[3],
                                        this_offsets=offsets[3],
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                        this_scale=scales[4],
                                        next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        this_steps=steps[4],
                                        this_offsets=offsets[4],
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width,
                                        this_scale=scales[5],
                                        next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        this_steps=steps[5],
                                        this_offsets=offsets[5],
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    # Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes),
                                             name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8),
                                                 name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    # Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    # mode == 'inference_fast'
    else:
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)

    if return_predictor_sizes:
        predictor_sizes = np.array([K.int_shape(conv4_3_norm_mbox_conf)[1:3],
                                    K.int_shape(fc7_mbox_conf)[1:3],
                                    K.int_shape(conv6_2_mbox_conf)[1:3],
                                    K.int_shape(conv7_2_mbox_conf)[1:3],
                                    K.int_shape(conv8_2_mbox_conf)[1:3],
                                    K.int_shape(conv9_2_mbox_conf)[1:3]])
        return model, predictor_sizes
    else:
        return model
