"""
A small 7-layer Keras model with SSD architecture. Also serves as a template to build arbitrary network architectures.

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
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def build_model(image_size,
                n_classes,
                mode='training',
                l2_regularization=0.0,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=(0.5, 1.0, 2.0),
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=(1.0, 1.0, 1.0, 1.0),
                coords='centroids',
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False):
    """
    Build a Keras model with SSD architecture, see references.

    The model consists of convolutional feature layers and a number of convolutional predictor layers that take their
    input from different feature layers.

    The model is fully convolutional.

    The implementation found here is a smaller version of the original architecture used in the paper (where the base
    network consists of a modified VGG-16 extended by a few convolutional feature layers), but of course it could easily
    be changed to an arbitrarily large SSD architecture by following the general design pattern used here.

    This implementation has 7 convolutional layers and 4 convolutional predictor layers that take their input from
    layers 4, 5, 6, and 7, respectively.

    Most of the arguments that this function takes are only needed for the anchor box layers.
    In case you're training the network, the parameters passed here must be the same as the ones used to set up
    `SSDBoxEncoder`.
    In case you're loading trained weights, the parameters passed here must be the same as the ones used to produce the
    trained weights.

    Some of these arguments are explained in more detail in the documentation of the `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Training currently works only with the TensorFlow backend (v1.0 or later).

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
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the largest will be
            linearly interpolated.
            Note that the second to last of the linearly interpolated scaling factors will actually be the scaling
            factor for the last predictor layer, while the last scaling factor is used for the second box for
            aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list/tuple, optional): The list/tuple of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all predictor layers.
            Note the original implementation uses more aspect ratios for some predictor layers and fewer for others.
            If you want to do that, too, then use the next argument instead.
        aspect_ratios_per_layer (list/tuple, optional): A nested list/tuple containing one aspect ratio list/tuple for
            each predictor layer. This allows you to set the aspect ratios for each predictor layer individually.
            If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored
            otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1.
            The first will be generated using the scaling factor for the respective layer,
            the second one will be generated using geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a nested tuple/list with as many elements as there are predictor layers.
            The elements can be either ints/floats or tuples of two ints/floats.
            These numbers represent for each predictor layer how many pixels apart the anchor box center points
            should be vertically and horizontally along the spatial grid over the image.
            If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a nested tuple/list with as many elements as there are predictor layers.
            The elements can be either floats or tuples of two floats.
            These numbers represent for each predictor layer how many pixels from the top and left boarders of the image
            the top-most and left-most anchor box center points should be as a fraction of `steps`.
            Note the last bit is important: The offsets are not absolute pixel values, but fractions of the step size
             specified in the `steps` argument.
            If the list contains floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two floats, then they represent `(vertical_offset, horizontal_offset)`.
            If no offsets are provided, then they will default to 0.5 of the step size, which is also the recommended
            setting.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        # UNCLEAR:
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model
            (i.e. this is not the input format of the ground truth labels).
            Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
            'minmax' for the format `(xmin, xmax, ymin, ymax)`, or
            'corners' for the format `(xmin, ymin, xmax, ymax)`.
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
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which
            the input image channels should be swapped.
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
        model: The Keras SSD model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion of the output tensor shape
            for each convolutional predictor layer. During training, the generator function needs this in order to
            transform the ground truth labels into tensors of identical structure as the output tensors of the model,
            which is in turn needed for the cost function.

    References:
        https://arxiv.org/abs/1512.02325v5
    """

    # The number of predictor conv layers in the network
    n_predictor_layers = 4
    # Make the internal name shorter.
    l2_reg = l2_regularization

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################
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
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if subtract_mean is not None:
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if divide_by_stddev is not None:
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    # 这里的架构和 vgg 区别还是很大的
    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv1')(x1)
    # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    # The next part is to add the convolutional predictor layers on top of the base network that we defined above.
    # Note that I use the term "base network" differently than the paper does.
    # To me, the base network is everything that is not convolutional predictor layers or anchor box layers.
    # In this case we'll have four predictor layers, but of course you could easily rewrite this into an arbitrarily
    # deep base network and add an arbitrary number of predictor layers on top of the base network by simply following
    # the pattern shown here.

    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers:
    # One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box,
    # hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
    classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
    classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
    classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes7')(conv7)
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
    boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
    boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                           aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                           aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                           aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors7')(boxes7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    # Reshape() 的参数是 target_shape, 并不包含 batch_size, 返回的 shape 为 (batch_size, ) + target_shape
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the associated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

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
    elif mode == 'inference_fast':
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
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        # 就是 feature_map 的 size
        predictor_sizes = np.array([K.shape(classes4)[1:3],
                                    K.shape(classes5)[1:3],
                                    K.shape(classes6)[1:3],
                                    K.shape(classes7)[1:3]])
        return model, predictor_sizes
    else:
        return model
