"""
Includes:
* Function to compute the IoU similarity for axis-aligned, rectangular, 2D bounding boxes
* Function for coordinate conversion for axis-aligned, rectangular, 2D bounding boxes

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


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    """
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place.
    Currently there are three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (np.array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'.
            If 'include', the border pixels belong to the boxes.
            If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong to the boxes,
            but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original tensor elsewhere.
    """
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include, exclude')

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        # Set cx
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 1]) / 2.0
        # Set cy
        tensor1[..., ind + 1] = (tensor[..., ind + 2] + tensor[..., ind + 3]) / 2.0
        # Set w
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind] + d
        # Set h
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 2] + d
    elif conversion == 'centroids2minmax':
        # Set xmin
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0
        # Set xmax
        tensor1[..., ind + 1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0
        # Set ymin
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0
        # Set ymax
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0
    elif conversion == 'corners2centroids':
        # Set cx
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 2]) / 2.0
        # Set cy
        tensor1[..., ind + 1] = (tensor[..., ind + 1] + tensor[..., ind + 3]) / 2.0
        # Set w
        tensor1[..., ind + 2] = tensor[..., ind + 2] - tensor[..., ind] + d
        # Set h
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 1] + d
    elif conversion == 'centroids2corners':
        # Set xmin = cx - w / 2
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0
        # Set ymin = cy - h / 2
        tensor1[..., ind + 1] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0
        # Set xmax = cx + w / 2
        tensor1[..., ind + 2] = tensor[..., ind] + tensor[..., ind + 2] / 2.0
        # Set ymax = cy + h / 2
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        # (xmin, xmax, ymin, ymax) <--> (xmin, ymin, xmax, ymax)
        tensor1[..., ind + 1] = tensor[..., ind + 2]
        tensor1[..., ind + 2] = tensor[..., ind + 1]
    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are "
            "'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', "
            "and 'corners2minmax'.")

    return tensor1


def convert_coordinates2(tensor, start_index, conversion):
    """
    A matrix multiplication implementation of `convert_coordinates()`.
    Supports only conversion between the 'centroids' and 'minmax' formats.

    This function is marginally slower on average than `convert_coordinates()`,
    probably because it involves more (unnecessary) arithmetic operations (unnecessary
    because the two matrices are sparse).

    For details please refer to the documentation of `convert_coordinates()`.
    """
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        m = np.array([[0.5, 0., -1., 0.],
                      [0.5, 0., 1., 0.],
                      [0., 0.5, 0., -1.],
                      [0., 0.5, 0., 1.]])
        tensor1[..., ind:ind + 4] = np.dot(tensor1[..., ind:ind + 4], m)
    elif conversion == 'centroids2minmax':
        m = np.array([[1., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [-0.5, 0.5, 0., 0.],
                      [0., 0., -0.5, 0.5]])  # The multiplicative inverse of the matrix above
        tensor1[..., ind:ind + 4] = np.dot(tensor1[..., ind:ind + 4], m)
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def intersection_area(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    """
    Computes the intersection areas of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'.
            In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible combinations
            of the `m` boxes in `boxes1` with the `n` boxes in `boxes2`.
            In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2` must be
            boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of length
            `m` where the i-th position contains the intersection area of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxes, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values with
        the intersection areas of the boxes in `boxes1` and `boxes2`.
    """

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, "
                         "but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively."
                         .format(boxes1.shape[1], boxes2.shape[1]))
    if mode not in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.", format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif coords not in {'minmax', 'corners'}:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # The number of boxes in `boxes1`
    m = boxes1.shape[0]
    # The number of boxes in `boxes2`
    n = boxes2.shape[0]

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    # coords == 'minmax'
    else:
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    # If border pixels are supposed to belong to the bounding boxes,
    # we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'include':
        d = 1
    # If border pixels are not supposed to belong to the bounding boxes,
    # we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include and exclude')

    # Compute the intersection areas.
    if mode == 'outer_product':
        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        # side_lengths[:, :, 0] 表示 width, side_lengths[:, :, 1] 表示 height, 相乘就表示 area
        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':
        # 假设此时 boxes1[:, [xmin, ymin]] shape 是 (1, 2), boxes2[:, [xmin, ymin]] 的 shape 是 (n, 2),
        # 在做 maximum, minimum 操作时, 先把 boxes1 广播, 变成 (n, 2) 在逐个比较每个位置上的元素, 返回结果的 shape 也是 (n, 2)
        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        return side_lengths[:, 0] * side_lengths[:, 1]


def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    """
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    """

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`
    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
    else:
        raise ValueError('`coords` must be corners or minmax')
    if border_pixels == 'half':
        d = 0
    # If border pixels are supposed to belong to the bounding boxes, we have to
    # add one pixel to any difference `x_max - x_min` or `y_max - y_min`.
    # 举个简单的例子, 假设 x_max = 5, x_min = 3, 那么 half 时 w = 2,include 时 w = 3,exclude 时 w = 1
    elif border_pixels == 'include':
        d = 1
    # If border pixels are not supposed to belong to the bounding boxes, we have to
    # subtract one pixel from any difference `x_max - x_min` or `y_max - y_min`.
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include and exclude')

    # Compute the intersection areas.
    if mode == 'outer_product':
        # For all possible box combinations, get the greater x_min and y_min values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))
        # For all possible box combinations, get the smaller x_max and y_max values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))
        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        # shape 为 (m, n)
        return side_lengths[:, :, 0] * side_lengths[:, :, 1]
    elif mode == 'element-wise':
        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])
        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    """
    Computes the intersection-over-union similarity (also known as Jaccard similarity) of two sets of axis-aligned 2D
    rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible combinations of the boxes in
        `boxes1` and `boxes2`.
    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation of the `mode` argument
        for details.

    Arguments:
        boxes1 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(x_min, x_max, y_min, y_max)`, or 'corners' for the format
            `(x_min, y_min, x_max, y_max)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'.
            In 'outer_product' mode, returns an `(m,n)` matrix with the IoU overlaps for all possible combinations of
            the `m` boxes in `boxes1` with the `n` boxes in `boxes2`.
            In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2` must be
            broadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'.
            Note 想了一下, 如果 boxes 的坐标被 normalize 过, border_pixels 只能为 half 否则加 1 减 1, 对结果的影响都很大.
            If 'include', the border pixels belong to the boxes.
            If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong to the boxes, but not the
            other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`.
        0 means there is no overlap between two given boxes,
        1 means their coordinates are identical.
    """

    #########################################################################################
    # Check for arguments' validation
    #########################################################################################
    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, "
                         "but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                            boxes1.shape[1], boxes2.shape[1]))
    if mode not in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif coords not in {'minmax', 'corners'}:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    #########################################################################################
    # Compute the IoU
    #########################################################################################

    # Compute the intersection areas.
    # Adam 用 intersection_area 代替 intersection_area_
    intersection_areas = intersection_area(boxes1, boxes2, coords=coords, mode=mode)
    # intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)
    # The number of boxes in `boxes1`
    m = boxes1.shape[0]
    # The number of boxes in `boxes2`
    n = boxes2.shape[0]

    # Compute the union areas.
    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    # coords == 'minmax'
    else:
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3
    if border_pixels == 'half':
        d = 0
    # If border pixels are supposed to belong to the bounding boxes,
    # we have to add one pixel to any difference `x_max - x_min` or `y_max - y_min`.
    elif border_pixels == 'include':
        d = 1
    # If border pixels are not supposed to belong to the bounding boxes,
    # we have to subtract one pixel from any difference `x_max - x_min` or `y_max - y_min`.
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include and exclude')

    if mode == 'outer_product':
        # 每一行 n 个相同的数, 表示 boxes1 中某个 box 的 area, 一共 m 行
        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        # 每一行 n 个不同的数, 表示 boxes2 中所有 boxes 的 area, m 行都相同
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))
    # mode == 'element-wise'
    else:
        # 假设 boxes1 的 shape 为 (1, 4) 那么 boxes1_areas 的 shape 是 (1,)
        # boxes2 的 shape 为 (n, 4), 那么 boxes2_areas 的 shape 就是 (n, )
        # 后面两者相加时做一次广播, 相加结果的 shape 也是 (n, )
        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    # boxes1_areas + boxes2_area 的 shape 为 (m,n), (m,) or (n,)
    # 如果是 (m,n), 每一行表示 boxes1 中某个 box 的 area 和 boxes2 中所有 box 的 area 的和
    # 如果是 (m,), 每一个元素表示 boxes1 中某个 box 的 area 和 boxes2 中 box 的 area 的和
    # 如果是 (n,), 每一个元素表示 boxes2 中某个 box 的 area 和 boxes1 中 box 的 area 的和
    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
