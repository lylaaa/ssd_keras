"""
Utilities to match ground truth boxes to anchor boxes.

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


def match_bipartite_greedy(weight_matrix):
    """
    Returns a bipartite matching according to the given weight matrix.

    The algorithm works as follows:

    Let the first axis of `weight_matrix` represent ground truth boxes and the second axis anchor boxes.
    The ground truth box that has the greatest similarity with any anchor box will be matched first,
    then out of the remaining ground truth boxes, the ground truth box that has the greatest similarity with any of the
    remaining anchor boxes will be matched second, and so on.
    先找到最大 overlap 的 gt_box 和 anchor_box
    That is, the ground truth boxes will be matched in descending order by maximum similarity with any of the
    respectively remaining anchor boxes.
    The runtime complexity is O(m^2 * n), where `m` is the number of ground truth boxes and `n` is the number of
    anchor boxes.
    # 找一行中的最大 iou, O(n), 那么 m 行最大 iou 的时间复杂度就是 O(m * n)
    # 然后再在 m 个 最大 iou 中找到最最大的 iou, O(m^2 * n)

    Arguments:
        weight_matrix (np.array): A 2D Numpy array that represents the weight matrix for the matching process.
            If `(m,n)` is the shape of the weight matrix, it must be `m <= n`. 因为必须为每个 gt_box 找到一个匹配的 anchor.
            The weights can be integers or floating point numbers.
            The matching process will maximize, i.e. larger weights are preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index along the first axis.
        就是为每个 gt_box 找一个匹配的 anchor_box, 返回 array 的 shape 为 (m, )
    """
    # We'll modify this array.
    weight_matrix = np.copy(weight_matrix)
    num_ground_truth_boxes = weight_matrix.shape[0]
    # Only relevant for fancy-indexing below.
    all_gt_indices = list(range(num_ground_truth_boxes))

    # This 1D array will contain for each ground truth box the index of the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps:
        # First, reduce over the anchor boxes and then reduce over the ground truth boxes.
        # Reduce along the anchor box axis.
        # 先为每个 gt_box 找到最大 iou 的 anchor_box
        anchor_indices = np.argmax(weight_matrix, axis=1)
        # shape 为 (num_gt_boxes, )
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        # Reduce along the ground truth box axis.
        # 然后再在这 num_gt_boxes 个 iou 中找到最大的
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        # Set the match.
        matches[ground_truth_index] = anchor_index

        # Set the row of the matched ground truth box to all zeros, because it has found the matched anchor_box
        # Set the column of the matched anchor box to all zeros, because they will never be the best matches
        # for any other boxes.
        # 最后设置该 gt_box 和其他所有的 anchor_box 的 iou 为 0, 设置该 anchor_box 和其他所有的 gt_box 的 iou 为 0.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches


def match_multi(weight_matrix, threshold):
    """
    Matches all elements along the second axis of `weight_matrix` to their best matches along the first axis subject to
    the constraint that the weight of a match must be greater than or equal to `threshold` in order to produce a match.
    一列一列地看, 找出某一列中的最大值, 如果最大值大于 threshold, 那么认为是 match 的

    If the weight matrix contains elements that should be ignored, the row or column representing the respective element
    should be set to a value below `threshold`. 在调用前已经把 match_bipartite_greedy 中匹配的 anchor_box 整列设置为 0 了.

    Arguments:
        weight_matrix (np.array): A 2D Numpy array that represents the weight matrix for the matching process.
            If `(m,n)` is the shape of the weight matrix, it must be `m <= n`.
            The weights can be integers or floating point numbers.
            The matching process will maximize, i.e. larger weights are preferred over smaller weights.
            # 某一列中只取最大的 overlap 值, 来和 threshold 比较, 所以一个 anchor_box 最多和一个 gt_box 匹配.
        threshold (float): A float that represents the threshold (i.e. lower bound) that must be met by a pair of
            elements to produce a match.

    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first array contains the indices
        along the first axis of `weight_matrix`, the second array contains the indices along the second axis.
    """

    num_anchor_boxes = weight_matrix.shape[1]
    # Only relevant for fancy-indexing below.
    all_anchor_indices = list(range(num_anchor_boxes))

    # Find the best ground truth match for every anchor box.
    # Array of shape (weight_matrix.shape[1],), 也就是 (num_anchor_boxes, )
    # 每一个元素表示每一个 anchor_box 对应最大 iou 的 gt_box 的 id
    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    # Array of shape (weight_matrix.shape[1],), 也就是 (num_anchor_boxes, )
    # 每一个元素表示每一个 anchor_box 和所有 gt_boxes 对应的最大 iou
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]

    # Filter out the matches with a weight below the threshold.
    # shape 为 (num_anchor_boxes, ), 每一个元素为 True 表示该 anchor_box 是满足条件的, 就是和 gt_boxes 的最大 iou 是大于阈值的
    # np.nonzero() 返回 tuple, tuple 的每一个值表示 非 zero 的值的在某一个维度上的 indices 和 np.where 比较像
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    # shape 为 (num_anchor_boxes, ), 每一个元素表示符合条件的 anchor_box 对应的 gt_box 的 id
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met
