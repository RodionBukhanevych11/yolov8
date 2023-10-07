import math
import warnings
from pathlib import Path

import numpy as np
import torch

from typing import List, Tuple


def bbox_iou_numpy(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_metrics_per_frame(metrics, predicts, targets, frame_target_path, iou_th):
    detected_target = False
    for predict in predicts:
        predicted_label = int(predict[-1])
        predicted_box = predict[:4]
        iou_all = bbox_iou_numpy(targets[:4], predicted_box)
        if iou_all >= iou_th:
            detected_target = True
            # check if target was already detected
            target_label = int(targets[-1])
            target_area = (targets[2] - targets[0]) * (targets[3] - targets[1])
            target_area = target_area.item()
            # increment tp if labels are equal, otherwise increment fp
            if predicted_label == target_label:
                for range_key in list(metrics.keys()):
                    range_int = list(map(int, range_key.split("_")))
                    if range_int[0] <= target_area <= range_int[1]:
                        metrics[range_key]['tp'].append(frame_target_path)
                        break
            else:
                for range_key in list(metrics.keys()):
                    range_int = list(map(int, range_key.split("_")))
                    if range_int[0] <= target_area <= range_int[1]:
                        metrics[range_key]['fp'].append(frame_target_path)
                        break
        else:
            for range_key in list(metrics.keys()):
                range_int = list(map(int, range_key.split("_")))
                if range_int[0] <= target_area <= range_int[1]:
                    metrics[range_key]['fp'].append(frame_target_path)
                    break
    
    # get not detected matched targets from all targets and matched ones
    if not detected_target:
        # increment fn for each not detected matched targets
        target_area = (targets[2] - targets[0]) * (targets[3] - targets[1])
        target_area = target_area.item()
        for range_key in list(metrics.keys()):
            range_int = list(map(int, range_key.split("_")))
            if range_int[0] <= target_area <= range_int[1]:
                metrics[range_key]['fn'].append(frame_target_path)
    return metrics


def get_levelbased_metrics(
    batch_targets,
    batch_targets_paths,
    batch_predicts,
    metrics,
    iou_th,
    conf_th
    ):
    assert len(batch_predicts) == len(batch_targets)
    for frame_index, frame_predicts in enumerate(batch_predicts):
        # get predicts higher with prob higher than th
        frame_predicts = frame_predicts[frame_predicts[:, -2] >= conf_th]
        frame_predicts = np.array(frame_predicts)
        # get targets labels, paths correspond to current frame index
        frame_targets = batch_targets[frame_index].numpy()
        frame_target_path = batch_targets_paths[frame_index]
        # calculate metrics for area_based_target_name
        metrics = get_metrics_per_frame(metrics, frame_predicts, frame_targets, frame_target_path, iou_th) 

    return metrics
