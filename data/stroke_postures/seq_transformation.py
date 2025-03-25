# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import logging
import os
import os.path as osp
import pickle

import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def frame_translation(skes_joints, skes_name, frames_cnt):
    """
    沒有用到
    """
    nan_logger = logging.getLogger("nan_skes")
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info("{}\t{}\t{}".format("Skeleton", "Frame", "Joints"))

    for idx, ske_joints in tqdm(enumerate(skes_joints)):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    """
    沒有用到
    """
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info("{}\t{:^5}\t{}".format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]


def load_data(dir_path):
    annotation_paths = []
    folders = [f for f in sorted(os.listdir(dir_path)) if os.path.isdir(os.path.join(dir_path, f))]
    for folder in folders:
        annotation_dir = os.path.join(dir_path, folder, "annotation")
        annotation_paths += sorted([os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir)])

    skes_joints = []
    frames_cnt = []
    labels = []
    for annotation_path in annotation_paths:
        print("dealing {}".format(annotation_path))
        with open(annotation_path, "r") as f:
            data = json.load(f)
        assert "skeletons" in data, f"skeletons not found in {annotation_path}"

        ske_joints = np.array(data["skeletons"], dtype=np.float32)
        num_frames = int(data["length"])
        label = int(data["label"])

        if num_frames == 0:
            continue

        if not np.any(ske_joints != 0.0):
            continue

        assert num_frames == ske_joints.shape[0]
        assert ske_joints.shape[0] != 0
        assert num_frames != 0

        skes_joints.append(ske_joints)
        frames_cnt.append(num_frames)
        labels.append(label)

    frames_cnt = np.array(frames_cnt)
    labels = np.array(labels)
    return skes_joints, frames_cnt, labels


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 34), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        flattened_ske_joints = ske_joints.reshape(num_frames, -1)
        aligned_skes_joints[idx, :num_frames] = flattened_ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    encoder = OneHotEncoder(sparse_output=False)
    labels = labels.reshape(-1, 1)
    return encoder.fit_transform(labels)


def split_dataset(train_skes_joints, train_labels, valid_skes_joints, valid_labels):
    train_x = train_skes_joints
    train_y = one_hot_vector(train_labels)
    test_x = valid_skes_joints
    test_y = one_hot_vector(valid_labels)

    save_name = "jhmdb.npz"
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)


if __name__ == "__main__":
    train_skes_joints, train_frames_cnt, train_labels = load_data("train")
    train_skes_joints = align_frames(train_skes_joints, train_frames_cnt)  # aligned to the same frame length
    valid_skes_joints, valid_frames_cnt, valid_labels = load_data("valid")
    valid_skes_joints = align_frames(valid_skes_joints, valid_frames_cnt)  # aligned to the same frame length
    split_dataset(train_skes_joints, train_labels, valid_skes_joints, valid_labels)
