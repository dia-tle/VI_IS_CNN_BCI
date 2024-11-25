""" Functions for IS training CNN


"""

import os
import sys
from typing import List, Tuple
import mne
import numpy as np
import tensorflow as tf
import pickle
import scipy.io


def convert_labels(train_set: Tuple[np.ndarray, np.ndarray],
                   val_set: Tuple[np.ndarray, np.ndarray],
                   test_set: Tuple[np.ndarray, np.ndarray], event_ids: List[int]) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """ Convert the labels to 0s and 1s to enable one-hot labels encoding for Deep Learning algorithm.

    Args:
    train_set (Tuple[np.ndarray, np.ndarray]): The training set containing features (X) and labels (y)
    val_set (Tuple[np.ndarray, np.ndarray]): The validation set containing features (X) and labels (y)
    test_set (Tuple[np.ndarray, np.ndarray]): The test set containing features (X) and labels (y)

    Returns:
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Tuple containing the train, validation and test sets
    """
    if val_set is None:
        val_set = (np.empty(shape=0), np.empty(shape=0))

    if test_set is None:
        test_set = (np.empty(shape=0), np.empty(shape=0))

    # Convert labels from original values to 0 and 1
    for set_type in [train_set, val_set, test_set]:
        r_relevant_idx = np.where(set_type[-1] == event_ids[0])[0]
        p_relevant_idx = np.where(set_type[-1] == event_ids[1])[0]

        set_type[-1][r_relevant_idx] = 0
        set_type[-1][p_relevant_idx] = 1

    return train_set, val_set, test_set


# Generate augmented data - Random Cropped training
def generate_segmented_epochs(X: np.ndarray,
                              y: np.ndarray,
                              srate: int,
                              overlap_factor: float,
                              segment_length: int) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): The epoched data in shape: nEpochs, nChans, nTime
        y (np.ndarray): The label array of shape nEpochs
        srate (int): the sampling rate
        segment_length (int, optional): The length of the new segment in seconds. Defaults to 1.
        overlap_factor (float, optional): The overlapping factor between segments. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The segmented X and y arrays
    """

    n_epochs, n_chans, n_time = X.shape

    assert segment_length <= n_time, "segment_length cannot be greater than the number of time points in epochs_data"
    assert X.ndim == 3, "X must be a 3-dimensional array"
    assert y.ndim == 1, "y must be a 1-dimensional array"
    assert len(X) == len(y), "X and y must have the same length"

    segment_length = int(segment_length * srate)  # get the index based on time lenght (e.g. 1s)
    overlap_length = int(segment_length * overlap_factor)
    stride = segment_length - overlap_length

    n_segments = (n_time - segment_length) // stride + 1  # number of segments per epoch
    total_segments = n_segments * n_epochs

    # Check if the remaining time points are enough for an additional segment
    remaining_time = n_time - segment_length - (n_segments - 1) * stride

    total_segments = n_segments * n_epochs + (remaining_time >= segment_length)

    segmented_X = np.zeros(shape=(total_segments, n_chans, segment_length))
    segmented_y = np.zeros(shape=(total_segments), dtype=float)

    for n_epoch in range(n_epochs):
        epoch = X[n_epoch]
        label = y[n_epoch]

        for n_segment in range(n_segments):
            start_idx = n_segment * stride
            end_idx = start_idx + segment_length

            # Adjust the segment end if it exceeds the available time points
            if end_idx > n_time:
                end_idx = n_time

            segmented_X[n_epoch * n_segments + n_segment, :, :] = epoch[:, start_idx:end_idx]
            segmented_y[n_epoch * n_segments + n_segment] = label

    segmented_y = segmented_y.astype(float)  # ensure that the label data type is of type float

    return segmented_X, segmented_y


def to_one_hot(y, by_sub=False):
    if by_sub:
        new_array = np.array(["nan" for nan in range(len(y))])
        for index, label in enumerate(y):
            new_array[index] = ''.join([i for i in label if not i.isdigit()])
    else:
        new_array = y.copy()
    total_labels = np.unique(new_array)
    mapping = {}
    for x in range(len(total_labels)):
        mapping[total_labels[x]] = x
    for x in range(len(new_array)):
        new_array[x] = mapping[new_array[x]]

    return tf.keras.utils.to_categorical(new_array)
