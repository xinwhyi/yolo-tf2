import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy

from yolo_tf2.core.models import YoloParser
from yolo_tf2.utils.common import get_boxes
from yolo_tf2.utils.dataset import create_tfrecord, read_tfrecord


def broadcast_iou(box_1, box_2):
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    w = tf.maximum(
        tf.minimum(box_1[..., 2], box_2[..., 2])
        - tf.maximum(box_1[..., 0], box_2[..., 0]),
        0,
    )
    h = tf.maximum(
        tf.minimum(box_1[..., 3], box_2[..., 3])
        - tf.maximum(box_1[..., 1], box_2[..., 1]),
        0,
    )
    int_area = w * h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def calculate_loss(anchors, total_classes, ignore_thresh):
    def yolo_loss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = get_boxes(
            y_pred, anchors, total_classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        obj_mask = tf.squeeze(true_obj, -1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))),
                axis=-1,
            ),
            (pred_box, true_box, obj_mask),
            fn_output_signature=tf.float32,
        )
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        xy_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        )
        wh_loss = (
            obj_mask
            * box_loss_scale
            * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        )
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class
        )
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss


def train():
    input_shape = 416, 416, 3
    anchors = np.array(
        [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ]
    ) / np.array(input_shape[:-1])
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    classes_file = '/content/yolo-data/classes.txt'
    train_tfrecord_path = '/content/yolo-data/dummy-train-dataset.tfrecord'
    valid_tfrecord_path = '/content/yolo-data/dummy-valid-dataset.tfrecord'
    classes = [c.strip() for c in open(classes_file)]
    shuffle_buffer_size = 512
    batch_size = 8
    parser = YoloParser(len(classes))
    model = parser.from_cfg(
        '/content/code/yolo-tf2/yolo_tf2/config/yolo4.cfg',
        input_shape,
        anchors=anchors,
        masks=masks,
    )
    labels = pd.read_csv('/content/yolo-data/bh_labels.csv')
    groups = [*labels.groupby('image')]
    random.shuffle(groups)
    max_boxes = max(i[1].shape[0] for i in groups)
    sep_idx = int(0.9 * len(groups))
    train_groups = groups[:sep_idx]
    valid_groups = groups[sep_idx:]
    create_tfrecord(train_tfrecord_path, train_groups)
    create_tfrecord(valid_tfrecord_path, valid_groups)
    training_dataset = read_tfrecord(
        train_tfrecord_path,
        classes_file,
        input_shape[:-1],
        max_boxes,
        shuffle_buffer_size,
        batch_size,
        anchors,
        masks,
    )
    valid_dataset = read_tfrecord(
        valid_tfrecord_path,
        classes_file,
        input_shape[:-1],
        max_boxes,
        shuffle_buffer_size,
        batch_size,
        anchors,
        masks,
    )
    loss = [calculate_loss(anchors[mask], len(classes), 0.5) for mask in masks]
    model.compile('adam', loss=loss)
    callbacks = [
        ModelCheckpoint(
            '/content/drive/MyDrive/yolo-new.tf',
            verbose=True,
            save_weights_only=True,
            save_best_only=True,
        ),
        EarlyStopping(patience=3, verbose=True),
    ]
    model.fit(training_dataset, validation_data=valid_dataset, callbacks=callbacks)
