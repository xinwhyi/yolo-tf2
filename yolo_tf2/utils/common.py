import logging
import xml.etree.ElementTree as Et
from logging import handlers
from pathlib import Path
from time import perf_counter

import imagesize
import numpy as np
import pandas as pd
import tensorflow as tf


def iou(relative_sizes, centroids, k):
    """
    Calculate intersection over union for relative box sizes.
    Args:
        relative_sizes: 2D array of relative box sizes.
        centroids: 2D array of shape(k, 2)
        k: int, number of clusters.

    Returns:
        IOU array.
    """
    n = relative_sizes.shape[0]
    box_area = relative_sizes[:, 0] * relative_sizes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))
    cluster_area = centroids[:, 0] * centroids[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))
    box_w_matrix = np.reshape(relative_sizes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(centroids[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)
    box_h_matrix = np.reshape(relative_sizes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(centroids[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)
    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def k_means(relative_sizes, k, distance_func=np.median, frame=None):
    """
    Calculate optimal anchor relative sizes.
    Args:
        relative_sizes: 2D array of relative box sizes.
        k: int, number of clusters.
        distance_func: function to calculate distance.
        frame: pandas DataFrame with the annotation data(for visualization purposes).

    Returns:
        Optimal relative sizes.
    """
    box_number = relative_sizes.shape[0]
    last_nearest = np.zeros((box_number,))
    centroids = relative_sizes[np.random.randint(0, box_number, k)]
    old_distances = np.zeros((relative_sizes.shape[0], k))
    iteration = 0
    while True:
        distances = 1 - iou(relative_sizes, centroids, k)
        print(
            f'Iteration: {iteration} Loss: '
            f'{np.sum(np.abs(distances - old_distances))}'
        )
        old_distances = distances.copy()
        iteration += 1
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            LOGGER.info(
                f'Generated {len(centroids)} anchors in ' f'{iteration} iterations'
            )
            return centroids, frame
        for anchor in range(k):
            centroids[anchor] = distance_func(
                relative_sizes[current_nearest == anchor], axis=0
            )
        last_nearest = current_nearest


def generate_anchors(width, height, centroids):
    """
    Generate anchors for image of size(width, height)
    Args:
        width: Width of image.
        height: Height of image.
        centroids: output of k-means.

    Returns:
        2D array of resulting anchors.
    """
    return (centroids * np.array([width, height])).astype(int)


def get_logger(log_file=None):
    """
    Initialize logger configuration.

    Returns:
        logger.
    """
    formatter = logging.Formatter(
        '%(asctime)s %(name)s.%(funcName)s +%(lineno)s: '
        '%(levelname)-8s [%(process)d] %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if log_file:
        file_handler = handlers.RotatingFileHandler(log_file, backupCount=10)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = get_logger()


def timer(logger):
    """
    Timer wrapper.
    logger: logging.RootLogger object

    Returns:
        timed
    """

    def timed(func):
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            total_time = perf_counter() - start_time
            if logger is not None:
                logger.info(f'{func.__name__} execution time: ' f'{total_time} seconds')
            if result is not None:
                return result

        return wrapper

    return timed


def get_boxes(detections, anchors, total_classes):
    """
    Convert yolo model output layer output to bounding boxes.
    Args:
        detections: Output layer tensor.
        anchors: Anchors as numpy array.
        total_classes: Total object classes in training dataset.

    Returns:
        boxes and probabilities.
    """
    grid_size = tf.shape(detections)[1]
    box_xy, box_wh, object_probability, class_probabilities = tf.split(
        detections, (2, 2, 1, total_classes), axis=-1
    )
    box_xy = tf.sigmoid(box_xy)
    object_probability = tf.sigmoid(object_probability)
    class_probabilities = tf.sigmoid(class_probabilities)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox, object_probability, class_probabilities, pred_box


def parse_voc(labels_dir, images_dir):
    """
    Parse XML files in VOC format found in `labels_dir`
    Args:
        labels_dir: Path to directory containing .xml files.
        images_dir: Path to directory containing images referenced in the .xml files.

    Returns:
        `pd.DataFrame` having `image`, `object_name`, `object_index`, `x0`, `y0`,
        `x1`, `y1` as columns.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    labels = []
    for fp in labels_dir.glob('*.xml'):
        label = {}
        root = Et.parse(fp).getroot()
        image_path = label['image'] = (
            images_dir / root.find('filename').text
        ).as_posix()
        w, h = imagesize.get(image_path)
        for tag in root.findall('object'):
            label['object_name'] = tag.find('name').text
            label['x0'] = float(tag.find('bndbox/xmin').text) / w
            label['y0'] = float(tag.find('bndbox/ymin').text) / h
            label['x1'] = float(tag.find('bndbox/xmax').text) / w
            label['y1'] = float(tag.find('bndbox/ymax').text) / h
        labels.append(label)
    labels = pd.DataFrame(labels)
    for i, object_name in enumerate(labels['object_name'].drop_duplicates().values):
        labels.loc[labels['object_name'] == object_name, 'object_index'] = int(i)
    labels['object_index'] = labels['object_index'].astype('int32')
    return labels


class YoloObject:
    """
    Utility base class for training, and detection classes.
    """

    def __init__(
        self,
        input_shape,
        classes,
        anchors,
        masks,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        """
        Initialize yolo most common settings.
        Args:
            input_shape: Input shape passed to `tf.image.resize` and
                `keras.engine.input_layer.Input`.
            classes: Path to .txt file containing object names \n delimited.
            anchors: Path to .txt file containing x,y pairs \n delimited.
            masks: Path to .txt file containing x,y,z triplets \n delimited.
            max_boxes: Maximum total boxes per image.
            iou_threshold: Percentage above which detections overlapping with
                ground truths are considered true positive.
            score_threshold: Detection confidence score above which a detection
                is considered relevant, others are discarded.
        """
        self.input_shape = input_shape
        self.classes = [c.strip() for c in open(classes)]
        self.class_map = {i: c.strip() for i, c in enumerate(self.classes)}
        self.anchors = np.array([xy.strip().split(',') for xy in open(anchors)]).astype(
            'int32'
        ) / np.array(input_shape[:-1])
        self.masks = np.array([xyz.strip().split(',') for xyz in open(masks)]).astype(
            'int32'
        )
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_boxes = max_boxes
