import logging
import xml.etree.ElementTree as Et
from logging import handlers
from pathlib import Path
from time import perf_counter

import imagesize
import pandas as pd
import tensorflow as tf


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


def get_boxes(pred, anchors, total_classes):
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, object_probability, class_probabilities = tf.split(
        pred, (2, 2, 1, total_classes), axis=-1
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


def parse_voc(label_dir, image_dir):
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    labels = []
    for fp in label_dir.glob('*.xml'):
        label = {}
        root = Et.parse(fp).getroot()
        image_path = label['image'] = (
            image_dir / root.find('filename').text
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
