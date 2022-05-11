import tensorflow as tf
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from ml_utils.tensorflow.training import Trainer
from yolo_tf2.core.models import YoloParser
from yolo_tf2.utils.common import YoloObject, get_boxes
from yolo_tf2.utils.dataset import create_tfrecord, read_tfrecord


def broadcast_iou(box_1, box_2):
    """
    Calculate iou given ground truth and predicted boxes.
    Args:
        box_1: Ground truth boxes as tensor.
        box_2: Detected boxes as tensor.

    Returns:
        IOU tensor.
    """
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


def calculate_loss(anchors, total_classes, iou_threshold):
    """
    Loss function wrapper which result is passed to `tf.keras.Model.compile`.
    Args:
        anchors: Anchors as numpy array.
        total_classes: Total object classes in training dataset.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.
    Returns:
        `yolo_loss` loss function.
    """

    def yolo_loss(y_true, y_pred):
        """
        Loss function passed to `tf.keras.Model.compile`.
        Args:
            y_true: Ground truth tensor.
            y_pred: Detections tensor.

        Returns:
            Total loss.
        """
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
        ignore_mask = tf.cast(best_iou < iou_threshold, tf.float32)
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


class YoloTrainer(Trainer, YoloObject):
    """
    Yolo training tool.
    """

    def __init__(
        self,
        input_shape,
        batch_size,
        classes,
        model_configuration,
        anchors,
        masks,
        max_boxes,
        iou_threshold,
        score_threshold,
        v4=False,
        **kwargs,
    ):
        """
        Initialize training settings.
        Args:
            input_shape: Input shape passed to `tf.image.resize` and
                `keras.engine.input_layer.Input`.
            batch_size: Batch size passed to `tf.data.Dataset.batch`
            classes: Path to .txt file containing object names \n delimited.
            model_configuration: Path to .cfg DarkNet file.
            anchors: Path to .txt file containing x,y pairs \n delimited.
            masks: Path to .txt file containing x,y,z triplets \n delimited.
            max_boxes: Maximum total boxes per image.
            iou_threshold: Percentage above which detections overlapping with
                ground truths are considered true positive.
            score_threshold: Detection confidence score above which a detection
                is considered relevant, others are discarded.
            **kwargs: kwargs passed to `ml_utils.tensorflow.training.Trainer`
        """
        Trainer.__init__(self, input_shape, batch_size, **kwargs)
        YoloObject.__init__(
            self,
            input_shape,
            classes,
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
        )
        self.classes_file = classes
        self.model_configuration = model_configuration
        self.v4 = v4

    def write_tfrecord(self, fp, data, shards):
        """
        Write labeled examples to .tfrecord file(s).
        Args:
            fp: Path to output .tfrecord file.
            data: A list of `pd.DataFrame.groupby` result.
            shards: Total .tfrecord output files.

        Returns:
            None
        """
        create_tfrecord(fp, data, shards, self.delete_tfrecord_images, self.verbose)

    def read_tfrecord(self, fp):
        """
        Read .tfrecord file.
        Args:
            fp: Path to .tfrecord file.

        Returns:
            `tf.data.Dataset`.
        """
        return read_tfrecord(
            fp,
            self.classes_file,
            self.input_shape[:-1],
            self.max_boxes,
            self.shuffle_buffer_size,
            self.batch_size,
            self.anchors,
            self.masks,
        )

    def calculate_loss(self):
        """
        Get yolo output layer loss functions.
        Returns:
            None
        """
        return [
            calculate_loss(self.anchors[mask], len(self.classes), self.iou_threshold)
            for mask in self.masks
        ]

    def create_model(self):
        """
        Create yolo `tf.keras.Model`.
        Returns:
            `tf.keras.Model`.
        """
        parser = YoloParser(len(self.classes))
        return parser.from_cfg(self.model_configuration, self.input_shape, self.v4)
