import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from yolo_tf2.utils.common import YoloObject, get_boxes


def get_nms(outputs, max_boxes, iou_threshold, score_threshold):
    """
    Filter out overlapping detections by applying non-maximum suppression.
    Args:
        outputs: The result of `yolo_tf2.utils.common.get_boxes` applied on
            `tf.keras.Model.predict` output.
        max_boxes: Maximum total boxes per image.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.
        score_threshold: Detection confidence score above which a detection
            is considered relevant, others are discarded.

    Returns:
        boxes, scores, classes, total valid detections which results from
        `tf.image.combined_non_max_suppression`
    """
    boxes, confidences, type_ = [], [], []
    for output in outputs:
        boxes.append(
            tf.reshape(
                output[0],
                (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1]),
            )
        )
        confidences.append(
            tf.reshape(
                output[1],
                (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1]),
            )
        )
        type_.append(
            tf.reshape(
                output[2],
                (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1]),
            )
        )
    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(confidences, axis=1)
    class_probabilities = tf.concat(type_, axis=1)
    scores = confidence * class_probabilities
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes,
        max_total_size=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return boxes, scores, classes, valid_detections


def draw_boxes(image, colors, detections, show_scores=True, font_scale=0.6):
    """
    Draw bounding boxes over image, displaying detected classes and confidence
    scores in distinct colors.
    Args:
        image: Image as `np.ndarray`.
        colors: Mapping of object name -> color.
        detections: Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        show_scores: If False, object names and confidence scores will not
            be displayed above each bounding box.
        font_scale: `fontScale` parameter passed to `cv2.cv2.putText`

    Returns:
        None
    """
    for i, row in detections.iterrows():
        x0 = int(row['x0'] * image.shape[1])
        y0 = int(row['y0'] * image.shape[0])
        x1 = int(row['x1'] * image.shape[1])
        y1 = int(row['y1'] * image.shape[0])
        color = colors.get(row['object_name'])
        cv2.rectangle(
            image,
            (x0, y0),
            (x1, y1),
            color,
            1,
        )
        if show_scores:
            cv2.putText(
                image,
                f"{row['object_name']}-{round(row['score'], 2)}",
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                font_scale,
                color,
            )


class YoloDetector(YoloObject):
    """
    Yolo detection tool.
    """

    def __init__(self, input_shape, classes, anchors, masks, **kwargs):
        """
        Initialize detection settings.
        Args:
            input_shape: Input shape passed to `tf.image.resize`
            classes: Path to .txt file containing object names \n delimited.
            anchors: Path to .txt file containing x,y pairs \n delimited.
            masks: Path to .txt file containing x,y,z triplets \n delimited.
            **kwargs: kwargs passed to `yolo_tf2.utils.common.YoloObject`
        """
        super(YoloDetector, self).__init__(
            input_shape, classes, anchors, masks, **kwargs
        )

    def detections_to_boxes(self, detections):
        """
        Convert `tf.keras.Model.predict` output to bounding boxes and apply NMS.
        Args:
            detections: The result of `tf.keras.Model.predict`.

        Returns:
            Bounding boxes after NMS.
        """
        total_classes = len(self.classes)
        output_0, output_1, output_2 = detections
        boxes_0 = get_boxes(output_0, self.anchors[self.masks[0]], total_classes)
        boxes_1 = get_boxes(output_1, self.anchors[self.masks[1]], total_classes)
        boxes_2 = get_boxes(output_2, self.anchors[self.masks[2]], total_classes)
        return get_nms(
            [box[:3] for box in [boxes_0, boxes_1, boxes_2]],
            self.max_boxes,
            self.iou_threshold,
            self.score_threshold,
        )

    def detections_to_df(
        self,
        image_paths,
        detections,
    ):
        """
        Convert `tf.keras.Model.predict` output, apply NMS, and get results as `pd.DataFrame`.
        Args:
            image_paths: An iterable of image paths.
            detections: The result of `tf.keras.Model.predict`.

        Returns:
            Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        """
        boxes = self.detections_to_boxes(detections)
        outputs = []
        for image, boxes, scores, classes, total in zip(image_paths, *boxes):
            df = pd.DataFrame(boxes[:total], columns=['x0', 'y0', 'x1', 'y1'])
            df['image'] = image
            df['object_index'] = classes[:total]
            df['object_name'] = df['object_index'].apply(self.class_map.get)
            df['score'] = scores[:total]
            outputs.append(df)
        return pd.concat(outputs)

    def detect_images(
        self,
        image_paths,
        model,
        **kwargs,
    ):
        """
        Perform detections given images and model and get results as `pd.DataFrame`.
        Args:
            image_paths: An iterable of image paths.
            model: `tf.keras.Model`
            **kwargs: kwargs passed to `tf.keras.Model.predict`

        Returns:
            Detections as `pd.DataFrame`, the result of `detections_to_df`
        """
        adjusted_images = []
        for image in image_paths:
            with open(image, 'rb') as raw_image:
                image = tf.image.decode_png(raw_image.read(), channels=3)
            image = tf.image.resize(image, self.input_shape[1:-1])
            adjusted_images.append(image)
        detections = model.predict(tf.stack(adjusted_images) / 255, **kwargs)
        return self.detections_to_df(image_paths, detections)

    def detect_vid(
        self,
        src,
        dest,
        model,
        colors,
        codec='mp4v',
        display=False,
        **kwargs,
    ):
        """
        Perform detection on video and save the resulting video.
        Args:
            src: Path to video to predict.
            dest: Path to output video.
            model: `tf.keras.Model`
            colors: Mapping of object name -> color.
            codec: Codec passed to `cv2.VideoWriter`.
            display: If True, the resulting video will be rendered during detection.
            **kwargs: kwargs passed to `tf.keras.Model.predict`

        Returns:
            None
        """
        vid = cv2.VideoCapture(src)
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        current = 1
        codec = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(dest, codec, fps, (width, height))
        while vid.isOpened():
            _, frame = vid.read()
            detections = self.detect_images(
                np.expand_dims(frame, 0),
                model,
                **kwargs,
            )
            draw_boxes(frame, colors, detections)
            writer.write(frame)
            completed = f'{(current / length) * 100}% completed'
            print(
                f'\rframe {current}/{length}\tdetections: '
                f'{len(detections)}\tcompleted: {completed}',
                end='',
            )
            if display:
                cv2.destroyAllWindows()
                cv2.imshow(f'frame {current}', frame)
            current += 1
            if cv2.waitKey(1) == ord('q'):
                print(
                    f'Video detection aborted {current}/{length} ' f'frames completed'
                )
                break
