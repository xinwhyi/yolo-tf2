import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import tensorflow as tf

from yolo_tf2.core.models import BaseModel
from yolo_tf2.utils.common import (
    LOGGER,
    activate_gpu,
    get_abs_path,
    get_detection_data,
    timer,
    transform_images,
)


class Detector(BaseModel):
    """Tool for detection on photos/videos"""

    def __init__(
        self,
        input_shape,
        model_configuration,
        classes_file,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        """
        Initialize detection settings.

        Args:
            input_shape: tuple, (n, n, c)
            model_configuration: Path to DarkNet cfg file.
            classes_file: File containing class names \n delimited.
            anchors: numpy array of (w, h) pairs.
            masks: numpy array of masks.
            max_boxes: Maximum boxes of the tfrecords provided(if any) or
                maximum boxes setting.
            iou_threshold: float, values less than the threshold are ignored.
            score_threshold: float, values less than the threshold are ignored.
        """
        self.class_names = [item.strip() for item in open(classes_file)]
        self.box_colors = {
            class_name: color
            for class_name, color in zip(
                self.class_names,
                [
                    list(np.random.random(size=3) * 256)
                    for _ in range(len(self.class_names))
                ],
            )
        }
        super().__init__(
            input_shape=input_shape,
            model_configuration=model_configuration,
            classes=len(self.class_names),
            anchors=anchors,
            masks=masks,
            max_boxes=max_boxes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        activate_gpu()

    def detect_image(self, image_data, image_name):
        """
        Given an image, get detections for the image.
        Args:
            image_data: image as numpy array/tf.Tensor
            image_name: str, name of the image

        Returns:
            pandas DataFrame with detections found.
        """
        image = tf.expand_dims(image_data, 0)
        resized = transform_images(image, self.input_shape[0])
        out = self.inference_model.predict(resized)
        if isinstance(image_data, np.ndarray):
            adjusted = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:
            adjusted = cv2.cvtColor(image_data.numpy(), cv2.COLOR_RGB2BGR)
        detections = get_detection_data(
            adjusted,
            image_name,
            out,
            self.class_names,
        )
        return detections, adjusted

    def draw_on_image(self, adjusted, detections):
        """
        Draw bounding boxes over the image.
        Args:
            adjusted: BGR image.
            detections: pandas DataFrame containing detections

        Returns:
            None
        """
        for index, row in detections.iterrows():
            img, obj, x1, y1, x2, y2, score, *_ = row.values
            color = self.box_colors.get(obj)
            cv2.rectangle(adjusted, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                adjusted,
                f'{obj}-{round(score, 2)}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.6,
                color,
                1,
            )

    def predict_on_image(self, image_path, output_dir=None):
        """
        Detect, draw detections and save result to output folder.
        Args:
            image_path: Path to image.
            output_dir: Path to output dir, defaults to output/detections

        Returns:
            Output path.
        """
        image_path = get_abs_path(image_path, verify=True)
        image_name = os.path.basename(image_path)
        image_data = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
        detections, adjusted = self.detect_image(image_data, image_name)
        self.draw_on_image(adjusted, detections)
        if output_dir:
            output_path = get_abs_path(
                output_dir, f'predicted-{image_name}', create_parents=True
            )
        else:
            output_dir = get_abs_path('output', 'detections')
            output_path = get_abs_path(
                output_dir, f'predicted-{image_name}', create_parents=True
            )
        cv2.imwrite(output_path, adjusted)
        return output_path

    @timer(LOGGER)
    def predict_photos(
        self, photos, trained_weights, batch_size=32, workers=16, output_dir=None
    ):
        """
        Predict a list of image paths and save results to output folder.
        Args:
            photos: A list of image paths.
            trained_weights: .weights or .tf file
            batch_size: Prediction batch size.
            workers: Parallel predictions.
            output_dir: Path to output dir, defaults to output/detections

        Returns:
            None
        """
        self.create_models(reverse_v4=True if trained_weights.endswith('tf') else False)
        self.load_weights(get_abs_path(trained_weights, verify=True))
        to_predict = photos.copy()
        saved_paths = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            predicted = 1
            total_photos = len(photos)
            while to_predict:
                current_batch = [
                    to_predict.pop()
                    for _ in range(min(batch_size, len(to_predict)))
                    if to_predict
                ]
                future_predictions = {
                    executor.submit(
                        self.predict_on_image,
                        image,
                        output_dir,
                    ): image
                    for image in current_batch
                }
                for future_prediction in as_completed(future_predictions):
                    saved_path = future_prediction.result()
                    saved_paths.add(saved_path)
                    completed = f'{predicted}/{total_photos}'
                    current_image = future_predictions[future_prediction]
                    percent = (predicted / total_photos) * 100
                    print(
                        f'\rpredicting {os.path.basename(current_image)} '
                        f'{completed}\t{percent}% completed',
                        end='',
                    )
                    predicted += 1
        print()
        for saved_path in saved_paths:
            LOGGER.info(f'Saved prediction: {saved_path}')

    @timer(LOGGER)
    def detect_video(
        self, video, trained_weights, codec='mp4v', display=False, output_dir=None
    ):
        """
        Perform detection on a video, stream(optional) and save results.
        Args:
            video: Path to video file.
            trained_weights: .tf or .weights file
            codec: str ex: mp4v
            display: If True, detections will be displayed during
                the detection operation.
            output_dir: Path to output dir, defaults to output/detections

        Returns:
            None
        """
        self.create_models(reverse_v4=True if trained_weights.endswith('tf') else False)
        self.load_weights(trained_weights)
        vid = cv2.VideoCapture(video)
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        current = 1
        codec = cv2.VideoWriter_fourcc(*codec)
        out = (
            get_abs_path(output_dir, 'predicted_vid.mp4', create_parents=True)
            if output_dir
            else get_abs_path(
                'output', 'detections', 'predicted_vid.mp4', create_parents=True
            )
        )
        writer = cv2.VideoWriter(out, codec, fps, (width, height))
        while vid.isOpened():
            _, frame = vid.read()
            detections, adjusted = self.detect_image(frame, f'frame_{current}')
            self.draw_on_image(adjusted, detections)
            writer.write(adjusted)
            completed = f'{(current / length) * 100}% completed'
            print(
                f'\rframe {current}/{length}\tdetections: '
                f'{len(detections)}\tcompleted: {completed}',
                end='',
            )
            if display:
                cv2.destroyAllWindows()
                cv2.imshow(f'frame {current}', adjusted)
            current += 1
            if cv2.waitKey(1) == ord('q'):
                LOGGER.info(
                    f'Video detection aborted {current}/{length} ' f'frames completed'
                )
                break
