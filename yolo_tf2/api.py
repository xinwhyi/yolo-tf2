import random
from pathlib import Path

import cv2
import pandas as pd
from yolo_tf2.core.evaluation import calculate_map
from yolo_tf2.core.inference import YoloDetector, draw_boxes
from yolo_tf2.core.models import YoloParser
from yolo_tf2.core.training import YoloTrainer
from yolo_tf2.utils.common import parse_voc


def train(
    input_shape,
    classes,
    model_cfg,
    anchors,
    masks,
    labeled_examples=None,
    xml_dir=None,
    image_dir=None,
    batch_size=8,
    max_boxes=100,
    iou_threshold=0.5,
    score_threshold=0.5,
    train_tfrecord=None,
    valid_tfrecord=None,
    dataset_name=None,
    output_dir='.',
    valid_frac=0.1,
    shuffle_buffer_size=512,
    verbose=True,
    delete_images=False,
    train_shards=1,
    valid_shards=1,
    epochs=100,
    es_patience=3,
    weights=None,
    optimizer='adam',
):
    """
    Train yolo model on existing/yet to create dataset.
    Args:
        input_shape: Input shape passed to `tf.image.resize` and
            `keras.engine.input_layer.Input`.
        classes: Path to .txt file containing object names \n delimited.
        model_cfg: Path to .cfg DarkNet file.
        anchors: Path to .txt file containing x,y pairs \n delimited.
        masks: Path to .txt file containing x,y,z triplets \n delimited.
        labeled_examples: Path to .csv having `image`, `object_name`,
            `object_index`, `x0`, `y0`, `x1`, `y1` as columns.
        xml_dir: Path to folder containing .xml labels in VOC format.
        image_dir: Path to folder containing images referenced by xml labels.
        batch_size: Batch size passed to `tf.data.Dataset.batch`
        max_boxes: Maximum total boxes per image.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.
        score_threshold: Detection confidence score above which a detection
            is considered relevant, others are discarded.
        train_tfrecord: Path to training .tfrecord file(s).
        valid_tfrecord: Path to validation .tfrecord file(s)
        dataset_name: Prefix used in .tfrecord and model checkpoint .tf files.
        output_dir: Path to which .tfrecord and model checkpoint .tf files
            will be saved.
        valid_frac: Fraction of dataset to be included in validation .tfrecord.
        shuffle_buffer_size: `buffer_size` passed to `tf.data.Dataset.shuffle`.
        verbose: If False, training progress will not be displayed.
        delete_images: If True, after the serialization of a given example
            image and labels, the respective image will be deleted.
        train_shards: Total number of .tfrecord files to split training' 'dataset into.
        valid_shards: Total number of .tfrecord files to split validation' 'dataset into.
        epochs: Training epochs passed to `tf.keras.Model.fit`
        es_patience: Early stopping patience.
        weights: Path to pretrained model weights to load.
        optimizer: Optimizer passed to `tf.keras.Model.compile`.

    Returns:
        A History object, the result of `tf.keras.Model.fit`.
    """
    assert (
        labeled_examples
        or (xml_dir and image_dir)
        or (train_tfrecord and valid_tfrecord)
    ), (
        f'One of `labeled_examples or (`xml_dir` and `image_dir`) or '
        f'(`train_tfrecord` and `valid_tfrecord`) should be specified.'
    )
    if labeled_examples:
        labeled_examples = [*pd.read_csv(str(labeled_examples)).groupby('image')]
    elif xml_dir:
        labeled_examples = [*parse_voc(xml_dir, image_dir).groupby('image')]
    trainer = YoloTrainer(
        input_shape=input_shape,
        batch_size=batch_size,
        classes=classes,
        model_configuration=model_cfg,
        anchors=anchors,
        masks=masks,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        labeled_examples=labeled_examples,
        train_tfrecord=train_tfrecord,
        valid_tfrecord=valid_tfrecord,
        dataset_name=dataset_name,
        output_folder=output_dir,
        validation_frac=valid_frac,
        shuffle_buffer_size=shuffle_buffer_size,
        verbose=verbose,
        delete_tfrecord_images=delete_images,
        training_dataset_shards=train_shards,
        valid_dataset_shards=valid_shards,
    )
    return trainer.fit(
        epochs=epochs, es_patience=es_patience, weights=weights, optimizer=optimizer
    )


def detect(
    input_shape,
    classes,
    anchors,
    masks,
    model_cfg,
    weights,
    images=None,
    image_dir=None,
    video=None,
    max_boxes=100,
    iou_threshold=0.5,
    score_threshold=0.5,
    batch_size=8,
    verbose=True,
    output_dir='.',
    codec='mp4v',
    display_vid=False,
    evaluation_examples=None,
):
    """
    Perform detection on given images/directory of images/video, save
    results to image/video.
    Args:
        input_shape: Input shape passed to `tf.image.resize` and
            `keras.engine.input_layer.Input`.
        classes: Path to .txt file containing object names \n delimited.
        anchors: Path to .txt file containing x,y pairs \n delimited.
        masks: Path to .txt file containing x,y,z triplets \n delimited.
        model_cfg: Path to .cfg DarkNet file.
        weights: Path to pretrained model weights to load.
        images: An iterable of image paths to detect.
        image_dir: Path to directory full of images to detect.
        video: Path to video to detect.
        max_boxes: Maximum total boxes per image.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.
        score_threshold: Detection confidence score above which a detection
            is considered relevant, others are discarded.
        batch_size: Batch size passed to `tf.data.Dataset.batch`
        verbose: If False, detection progress will not be displayed.
        output_dir: Path to directory to which detection images/video will be saved.
        codec: Codec passed to `cv2.VideoWriter`.
        display_vid: If True, the given video will be rendered during detection.
        evaluation_examples: Path to .csv file with ground truth for evaluation of
            the trained model and mAP score calculation.
    Returns:
        None
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    detector = YoloDetector(
        input_shape=input_shape,
        classes=classes,
        anchors=anchors,
        masks=masks,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_boxes=max_boxes,
    )
    colors = {
        object_name: tuple([random.randint(0, 255) for _ in range(3)])
        for object_name in detector.class_map.values()
    }
    model = YoloParser(len(detector.class_map)).from_cfg(model_cfg, input_shape)
    model.load_weights(weights).expect_partial()
    target_images = []
    if images:
        target_images.extend(images)
    if image_dir:
        target_images.extend([image.as_posix() for image in Path(image_dir).iterdir()])
    if video:
        detector.detect_vid(
            src=video,
            dest=(Path(output_dir) / Path(video).name).as_posix(),
            model=model,
            colors=colors,
            codec=codec,
            display=display_vid,
            verbose=verbose,
            batch_size=batch_size,
        )
    if target_images:
        detections = detector.detect_images(
            images, model, verbose=verbose, batch_size=batch_size
        )
        detections.to_csv((Path(output_dir) / 'detections.csv').as_posix(), index=False)
        for image_path, image_detections in detections.groupby('image'):
            image = cv2.imread(image_path)
            draw_boxes(image, colors, image_detections)
            cv2.imwrite((Path(output_dir) / Path(image_path).name).as_posix(), image)
        if evaluation_examples:
            actual = pd.read_csv(evaluation_examples)
            unknown_images = set(target_images) - set(actual['image'].values)
            assert (
                not unknown_images
            ), f'Failed to find the following images in actual: {unknown_images}'
            stats = calculate_map(actual, detections, iou_threshold)
            stats.to_csv((Path(output_dir) / 'mAP scores.csv').as_posix(), index=False)
            if verbose:
                print(stats)
                print(f'mAP: {stats["average_precision"].mean()}')
