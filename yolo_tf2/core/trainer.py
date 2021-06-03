import os
import shutil

import imagesize
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)
from yolo_tf2.config.augmentation_options import AUGMENTATIONS
from yolo_tf2.core.augmentor import DataAugment
from yolo_tf2.core.evaluator import Evaluator
from yolo_tf2.core.models import BaseModel
from yolo_tf2.utils.anchors import generate_anchors, k_means
from yolo_tf2.utils.annotation_parsers import (adjust_non_voc_csv,
                                               parse_voc_folder)
from yolo_tf2.utils.common import (LOGGER, activate_gpu, calculate_loss,
                                   get_abs_path, get_image_files, timer,
                                   transform_images, transform_targets)
from yolo_tf2.utils.dataset_handlers import get_feature_map, read_tfr, save_tfr


class Trainer(BaseModel):
    """
    Create a training instance.
    """

    def __init__(
        self,
        input_shape,
        model_configuration,
        classes_file,
        train_tf_record=None,
        valid_tf_record=None,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
        image_folder=None,
    ):
        """
        Initialize trainer.
        Args:
            input_shape: tuple, (n, n, c)
            model_configuration: Path to yolo DarkNet configuration .cfg file.
            classes_file: Path to file containing dataset classes.
            train_tf_record: Path to training tfrecord.
            valid_tf_record: Path to validation tfrecord.
            anchors: numpy array of (w, h) pairs.
            masks: numpy array of masks.
            max_boxes: Maximum boxes of the tfrecords provided(if any) or
                maximum boxes setting.
            iou_threshold: float, values less than the threshold are ignored.
            score_threshold: float, values less than the threshold are ignored.
            image_folder: Folder that contains images, defaults to data/photos.
        """
        if image_folder:
            self.image_folder = get_abs_path(image_folder, verify=True)
        if not image_folder:
            self.image_folder = get_abs_path('data', 'photos', verify=True)
        assert (
            len((images := get_image_files(self.image_folder))) > 1
        ), f'Empty image folder: {self.image_folder}'
        self.image_width, self.image_height = imagesize.get(images[0])
        self.classes_file = get_abs_path(classes_file, verify=True)
        self.class_names = [item.strip() for item in open(self.classes_file)]
        super().__init__(
            input_shape,
            model_configuration,
            len(self.class_names),
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
        )
        self.train_tf_record = train_tf_record
        self.valid_tf_record = valid_tf_record
        if train_tf_record:
            self.train_tf_record = get_abs_path(train_tf_record, verify=True)
        if valid_tf_record:
            self.valid_tf_record = get_abs_path(valid_tf_record, verify=True)

    def get_adjusted_labels(self, configuration):
        """
        Adjust labels according to given configuration.
        Args:
            configuration: A dictionary containing any of the following keys:
                - relative_labels
                - xml_labels_folder
                - voc_conf (required if xml_labels_folder)
                - coordinate_labels

        Returns:
            pandas DataFrame with adjusted labels.
        """
        labels_frame = None
        check = 0
        if configuration.get('relative_labels'):
            labels_frame = adjust_non_voc_csv(
                configuration['relative_labels'],
                self.image_folder,
                self.image_width,
                self.image_height,
            )
            check += 1
        if xml_folder := configuration.get('xml_labels_folder'):
            if check:
                raise ValueError(f'Got more than one configuration')
            voc_conf = configuration.get('voc_conf')
            assert voc_conf, f'Missing VOC configuration json file.'
            labels_frame = parse_voc_folder(
                xml_folder,
                get_abs_path(voc_conf, verify=True),
            )
            labels_frame.to_csv(
                get_abs_path(
                    'output', 'data', 'parsed_from_xml.csv', create_parents=True
                ),
                index=False,
            )
            check += 1
        if coordinate_labels := configuration.get('coordinate_labels'):
            if check:
                raise ValueError(f'Got more than one configuration')
            labels_frame = pd.read_csv(get_abs_path(coordinate_labels, verify=True))
            check += 1
        return labels_frame

    def generate_new_anchors(self, new_anchors_conf):
        """
        Create new anchors according to given configuration.
        Args:
            new_anchors_conf: A dictionary containing the following keys:
                - anchor_no
                and one of the following:
                    - relative_labels
                    - from_xml
                    - coordinate_labels
        Returns:
            None
        """
        anchor_no = new_anchors_conf.get('anchor_no')
        if not anchor_no:
            raise ValueError(f'No "anchor_no" found in new_anchors_conf')
        labels_frame = self.get_adjusted_labels(new_anchors_conf)
        relative_dims = np.array(
            list(
                zip(
                    labels_frame['relative_width'],
                    labels_frame['relative_height'],
                )
            )
        )
        centroids, _ = k_means(relative_dims, anchor_no, frame=labels_frame)
        self.anchors = (
            generate_anchors(self.image_width, self.image_height, centroids)
            / self.input_shape[0]
        )
        LOGGER.info('Changed default anchors to generated ones')

    def generate_new_frame(self, new_dataset_conf):
        """
        Create new labels frame according to given configuration.
        Args:
            new_dataset_conf: A dictionary containing dataset configuration.
        Returns:
            pandas DataFrame adjusted for building the dataset containing
            labels or labels and augmented labels combined
        """
        if not new_dataset_conf.get('dataset_name'):
            raise ValueError('dataset_name not found in new_dataset_conf')
        labels_frame = self.get_adjusted_labels(new_dataset_conf)
        if new_dataset_conf.get('augmentation'):
            labels_frame = self.augment_photos(new_dataset_conf)
        return labels_frame

    def initialize_dataset(self, tf_record, batch_size, shuffle_buffer=512):
        """
        Initialize and prepare TFRecord dataset for training.
        Args:
            tf_record: TFRecord file.
            batch_size: int, training batch size
            shuffle_buffer: Buffer size for shuffling dataset.

        Returns:
            dataset.
        """
        dataset = read_tfr(
            tf_record, self.classes_file, get_feature_map(), self.max_boxes
        )
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda x, y: (
                transform_images(x, self.input_shape[0]),
                transform_targets(y, self.anchors, self.masks, self.input_shape[0]),
            )
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def augment_photos(self, new_dataset_conf):
        """
        Augment photos in self.image_paths
        Args:
            new_dataset_conf: A dictionary containing dataset configuration.

        Returns:
            pandas DataFrame with both original and augmented data.
        """
        sequences = new_dataset_conf.get('sequences')
        relative_labels = new_dataset_conf.get('relative_labels')
        coordinate_labels = new_dataset_conf.get('coordinate_labels')
        workers = new_dataset_conf.get('aug_workers')
        batch_size = new_dataset_conf.get('aug_batch_size')
        if not sequences:
            raise ValueError(f'"sequences" not found in new_dataset_conf')
        if not relative_labels:
            raise ValueError(f'No "relative_labels" found in new_dataset_conf')
        augment = DataAugment(
            relative_labels,
            AUGMENTATIONS,
            workers or 32,
            coordinate_labels,
            self.image_folder,
        )
        augment.create_sequences(sequences)
        return augment.augment_photos_folder(batch_size or 64)

    @timer(LOGGER)
    def evaluate(
        self,
        weights_file,
        merge,
        workers,
        shuffle_buffer,
        min_overlaps,
        display_stats=True,
        plot_stats=True,
        save_figs=True,
    ):
        """
        Evaluate on training and validation datasets.
        Args:
            weights_file: Path to trained .tf file.
            merge: If False, training and validation datasets will be evaluated
                separately.
            workers: Parallel predictions.
            shuffle_buffer: Buffer size for shuffling datasets.
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap
            display_stats: If True evaluation statistics will be printed.
            plot_stats: If True, evaluation statistics will be plotted including
                precision and recall curves and mAP
            save_figs: If True, resulting plots will be save to output folder.

        Returns:
            stats, map_score.
        """
        LOGGER.info('Starting evaluation ...')
        evaluator = Evaluator(
            self.input_shape,
            self.model_configuration,
            self.train_tf_record,
            self.valid_tf_record,
            self.classes_file,
            self.anchors,
            self.masks,
            self.max_boxes,
            self.iou_threshold,
            self.score_threshold,
        )
        predictions = evaluator.make_predictions(
            weights_file, merge, workers, shuffle_buffer
        )
        if isinstance(predictions, tuple):
            training_predictions, valid_predictions = predictions
            if any([training_predictions.empty, valid_predictions.empty]):
                LOGGER.info('Aborting evaluations, no detections found')
                return
            training_actual = pd.read_csv(
                get_abs_path('data', 'tfrecords', 'training_data.csv', verify=True)
            )
            valid_actual = pd.read_csv(
                get_abs_path('data', 'tfrecords', 'test_data.csv', verify=True)
            )
            training_stats, training_map = evaluator.calculate_map(
                training_predictions,
                training_actual,
                min_overlaps,
                display_stats,
                'Train',
                save_figs,
                plot_stats,
            )
            valid_stats, valid_map = evaluator.calculate_map(
                valid_predictions,
                valid_actual,
                min_overlaps,
                display_stats,
                'Valid',
                save_figs,
                plot_stats,
            )
            return training_stats, training_map, valid_stats, valid_map
        actual_data = pd.read_csv(
            get_abs_path('data', 'tfrecords', 'full_data.csv', verify=True)
        )
        if predictions.empty:
            LOGGER.info('Aborting evaluations, no detections found')
            return
        stats, map_score = evaluator.calculate_map(
            predictions,
            actual_data,
            min_overlaps,
            display_stats,
            save_figs=save_figs,
            plot_results=plot_stats,
        )
        return stats, map_score

    @staticmethod
    def clear_outputs():
        """
        Clear output folder.

        Returns:
            None
        """
        for folder_name in os.listdir(get_abs_path('output', verify=True)):
            if not folder_name.startswith('.'):
                full_path = get_abs_path('output', folder_name)
                for file_name in os.listdir(full_path):
                    full_file_path = get_abs_path(full_path, file_name)
                    if os.path.isdir(full_file_path):
                        shutil.rmtree(full_file_path)
                    else:
                        os.remove(full_file_path)
                    LOGGER.info(f'Deleted old output: {full_file_path}')

    def create_new_dataset(self, new_dataset_conf):
        """
        Create a new TFRecord dataset.
        Args:
            new_dataset_conf: A dictionary containing the following keys:
                - dataset_name(required) str representing a name for the dataset
                - test_size(required) ex: 0.1
                - augmentation(optional) True or False
                - sequences(required if augmentation is True)
                - aug_workers(optional if augmentation is True) defaults to 32.
                - aug_batch_size(optional if augmentation is True) defaults to 64.
                And one of the following is required:
                    - relative_labels: Path to csv file with the following columns:
                    ['image', 'object_name', 'object_index', 'bx', 'by', 'bw', 'bh']
                    - coordinate_labels: Path to csv file with the following columns:
                    ['image_path', 'object_name', 'img_width', 'img_height',
                    'x_min', 'y_min', 'x_max', 'y_max', 'relative_width',
                    'relative_height', 'object_id']
                    - xml_labels_folder: Path to folder containing xml labels.
        """
        LOGGER.info(f'Generating new dataset ...')
        test_size = new_dataset_conf.get('test_size')
        labels_frame = self.generate_new_frame(new_dataset_conf)
        save_tfr(
            labels_frame,
            get_abs_path('data', 'tfrecords', create_parents=True),
            new_dataset_conf['dataset_name'],
            test_size,
            self,
        )

    def check_tf_records(self):
        """
        Ensure tfrecords are specified to start training.

        Returns:
            None
        """
        if not self.train_tf_record:
            issue = 'No training TFRecord specified'
            LOGGER.error(issue)
            raise ValueError(issue)
        if not self.valid_tf_record:
            issue = 'No validation TFRecord specified'
            LOGGER.error(issue)
            raise ValueError(issue)

    @staticmethod
    def create_callbacks(checkpoint_path):
        """
        Create a list of tf.keras.callbacks.
        Args:
            checkpoint_path: Full path to checkpoint.

        Returns:
            callbacks.
        """
        return [
            ReduceLROnPlateau(verbose=1, patience=4),
            ModelCheckpoint(
                get_abs_path(checkpoint_path),
                verbose=1,
                save_weights_only=True,
            ),
            TensorBoard(log_dir=get_abs_path('data', 'tfrecords', create_parents=True)),
            EarlyStopping(monitor='val_loss', patience=6, verbose=1),
        ]

    @timer(LOGGER)
    def train(
        self,
        epochs,
        batch_size,
        learning_rate,
        new_anchors_conf=None,
        new_dataset_conf=None,
        dataset_name=None,
        weights=None,
        evaluate=True,
        merge_evaluation=True,
        evaluation_workers=8,
        shuffle_buffer=512,
        min_overlaps=None,
        display_stats=True,
        plot_stats=True,
        save_figs=True,
        clear_outputs=False,
        n_epoch_eval=None,
    ):
        """
        Train on the dataset.
        Args:
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: non-negative value.
            new_anchors_conf: A dictionary containing anchor generation configuration.
            new_dataset_conf: A dictionary containing dataset generation configuration.
            dataset_name: Name of the dataset for model checkpoints.
            weights: .tf or .weights file
            evaluate: If False, the trained model will not be evaluated after training.
            merge_evaluation: If False, training and validation maps will
                be calculated separately.
            evaluation_workers: Parallel predictions.
            shuffle_buffer: Buffer size for shuffling datasets.
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap
            display_stats: If True and evaluate=True, evaluation statistics will
                be displayed.
            plot_stats: If True, Precision and recall curves as well as
                comparative bar charts will be plotted
            save_figs: If True and plot_stats=True, figures will be saved
            clear_outputs: If True, old outputs will be cleared
            n_epoch_eval: Conduct evaluation every n epoch.

        Returns:
            history object, pandas DataFrame with statistics, mAP score.
        """
        min_overlaps = min_overlaps or 0.5
        if clear_outputs:
            self.clear_outputs()
        activate_gpu()
        LOGGER.info(f'Starting training ...')
        if new_anchors_conf:
            LOGGER.info(f'Generating new anchors ...')
            self.generate_new_anchors(new_anchors_conf)
        self.create_models(reverse_v4=True)
        if weights:
            self.load_weights(weights)
        if new_dataset_conf:
            self.create_new_dataset(new_dataset_conf)
        self.check_tf_records()
        training_dataset = self.initialize_dataset(
            self.train_tf_record, batch_size, shuffle_buffer
        )
        valid_dataset = self.initialize_dataset(
            self.valid_tf_record, batch_size, shuffle_buffer
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = [
            calculate_loss(self.anchors[mask], self.classes, self.iou_threshold)
            for mask in self.masks
        ]
        self.training_model.compile(optimizer=optimizer, loss=loss)
        checkpoint_path = get_abs_path(
            'models', f'{dataset_name or "trained"}_model.tf'
        )
        callbacks = self.create_callbacks(checkpoint_path)
        if n_epoch_eval:
            mid_train_eval = MidTrainingEvaluator(
                self.input_shape,
                self.model_configuration,
                self.classes_file,
                self.train_tf_record,
                self.valid_tf_record,
                self.anchors,
                self.masks,
                self.max_boxes,
                self.iou_threshold,
                self.score_threshold,
                n_epoch_eval,
                merge_evaluation,
                evaluation_workers,
                shuffle_buffer,
                min_overlaps,
                display_stats,
                plot_stats,
                save_figs,
                checkpoint_path,
                self.image_folder,
            )
            callbacks.append(mid_train_eval)
        history = self.training_model.fit(
            training_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=valid_dataset,
        )
        LOGGER.info('Training complete')
        if evaluate:
            evaluations = self.evaluate(
                checkpoint_path,
                merge_evaluation,
                evaluation_workers,
                shuffle_buffer,
                min_overlaps,
                display_stats,
                plot_stats,
                save_figs,
            )
            return evaluations, history
        return history


class MidTrainingEvaluator(Callback, Trainer):
    """
    Tool to evaluate trained model on the go(during the training, every n epochs).
    """

    def __init__(
        self,
        input_shape,
        model_configuration,
        classes_file,
        train_tf_record,
        valid_tf_record,
        anchors,
        masks,
        max_boxes,
        iou_threshold,
        score_threshold,
        n_epochs,
        merge,
        workers,
        shuffle_buffer,
        min_overlaps,
        display_stats,
        plot_stats,
        save_figs,
        weights_file,
        image_folder,
    ):
        """
        Initialize mid-training evaluation settings.
        Args:
            input_shape: tuple, (n, n, c)
            model_configuration: Path to DarkNet cfg file.
            classes_file: File containing class names \n delimited.
            train_tf_record: TFRecord file.
            valid_tf_record: TFRecord file.
            anchors: numpy array of (w, h) pairs.
            masks: numpy array of masks.
            max_boxes: Maximum boxes of the tfrecords provided(if any) or
                maximum boxes setting.
            iou_threshold: float, values less than the threshold are ignored.
            score_threshold: float, values less than the threshold are ignored.
            n_epochs: int, perform evaluation every n epochs
            merge: If True, The whole dataset(train + valid) will be evaluated
            workers: Parallel predictions
            shuffle_buffer: Buffer size for shuffling datasets
            min_overlaps: a float value between 0 and 1, or a dictionary
                containing each class in self.class_names mapped to its
                minimum overlap
            display_stats: If True, statistics will be displayed at the end.
            plot_stats: If True, precision and recall curves as well as
                comparison bar charts will be plotted.
            save_figs: If True and display_stats, plots will be save to output folder
            weights_file: .tf file(most recent checkpoint)
            image_folder: Path to folder containing training images.
        """
        Trainer.__init__(
            self,
            input_shape,
            model_configuration,
            classes_file,
            train_tf_record,
            valid_tf_record,
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
            image_folder,
        )
        self.n_epochs = n_epochs
        self.evaluation_args = [
            weights_file,
            merge,
            workers,
            shuffle_buffer,
            min_overlaps,
            display_stats,
            plot_stats,
            save_figs,
        ]

    def on_epoch_end(self, epoch, logs=None):
        """
        Start evaluation in valid epochs.
        Args:
            epoch: int, epoch number.
            logs: dict, TensorBoard log.

        Returns:
            None
        """
        if not (epoch + 1) % self.n_epochs == 0:
            return
        self.evaluate(*self.evaluation_args)
        evaluation_dir = get_abs_path(
            'output', 'evaluation', f'epoch-{epoch}-evaluation', create=True
        )
        current_predictions = [
            get_abs_path('output', 'data', item)
            for item in os.listdir(get_abs_path('output', 'data', verify=True))
        ]
        current_figures = [
            get_abs_path('output', 'plots', item)
            for item in os.listdir(get_abs_path('output', 'plots'))
        ]
        current_files = current_predictions + current_figures
        for file_path in current_files:
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                new_path = get_abs_path(evaluation_dir, file_name)
                shutil.move(file_path, new_path)
