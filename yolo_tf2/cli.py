import argparse
import sys

import pandas as pd
import tensorflow as tf
import yolo_tf2
from yolo_tf2.api import detect, train
from yolo_tf2.config.cli import detection_args, general_args, training_args


def display_section(section):
    """
    Display a dictionary of command line options
    Args:
        section: A dictionary having cli options.

    Returns:
        None
    """
    section_frame = pd.DataFrame(section).T.fillna('-')
    section_frame['flags'] = section_frame.index.values
    section_frame = section_frame.sort_values(by='flags')
    section_frame['flags'] = section_frame['flags'].apply(lambda c: f'--{c}')
    section_frame = section_frame.reset_index(drop=True).set_index('flags')
    print(
        section_frame[
            [
                column_name
                for column_name in ('help', 'required', 'default')
                if column_name in section_frame.columns
            ]
        ].to_markdown()
    )


def display_commands(display_all=False):
    """
    Display available yolotf2 commands.
    Args:
        display_all: If True, all commands will be displayed
    Returns:
        None
    """
    available_commands = {
        'train': 'Create new or use existing dataset and train a model',
        'detect': 'Detect a folder of images or a video',
    }
    print(f'Yolo-tf2 {yolo_tf2.__version__}')
    print(f'\nUsage:')
    print(f'\tyolotf2 <command> [options] [args]')
    print(f'\nAvailable commands:')
    for command, description in available_commands.items():
        print(f'\t{command:<10} {description}')
    print()
    print('Use yolotf2 <command> -h to see more info about a command', end='\n\n')
    print('Use yolotf2 -h to display all command line options')
    if display_all:
        for section in (general_args, training_args, detection_args):
            display_section(section)


def add_args(process_args, parser):
    """
    Add given arguments to parser.
    Args:
        process_args: A dictionary of args and options.
        parser: argparse.ArgumentParser
    """
    for arg, options in process_args.items():
        _help = options.get('help')
        _default = options.get('default')
        _type = options.get('type')
        _action = options.get('action')
        _required = options.get('required')
        _nargs = options.get('nargs')
        if not _action:
            parser.add_argument(
                f'--{arg}',
                help=_help,
                default=_default,
                type=_type,
                required=_required,
                nargs=_nargs,
            )
        else:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, action=_action
            )


def train_from_parser(parser):
    """
    Parse cli options, create a training instance and train model.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    add_args(training_args, parser)
    args = parser.parse_args()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    train(
        input_shape=args.input_shape,
        classes=args.classes,
        model_cfg=args.model_cfg,
        anchors=args.anchors,
        masks=args.masks,
        labeled_examples=args.labeled_examples,
        xml_dir=args.xml_dir,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        max_boxes=args.max_boxes,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        train_tfrecord=args.train_tfrecord,
        valid_tfrecord=args.valid_tfrecord,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        valid_frac=args.valid_frac,
        shuffle_buffer_size=args.shuffle_buffer_size,
        verbose=not args.quiet,
        delete_images=args.delete_images,
        train_shards=args.train_shards,
        valid_shards=args.valid_shards,
        epochs=args.epochs,
        es_patience=args.es_patience,
        weights=args.weights,
        optimizer=optimizer,
        v4=args.v4,
    )


def detect_from_parser(parser):
    """
    Detect, draw boxes over an image / a folder of images / a video
    and save results.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    add_args(detection_args, parser)
    args = parser.parse_args()
    detect(
        classes=args.classes,
        anchors=args.anchors,
        input_shape=args.input_shape,
        masks=args.masks,
        model_cfg=args.model_cfg,
        weights=args.weights,
        images=args.images,
        image_dir=args.image_dir,
        video=args.video,
        max_boxes=args.max_boxes,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        codec=args.codec,
        display_vid=args.display_vid,
        evaluation_examples=args.evaluation_examples,
        v4=args.v4,
    )


def execute():
    """
    Execute training/detection/evaluation.
    Returns:
        None
    """
    valid_commands = {
        'train': (training_args, train_from_parser),
        'detect': (detection_args, detect_from_parser),
    }
    if (total := len(args := sys.argv)) == 1:
        display_commands()
        return
    if (command := args[1]) in valid_commands and total == 2:
        display_section(valid_commands[command][0])
        return
    if (help_flags := any(('-h' in args, '--help' in args))) and total == 2:
        display_commands(True)
        return
    if total == 3 and command in valid_commands and help_flags:
        display_section(valid_commands[command][0])
        return
    if command not in valid_commands:
        print(f'Invalid command {command}')
        return
    parser = argparse.ArgumentParser()
    del sys.argv[1]
    add_args(general_args, parser)
    valid_commands[command][1](parser)


if __name__ == '__main__':
    execute()
