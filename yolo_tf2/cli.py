import argparse
import sys

import pandas as pd
import tensorflow as tf
import yolo_tf2
from yolo_tf2.api import detect, train
from yolo_tf2.config.cli import DETECTION, EVALUATION, GENERAL, TRAINING


def display_section(name):
    """
    Display a dictionary of command line options
    Args:
        name: One of ['GENERAL', 'TRAINING', 'EVALUATION', 'DETECTION']

    Returns:
        None
    """
    assert all((GENERAL, TRAINING, DETECTION, EVALUATION))
    section_frame = pd.DataFrame(eval(name)).T.fillna('-')
    section_frame['flags'] = section_frame.index.values
    section_frame['flags'] = section_frame['flags'].apply(lambda c: f'--{c}')
    section_frame = section_frame.reset_index(drop=True).set_index('flags')
    print(f'\n{name.title()}\n')
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
        'evaluate': 'Evaluate a trained model',
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
        for name in ('GENERAL', 'TRAINING', 'EVALUATION', 'DETECTION'):
            display_section(name)


def add_args(process_args, parser):
    """
    Add given arguments to parser.
    Args:
        process_args: A dictionary of args and options.
        parser: argparse.ArgumentParser

    Returns:
        parser.
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
    return parser


def add_all_args(parser, process_args, *args):
    """
    Add general and process specific args.
    Args:
        parser: argparse.ArgumentParser
        process_args: One of [GENERAL, TRAINING, EVALUATION, DETECTION]
        *args: Process required args

    Returns:
        cli_args
    """
    parser = add_args(process_args, parser)
    cli_args = parser.parse_args()
    for arg in ['model_cfg', 'classes', *args]:
        assert eval(f'cli_args.{arg}'), f'{arg} is required'
    return cli_args


def train_from_parser(parser):
    """
    Parse cli options, create a training instance and train model.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    cli_args = add_all_args(parser, TRAINING)
    optimizer = tf.keras.optimizers.Adam(cli_args.learning_rate)
    train(
        input_shape=cli_args.input_shape,
        classes=cli_args.classes,
        model_cfg=cli_args.model_cfg,
        anchors=cli_args.anchors,
        masks=cli_args.masks,
        labeled_examples=cli_args.labeled_examples,
        batch_size=cli_args.batch_size,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
        train_tfrecord=cli_args.train_tfrecord,
        valid_tfrecord=cli_args.valid_tfrecord,
        dataset_name=cli_args.dataset_name,
        output_dir=cli_args.output_dir,
        valid_frac=cli_args.valid_frac,
        shuffle_buffer_size=cli_args.shuffle_buffer_size,
        verbose=not cli_args.quiet,
        delete_images=cli_args.delete_images,
        train_shards=cli_args.train_shards,
        valid_shards=cli_args.valid_shards,
        epochs=cli_args.epochs,
        es_patience=cli_args.es_patience,
        weights=cli_args.weights,
        optimizer=optimizer,
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
    cli_args = add_all_args(parser, DETECTION)
    detect(
        classes=cli_args.classes,
        anchors=cli_args.anchors,
        input_shape=cli_args.input_shape,
        masks=cli_args.masks,
        model_cfg=cli_args.model_cfg,
        weights=cli_args.weights,
        images=cli_args.images,
        image_dir=cli_args.image_dir,
        video=cli_args.video,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
        batch_size=cli_args.batch_size,
        verbose=not cli_args.quiet,
        output_dir=cli_args.output_dir,
        codec=cli_args.codec,
        display_vid=cli_args.display_vid,
    )


def evaluate(parser):
    """
    Parse cli options, create an evaluation instance and evaluate.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    cli_args = add_all_args(parser, EVALUATION)


def execute():
    """
    Train or evaluate or detect based on cli args.
    Returns:
        None
    """
    valid_commands = {
        'train': ('TRAINING', TRAINING, train_from_parser),
        'evaluate': ('EVALUATION', EVALUATION, evaluate),
        'detect': ('DETECTION', DETECTION, detect_from_parser),
    }
    if (total := len(cli_args := sys.argv)) == 1:
        display_commands()
        return
    if (command := cli_args[1]) in valid_commands and total == 2:
        display_section(valid_commands[command][0])
        return
    if (help_flags := any(('-h' in cli_args, '--help' in cli_args))) and total == 2:
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
    parser = add_args(GENERAL, parser)
    valid_commands[command][2](parser)


if __name__ == '__main__':
    execute()
