from yolo_tf2.config.augmentation_options import AUGMENTATION_PRESETS
from yolo_tf2.core.evaluator import Evaluator
from yolo_tf2.core.detector import Detector
from yolo_tf2.core.trainer import Trainer
from yolo_tf2.config.cli_args import (
    TRAINING,
    DETECTION,
    EVALUATION,
)
import yolo_tf2


def display_commands():
    """
    Display available yolotf2 commands.
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
        if not _action:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, type=_type, required=_required
            )
        else:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, action=_action
            )
    return parser


def train(parser):
    """
    Parse cli options, create a training instance and train model.
    Args:
        parser: argparse.ArgumentParser

    Returns:
        None
    """
    parser = add_args(TRAINING, parser)
    cli_args = parser.parse_args()
    for arg in ['input_shape', 'model_cfg', 'classes', 'image_width', 'image_height']:
        assert eval(f'cli_args.{arg}'), f'{arg} is required'
    if not cli_args.train_tfrecord and not cli_args.valid_tfrecord:
        assert cli_args.dataset_name and cli_args.test_size, (
            f'--dataset-name and --test-size are required or specify '
            f'--train-tfrecord and --valid-tfrecord'
        )
        assert (
            cli_args.relative_labels or cli_args.from_xml
        ), 'No labels provided: specify --relative-labels or --from-xml'
    if cli_args.augmentation_preset:
        assert (
            preset := cli_args.augmentation_preset
        ) in AUGMENTATION_PRESETS, f'Invalid augmentation preset {preset}'
    trainer = Trainer(
        input_shape=cli_args.input_shape,
        model_configuration=cli_args.model_cfg,
        classes_file=cli_args.classes,
        image_width=cli_args.image_width,
        image_height=cli_args.image_height,
        train_tf_record=cli_args.train_tfrecord,
        valid_tf_record=cli_args.valid_tfrecord,
        max_boxes=cli_args.max_boxes,
        iou_threshold=cli_args.iou_threshold,
        score_threshold=cli_args.score_threshold,
        image_folder=cli_args.image_folder,
        xml_labels_folder=cli_args.xml_labels_folder,
    )
    trainer.train(
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.learning_rate,
        new_dataset_conf={
            'dataset_name': (d_name := cli_args.dataset_name),
            'relative_labels': cli_args.relative_labels,
            'test_size': cli_args.test_size,
            'from_xml': cli_args.from_xml,
            'augmentation': bool((preset := cli_args.augmentation_preset)),
            'sequences': AUGMENTATION_PRESETS.get(preset),
            'aug_workers': cli_args.workers,
            'aug_batch_size': cli_args.process_batch_size,
        },
        dataset_name=d_name,
        weights=cli_args.weights,
        evaluate=cli_args.evaluate,
        merge_evaluation=cli_args.merge_evaluation,
        evaluation_workers=cli_args.workers,
        shuffle_buffer=cli_args.shuffle_buffer,
        min_overlaps=cli_args.min_overlaps,
        display_stats=cli_args.display_stats,
        plot_stats=cli_args.plot_stats,
        save_figs=cli_args.save_figs,
        clear_outputs=cli_args.clear_output,
        n_epoch_eval=cli_args.n_eval,
        create_dirs=cli_args.create_output_dirs,
    )
