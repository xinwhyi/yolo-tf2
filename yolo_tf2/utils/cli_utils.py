from yolo_tf2.config.cli_args import (
    TRAINING,
    DETECTION,
    EVALUATION,
)
from yolo_tf2.core.evaluator import Evaluator
from yolo_tf2.core.detector import Detector
from yolo_tf2.core.trainer import Trainer
import yolo_tf2
import random


def display_commands():
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
    for arg, options in process_args.items():
        _help = options.get('help')
        _default = options.get('default')
        _type = options.get('type')
        _action = options.get('action')
        if not _action:
            parser.add_argument(
                f'--{arg}',
                help=_help,
                default=_default,
                type=_type,
            )
        else:
            parser.add_argument(
                f'--{arg}', help=_help, default=_default, action=_action
            )
    return parser


def train(parser):
    parser = add_args(TRAINING, parser)
    cli_args = parser.parse_args()
    for item in ['input_shape', 'model_cfg', 'classes', 'image_width', 'image_height']:
        assert bool(eval(f'cli_args.{item}')), f'{item} is required'
    assert cli_args.from_xml or cli_args.relative_labels, 'No labels provided'
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
    )
    trainer.train(
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.learning_rate,
        new_dataset_conf={
            'dataset_name': (
                d_name := cli_args.dataset_name
                or f'dataset_{random.randint(10 ** 6, 10 ** 7)}'
            ),
            'relative_labels': cli_args.relative_labels,
            'from_xml': cli_args.from_xml,
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
    )
