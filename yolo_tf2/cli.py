import argparse
import yolo_tf2


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


def execute():
    display_commands()