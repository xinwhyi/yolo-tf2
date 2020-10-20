from yolo_tf2.utils.cli_utils import display_commands, add_args, train
from yolo_tf2.config.cli_args import GENERAL
import argparse
import sys


def execute():
    """
    Train or evaluate or detect based on cli args.
    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    if len(sys.argv) == 1:
        display_commands()
        return
    assert (command := sys.argv[1]) in (
        'train',
        'evaluate',
        'detect',
    ), f'Invalid command {command}'
    del sys.argv[1]
    parser = add_args(GENERAL, parser)
    if command == 'train':
        train(parser)
    if command == 'evaluate':
        pass
    if command == 'detect':
        pass


if __name__ == '__main__':
    execute()
