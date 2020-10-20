from yolo_tf2.config.cli_args import GENERAL
from yolo_tf2.utils.cli_utils import (
    display_section,
    display_commands,
    add_args,
    train,
    evaluate,
    detect
)
import argparse
import sys


def execute():
    """
    Train or evaluate or detect based on cli args.
    Returns:
        None
    """
    valid_commands = {
        'train': ('TRAINING', train),
        'evaluate': ('EVALUATION', evaluate),
        'detect': ('DETECTION', detect)
    }
    if (total := len(cli_args := sys.argv)) == 1:
        display_commands()
        return
    if (command := cli_args[1]) in valid_commands and total == 2:
        display_section(valid_commands[command][0])
        return
    if (help_flags := any(('-h' in cli_args, '--help' in cli_args))) and total == 2:
        display_commands()
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
    valid_commands[command][1](parser)


if __name__ == '__main__':
    execute()
