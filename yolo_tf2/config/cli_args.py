import ast


GENERAL = {
    'input-shape': {
        'help': 'Input shape ex: (416, 416, 3)',
        'default': (416, 416, 3),
        'type': ast.literal_eval,
    },
    'classes': {'help': 'Path to classes .txt file'},
    'weights': {'help': 'Path to .tf weights file', 'default': None},
    'model-cfg': {'help': 'Yolo DarkNet configuration .cfg file'},
    'max-boxes': {'help': 'Maximum boxes per image', 'default': 100, 'type': int},
    'train-tfrecord': {'help': 'Path to training .tfrecord file', 'default': None},
    'valid-tfrecord': {'help': 'Path to validation .tfrecord file', 'default': None},
    'iou-threshold': {
        'help': 'IOU(intersection over union threshold)',
        'default': 0.5,
        'type': float,
    },
    'score-threshold': {
        'help': 'Confidence score threshold',
        'default': 0.5,
        'type': float,
    },
    'workers': {
        'help': 'Concurrent tasks(in areas where that is possible)',
        'default': 4,
        'type': int,
    },
    'process-batch-size': {
        'help': 'Batch size of operations that needs batching (excluding training)',
        'default': 32,
        'type': int,
    },
}

TRAINING = {
    'image-width': {'help': 'Image actual width', 'type': int},
    'image-height': {'help': 'Image actual height', 'type': int},
    'epochs': {'help': 'Number of training epochs', 'type': int, 'default': 100},
    'batch-size': {'help': 'Training batch size', 'type': int, 'default': 8},
    'learning-rate': {'help': 'Training learning rate', 'type': float, 'default': 1e-3},
    'dataset-conf': {'help': 'New dataset configuration', 'default': None},
    'dataset-name': {'help': 'Name of the checkpoint', 'default': ''},
    'evaluate': {
        'help': 'If True, evaluation will be conducted after training',
        'action': 'store_true',
    },
    'merge-evaluation': {
        'help': 'If False, evaluate training and validation separately',
        'action': 'store_true',
    },
    'shuffle-buffer': {'help': 'Dataset shuffle buffer', 'default': 512, 'type': int},
    'min-overlaps': {
        'help': 'a float value between 0 and 1',
        'default': 0.5,
        'type': float,
    },
    'display-stats': {
        'help': 'If True, display evaluation statistics',
        'action': 'store_true',
    },
    'plot-stats': {'help': 'If True, plot results', 'action': 'store_true'},
    'save-figs': {'help': 'If True, save plots', 'action': 'store_true'},
    'clear-output': {'help': 'If True, clear output folders', 'action': 'store_true'},
    'n-eval': {'help': 'Evaluate every n epochs', 'default': None, 'type': int},
    'relative-labels': {'help': 'Path to .csv file that contains', 'default': None},
    'from-xml': {
        'help': 'Parse labels from XML files in data > xml_labels',
        'action': 'store_true',
    },
}

DETECTION = {
    'target-dir': {
        'help': 'A directory that contains images to predict',
        'default': None,
    },
    'video': {'help': 'A video to predict', 'default': None},
    'codec': {
        'help': 'Codec to use for predicting videos',
        'default': 'mp4v',
    },
    'display': {'help': 'Display video while predicting', 'action': 'store_true'},
}

EVALUATION = {
    'predicted-data': {
        'help': 'csv file with predictions',
    },
    'actual-data': {'help': 'csv file with actual data'},
}
