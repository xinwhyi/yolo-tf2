general_args = {
    'input-shape': {
        'help': 'Input shape ex: (m, m, c)',
        'default': (416, 416, 3),
        'nargs': '+',
        'type': int,
    },
    'batch-size': {'help': 'Training/detection batch size', 'type': int, 'default': 8},
    'classes': {'help': 'Path to classes .txt file', 'required': True},
    'anchors': {'help': 'Path to anchors .txt file', 'required': True},
    'masks': {'help': 'Path to masks .txt file', 'required': True},
    'model-cfg': {'help': 'Yolo DarkNet configuration .cfg file', 'required': True},
    'max-boxes': {'help': 'Maximum boxes per image', 'default': 100, 'type': int},
    'iou-threshold': {
        'help': 'iou (intersection over union) threshold',
        'default': 0.5,
        'type': float,
    },
    'score-threshold': {
        'help': 'Confidence score threshold',
        'default': 0.5,
        'type': float,
    },
    'quiet': {
        'help': 'If specified, verbosity is set to False',
        'action': 'store_true',
    },
    'v4': {
        'help': 'If yolov4 configuration is used, this should be specified',
        'action': 'store_true',
    },
}

training_args = {
    'weights': {'help': 'Path to trained weights .tf or .weights file'},
    'epochs': {'help': 'Number of training epochs', 'type': int, 'default': 100},
    'learning-rate': {'help': 'Training learning rate', 'type': float, 'default': 1e-3},
    'dataset-name': {'help': 'Checkpoint/dataset prefix'},
    'valid-frac': {
        'help': 'Validation dataset fraction',
        'type': float,
        'default': 0.1,
    },
    'shuffle-buffer-size': {
        'help': 'Dataset shuffle buffer',
        'default': 512,
        'type': int,
    },
    'labeled-examples': {'help': 'Path to labels .csv file'},
    'xml-dir': {'help': 'Path to folder containing .xml labels in VOC format'},
    'image-dir': {'help': 'Path to folder containing images referenced by xml labels'},
    'train-tfrecord': {'help': 'Path to training .tfrecord file'},
    'valid-tfrecord': {'help': 'Path to validation .tfrecord file'},
    'output-dir': {
        'help': 'Path to folder where training dataset / checkpoints '
        '/ other data will be saved',
        'default': '.',
    },
    'delete-images': {
        'help': 'If specified, dataset images will be deleted upon '
        'being saved to tfrecord.',
        'action': 'store_true',
    },
    'train-shards': {
        'help': 'Total number of .tfrecord files to split training dataset into',
        'default': 1,
        'type': int,
    },
    'valid-shards': {
        'help': 'Total number of .tfrecord files to split validation dataset into',
        'default': 1,
        'type': int,
    },
    'es-patience': {'help': 'Early stopping patience', 'type': int},
}


detection_args = {
    'images': {'help': 'Paths of images to detect', 'nargs': '+'},
    'image-dir': {
        'help': 'A directory that contains images to predict',
    },
    'video': {'help': 'Path to video to predict'},
    'codec': {
        'help': 'Codec to use for predicting videos',
        'default': 'mp4v',
    },
    'display-vid': {'help': 'Display video during detection', 'action': 'store_true'},
    'weights': {
        'help': 'Path to trained weights .tf or .weights file',
        'required': True,
    },
    'output-dir': {'help': 'Path to directory for saving results'},
    'evaluation-examples': {
        'help': 'Path to .csv file with ground truth for evaluation of '
        'the trained model and mAP score calculation.'
    },
}
