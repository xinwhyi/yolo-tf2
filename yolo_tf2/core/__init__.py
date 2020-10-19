from pathlib import Path
import os


dirs = [
    Path('data') / sub_folder for sub_folder in ('photos', 'tfrecords', 'xml_labels')
]
dirs.extend([Path('models'), Path('yolo_logs'), Path('kos')])
dirs.extend(
    [
        Path('output') / sub_folder
        for sub_folder in ('data', 'detections', 'evaluation', 'plots')
    ]
)
for dir_name in dirs:
    os.makedirs(dir_name.as_posix(), exist_ok=True)
