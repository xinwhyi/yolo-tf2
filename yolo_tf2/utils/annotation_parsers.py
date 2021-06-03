import json
import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from yolo_tf2.utils.common import LOGGER, get_abs_path, ratios_to_coordinates
from yolo_tf2.utils.visual_tools import visualization_wrapper


def get_tree_item(parent, tag, file_path, find_all=False):
    """
    Get item from xml tree element.
    Args:
        parent: Parent in xml element tree
        tag: tag to look for.
        file_path: Current xml file being handled.
        find_all: If True, all elements found will be returned.

    Returns:
        Tag item.
    """
    target = parent.find(tag)
    if find_all:
        target = parent.findall(tag)
    if target is None:
        raise ValueError(f'Could not find {tag} in {file_path}')
    return target


def parse_voc_file(file_path, voc_conf):
    """
    Parse voc annotation from xml file.
    Args:
        file_path: Path to xml file.
        voc_conf: voc configuration file.

    Returns:
        A list of image annotations.
    """
    file_path = get_abs_path(file_path, verify=True)
    voc_conf = get_abs_path(voc_conf, verify=True)
    image_data = []
    with open(voc_conf) as json_data:
        tags = json.load(json_data)
    tree = ElementTree.parse(file_path)
    image_path = get_tree_item(tree, tags['tree']['path'], file_path).text
    size_item = get_tree_item(tree, tags['size']['size_tag'], file_path)
    image_width = get_tree_item(size_item, tags['size']['width'], file_path).text
    image_height = get_tree_item(size_item, tags['size']['height'], file_path).text
    for item in get_tree_item(tree, tags['object']['object_tag'], file_path, True):
        name = get_tree_item(item, tags['object']['object_name'], file_path).text
        box_item = get_tree_item(
            item, tags['object']['object_box']['object_box_tag'], file_path
        )
        x0 = get_tree_item(box_item, tags['object']['object_box']['x0'], file_path).text
        y0 = get_tree_item(box_item, tags['object']['object_box']['y0'], file_path).text
        x1 = get_tree_item(box_item, tags['object']['object_box']['x1'], file_path).text
        y1 = get_tree_item(box_item, tags['object']['object_box']['y1'], file_path).text
        image_data.append([image_path, name, image_width, image_height, x0, y0, x1, y1])
    return image_data


def adjust_frame(frame, cache_file=None):
    """
    Add relative width, relative height and object ids to annotation pandas DataFrame.
    Args:
        frame: pandas DataFrame containing coordinates instead of relative labels.
        cache_file: cache_file: csv file name containing current session labels.

    Returns:
        Frame with the new columns
    """
    object_id = 1
    for item in frame.columns[2:]:
        frame[item] = frame[item].astype(float).astype(int)
    frame['relative_width'] = (frame['x_max'] - frame['x_min']) / frame['img_width']
    frame['relative_height'] = (frame['y_max'] - frame['y_min']) / frame['img_height']
    for object_name in list(frame['object_name'].drop_duplicates()):
        frame.loc[frame['object_name'] == object_name, 'object_id'] = object_id
        object_id += 1
    if cache_file:
        frame.to_csv(
            get_abs_path('output', 'data', cache_file, create_parents=True), index=False
        )
    LOGGER.info(f'Parsed labels:\n{frame["object_name"].value_counts()}')
    return frame


@visualization_wrapper
def parse_voc_folder(folder_path, voc_conf):
    """
    Parse a folder containing voc xml annotation files.
    Args:
        folder_path: Folder containing voc xml annotation files.
        voc_conf: Path to voc json configuration file.

    Returns:
        pandas DataFrame with the annotations.
    """
    folder_path = get_abs_path(folder_path, verify=True)
    cache_path = get_abs_path('output', 'data', 'parsed_from_xml.csv')
    if os.path.exists(cache_path):
        frame = pd.read_csv(cache_path)
        LOGGER.info(
            f'Labels retrieved from cache:' f'\n{frame["object_name"].value_counts()}'
        )
        return frame
    image_data = []
    frame_columns = [
        'image_path',
        'object_name',
        'img_width',
        'img_height',
        'x_min',
        'y_min',
        'x_max',
        'y_max',
    ]
    xml_files = [
        get_abs_path(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith('.xml')
    ]
    for file_name in xml_files:
        annotation_path = get_abs_path(folder_path, file_name)
        image_labels = parse_voc_file(annotation_path, voc_conf)
        image_data.extend(image_labels)
    frame = pd.DataFrame(image_data, columns=frame_columns)
    classes = frame['object_name'].drop_duplicates()
    LOGGER.info(f'Read {len(xml_files)} xml files')
    LOGGER.info(f'Received {len(frame)} labels containing ' f'{len(classes)} classes')
    if frame.empty:
        raise ValueError(f'No labels were found in {folder_path}')
    frame = adjust_frame(frame, 'parsed_from_xml.csv')
    return frame


@visualization_wrapper
def adjust_non_voc_csv(csv_file, image_path, image_width, image_height):
    """
    Read relative data and return adjusted frame accordingly.
    Args:
        csv_file: .csv file containing the following columns:
        [image, object_name, object_index, bx, by, bw, bh]
        image_path: Path prefix to be added.
        image_width: image width.
        image_height: image height
    Returns:
        pandas DataFrame with the following columns:
        ['image_path', 'object_name', 'img_width', 'img_height', 'x_min',
       'y_min', 'x_max', 'y_max', 'relative_width', 'relative_height',
       'object_id']
    """
    image_path = get_abs_path(image_path, verify=True)
    coordinates = []
    old_frame = pd.read_csv(get_abs_path(csv_file, verify=True))
    new_frame = pd.DataFrame()
    new_frame['image_path'] = old_frame['image'].apply(
        lambda item: get_abs_path(image_path, item)
    )
    new_frame['object_name'] = old_frame['object_name']
    new_frame['img_width'] = image_width
    new_frame['img_height'] = image_height
    new_frame['relative_width'] = old_frame['bw']
    new_frame['relative_height'] = old_frame['bh']
    new_frame['object_id'] = old_frame['object_index'] + 1
    for index, row in old_frame.iterrows():
        image, object_name, object_index, bx, by, bw, bh = row
        co = ratios_to_coordinates(bx, by, bw, bh, image_width, image_height)
        coordinates.append(co)
    (
        new_frame['x_min'],
        new_frame['y_min'],
        new_frame['x_max'],
        new_frame['y_max'],
    ) = np.array(coordinates).T
    new_frame[['x_min', 'y_min', 'x_max', 'y_max']] = new_frame[
        ['x_min', 'y_min', 'x_max', 'y_max']
    ].astype('int64')
    print(f'Parsed labels:\n{new_frame["object_name"].value_counts()}')
    classes = new_frame['object_name'].drop_duplicates()
    LOGGER.info(
        f'Adjustment from existing received {len(new_frame)} labels containing '
        f'{len(classes)} classes'
    )
    LOGGER.info(f'Added prefix to images: {image_path}')
    return new_frame[
        [
            'image_path',
            'object_name',
            'img_width',
            'img_height',
            'x_min',
            'y_min',
            'x_max',
            'y_max',
            'relative_width',
            'relative_height',
            'object_id',
        ]
    ]
