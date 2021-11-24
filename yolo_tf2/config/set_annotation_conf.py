import json
import os

from yolo_tf2.utils.common import get_abs_path


def set_voc_tags(
    tree='annotation',
    folder='folder',
    filename='filename',
    path='path',
    size='size',
    width='width',
    height='height',
    depth='depth',
    obj='object',
    obj_name='name',
    box='bndbox',
    x0='xmin',
    y0='ymin',
    x1='xmax',
    y1='ymax',
    conf_file='voc_conf.json',
    indent=4,
    sort_keys=False,
):
    """
    Create/modify json voc annotation tags.
    Args:
        tree: xml tree tag.
        folder: Image folder tag.
        filename: Image file tag.
        path: Path to image tag.
        size: Image size tag.
        width: Image width tag.
        height: Image height tag.
        depth: Image depth tag.
        obj: Object tag.
        obj_name: Object name tag.
        box: Bounding box tag.
        x0: Start x coordinate tag.
        y0: Start y coordinate tag.
        x1: End x coordinate tag.
        y1: End y coordinate tag.
        conf_file: Configuration file name.
        indent: json output indent.
        sort_keys: Sort json output keys.

    Returns:
        None.
    """
    if (conf_file := get_abs_path(conf_file, verify=True)) in os.listdir():
        os.remove(conf_file)
    conf = {
        'Tree': {
            'Tree Tag': tree,
            'Folder': folder,
            'Filename': filename,
            'Path': path,
        },
        'Size': {
            'Size Tag': size,
            'Width': width,
            'Height': height,
            'Depth': depth,
        },
        'Object': {
            'Object Tag': obj,
            'object_name': obj_name,
            'Object Box': {
                'Object Box Tag': box,
                'X0': x0,
                'Y0': y0,
                'X1': x1,
                'Y1': y1,
            },
        },
    }

    with open(conf_file, 'w') as conf_out:
        json.dump(conf, conf_out, indent=indent, sort_keys=sort_keys)


if __name__ == '__main__':
    set_voc_tags()
