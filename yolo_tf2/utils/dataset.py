import os

import tensorflow as tf
from ml_utils.python.generic import split_filename


def transform_label(y, grid_size, anchor_indices):
    n = tf.shape(y)[0]
    y_true_out = tf.zeros((n, grid_size, grid_size, tf.shape(anchor_indices)[0], 6))
    anchor_indices = tf.cast(anchor_indices, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(n):
        for j in tf.range(tf.shape(y)[1]):
            if tf.equal(y[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_indices, tf.cast(y[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                box = y[i][j][0:4]
                box_xy = (y[i][j][0:2] + y[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y[i][j][4]]
                )
                idx += 1
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_labels(y, anchors, masks, size):
    """
    Transform label to a training-friendly form.
    Args:
        y: Example label.
        anchors: Anchors as numpy array.
        masks: Masks as numpy array.
        size: Image width or height.

    Returns:
        Transformed label
    """
    y_outs = []
    grid_size = size // 32
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y[..., 2:4] - y[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1]
    )
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)
    y = tf.concat([y, anchor_idx], axis=-1)
    for anchor_indices in masks:
        y_outs.append(transform_label(y, grid_size, anchor_indices))
        grid_size *= 2
    return tuple(y_outs)


def serialize_example(image_path, labels, writer):
    """
    Serialize training example to tfrecord.
    Args:
        image_path: Path to example image.
        labels: `pd.DataFrame` having `image`, `object_name`, `object_index`,
            `x0`, `y0`, `x1`, `y1` as columns containing object coordinates in
             the example's image.
        writer: `tf.io.TFRecordWriter`.

    Returns:
        None
    """
    features = {}
    with open(image_path, 'rb') as image:
        features['image'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.read()])
        )
    features['x0'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=labels['x0'].values)
    )
    features['y0'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=labels['y0'].values)
    )
    features['x1'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=labels['x1'].values)
    )
    features['y1'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=labels['y1'].values)
    )
    features['object_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=labels['object_name'].values.astype("|S"))
    )
    features['object_index'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=labels['object_index'].values)
    )
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())


def create_tfrecord(
    output_path, grouped_labels, shards, delete_images=False, verbose=True
):
    """
    Create tfrecord file(s) given labeled examples.
    Args:
        output_path: Path to output .tfrecord file which will have a digit
            suffix according to `shards` ex: for shards = 2, and output
            path = example.tfrecord, there will be example-0.tfrecord and
            example-1.tfrecord.
        grouped_labels: A list of `pd.DataFrame.groupby` results.
        shards: Total .tfrecord output files.
        delete_images: If True, after the serialization of a given example
            image and labels, the respective image will be deleted.
        verbose: If False, progress will not be displayed.

    Returns:
        None
    """
    filenames = split_filename(output_path, shards)
    total_labels = len(grouped_labels)
    step = total_labels // shards
    total_examples = len(grouped_labels)
    for (filename, idx) in zip(filenames, range(0, total_labels, step)):
        chunk = grouped_labels[idx : idx + step]
        with tf.io.TFRecordWriter(filename) as writer:
            for i, (image_path, labels) in enumerate(chunk, idx):
                if verbose:
                    print(f'\rWriting example: {i + 1}/{total_examples}', end='')
                serialize_example(image_path, labels, writer)
                if delete_images:
                    os.remove(image_path)
        if verbose:
            print()


def read_example(
    example,
    feature_map,
    class_table,
    max_boxes,
    image_shape,
):
    """
    Read single example from .tfrecord.
    Args:
        example: `tf.Tensor` having a single serialized example.
        feature_map: A dictionary mapping feature names to `tf.io` types.
        class_table: `tf.lookup.StaticHashTable`
        max_boxes: Maximum total boxes per image.
        image_shape: input_shape: Input shape passed to `tf.image.resize`

    Returns:
        image, label
    """
    features = tf.io.parse_single_example(example, feature_map)
    image = tf.image.decode_png(features['image'], channels=3)
    image = tf.image.resize(image, image_shape)
    object_name = tf.sparse.to_dense(features['object_name'])
    label = tf.cast(class_table.lookup(object_name), tf.float32)
    label = tf.stack(
        [tf.sparse.to_dense(features[feature]) for feature in ['x0', 'y0', 'x1', 'y1']]
        + [label],
        1,
    )
    padding = [[0, max_boxes - tf.shape(label)[0]], [0, 0]]
    label = tf.pad(label, padding)
    return image, label


def read_tfrecord(
    fp,
    classes_file,
    image_shape,
    max_boxes,
    shuffle_buffer_size,
    batch_size,
    anchors,
    masks,
    classes_delimiter='\n',
):
    """
    Read .tfrecord file(s).
    Args:
        fp: `file_pattern` passed to `tf.data.Dataset.list_files`
        classes_file: Path to .txt file containing object names \n delimited.
        image_shape: Image shape passed to `tf.image.resize`.
        max_boxes: Maximum total boxes per image.
        shuffle_buffer_size: `buffer_size` passed to `tf.data.Dataset.shuffle`.
        batch_size: Batch size passed to `tf.data.Dataset.batch`.
        anchors: Anchors as numpy array.
        masks: Masks as numpy array.
        classes_delimiter: Delimiter used in classes file.

    Returns:
        `tf.data.Dataset`
    """
    text_initializer = tf.lookup.TextFileInitializer(
        classes_file, tf.string, 0, tf.int64, -1, delimiter=classes_delimiter
    )
    class_table = tf.lookup.StaticHashTable(text_initializer, -1)
    files = tf.data.Dataset.list_files(fp)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    feature_map = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'x0': tf.io.VarLenFeature(tf.float32),
        'y0': tf.io.VarLenFeature(tf.float32),
        'x1': tf.io.VarLenFeature(tf.float32),
        'y1': tf.io.VarLenFeature(tf.float32),
        'object_name': tf.io.VarLenFeature(tf.string),
        'object_index': tf.io.VarLenFeature(tf.int64),
    }
    return (
        dataset.map(
            lambda x: read_example(x, feature_map, class_table, max_boxes, image_shape),
            tf.data.experimental.AUTOTUNE,
        )
        .batch(batch_size)
        .shuffle(shuffle_buffer_size)
        .map(
            lambda x, y: (
                tf.image.resize(x, image_shape),
                transform_labels(y, anchors, masks, image_shape[0]),
            )
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
