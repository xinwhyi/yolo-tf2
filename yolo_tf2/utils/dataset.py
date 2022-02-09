import random

import tensorflow as tf


def transform_images(x, image_shape):
    x = tf.image.resize(x, image_shape)
    return x / 255


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_indices):
    n = tf.shape(y_true)[0]
    y_true_out = tf.zeros((n, grid_size, grid_size, tf.shape(anchor_indices)[0], 6))
    anchor_indices = tf.cast(anchor_indices, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(n):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(anchor_indices, tf.cast(y_true[i][j][5], tf.int32))
            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]
                )
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]
                )
                idx += 1
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transform_targets(y, anchors, anchor_masks, size):
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
    for anchor_indices in anchor_masks:
        y_outs.append(transform_targets_for_output(y, grid_size, anchor_indices))
        grid_size *= 2
    return tuple(y_outs)


def serialize_example(image_path, labels, writer):
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


def create_tfrecord(output_path, grouped_labels):
    total_examples = len(grouped_labels)
    with tf.io.TFRecordWriter(output_path) as writer:
        for i, (image_path, labels) in enumerate(grouped_labels):
            print(f'\rWriting example: {i + 1}/{total_examples}', end='')
            serialize_example(image_path, labels, writer)
    print()


def read_example(
    example,
    feature_map,
    class_table,
    max_boxes,
    image_shape,
):
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
    text_init = tf.lookup.TextFileInitializer(
        classes_file, tf.string, 0, tf.int64, -1, delimiter=classes_delimiter
    )
    class_table = tf.lookup.StaticHashTable(text_init, -1)
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
                transform_images(x, image_shape),
                transform_targets(y, anchors, masks, image_shape[0]),
            )
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
