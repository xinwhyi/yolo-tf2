import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import cv2
from yolo_tf2.utils.common import get_boxes


def get_nms(outputs, max_boxes, iou_threshold, score_threshold):
    boxes, confidences, type_ = [], [], []
    for output in outputs:
        boxes.append(
            tf.reshape(
                output[0],
                (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1]),
            )
        )
        confidences.append(
            tf.reshape(
                output[1],
                (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1]),
            )
        )
        type_.append(
            tf.reshape(
                output[2],
                (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1]),
            )
        )
    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(confidences, axis=1)
    class_probabilities = tf.concat(type_, axis=1)
    scores = confidence * class_probabilities
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes,
        max_total_size=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return boxes, scores, classes, valid_detections


def predictions_to_boxes(
    predictions,
    anchors,
    masks,
    total_classes,
    max_boxes,
    iou_threshold,
    score_threshold,
):
    output_0, output_1, output_2 = predictions
    boxes_0 = get_boxes(output_0, anchors[masks[0]], total_classes)
    boxes_1 = get_boxes(output_1, anchors[masks[1]], total_classes)
    boxes_2 = get_boxes(output_2, anchors[masks[2]], total_classes)
    return get_nms(
        [box[:3] for box in [boxes_0, boxes_1, boxes_2]],
        max_boxes,
        iou_threshold,
        score_threshold,
    )


def detect_images(
    images,
    model,
    class_names,
    anchors,
    masks,
    max_boxes,
    iou_threshold,
    score_threshold,
    **kwargs,
):
    adjusted_images = []
    for image in images:
        with open(image, 'rb') as raw_image:
            image = tf.image.decode_png(raw_image.read(), channels=3)
        image = tf.image.resize(image, model.input_shape[1:-1])
        adjusted_images.append(image)
    predictions = model.predict(tf.stack(adjusted_images) / 255, **kwargs)
    predictions = predictions_to_boxes(
        predictions,
        anchors,
        masks,
        len(class_names),
        max_boxes,
        iou_threshold,
        score_threshold,
    )
    outputs = []
    for image, boxes, scores, classes, total in zip(images, *predictions):
        df = pd.DataFrame(boxes[:total], columns=['x0', 'y0', 'x1', 'y1'])
        df['image'] = image
        df['object_index'] = classes[:total]
        df['object_name'] = df['object_index'].apply(class_names.get)
        df['score'] = scores[:total]
        outputs.append(df)
    return pd.concat(outputs)


def draw_boxes(image, colors, detections, font_scale=0.6):
    for i, row in detections.iterrows():
        x0 = int(row['x0'] * image.shape[1])
        y0 = int(row['y0'] * image.shape[0])
        x1 = int(row['x1'] * image.shape[1])
        y1 = int(row['y1'] * image.shape[0])
        color = colors.get(row['object_name'])
        cv2.rectangle(
            image,
            (x0, y0),
            (x1, y1),
            color,
            1,
        )
        cv2.putText(
            image,
            f"{row['object_name']}-{round(row['score'], 2)}",
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            font_scale,
            color,
        )


def detect_vid(
    src,
    dest,
    model,
    colors,
    class_names,
    anchors,
    masks,
    max_boxes,
    iou_threshold,
    score_threshold,
    codec='mp4v',
    display=False,
    **kwargs,
):
    vid = cv2.VideoCapture(src)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    current = 1
    codec = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(dest, codec, fps, (width, height))
    while vid.isOpened():
        _, frame = vid.read()
        detections = detect_images(
            np.expand_dims(frame, 0),
            model,
            class_names,
            anchors,
            masks,
            max_boxes,
            iou_threshold,
            score_threshold,
            **kwargs,
        )
        draw_boxes(frame, colors, detections)
        writer.write(frame)
        completed = f'{(current / length) * 100}% completed'
        print(
            f'\rframe {current}/{length}\tdetections: '
            f'{len(detections)}\tcompleted: {completed}',
            end='',
        )
        if display:
            cv2.destroyAllWindows()
            cv2.imshow(f'frame {current}', frame)
        current += 1
        if cv2.waitKey(1) == ord('q'):
            print(f'Video detection aborted {current}/{length} ' f'frames completed')
            break
