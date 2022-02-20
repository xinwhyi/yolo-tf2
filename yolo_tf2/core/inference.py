import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import cv2


def detect_images(images, model, class_names, **kwargs):
    adjusted_images = []
    for image in images:
        with open(image, 'rb') as raw_image:
            image = tf.image.decode_png(raw_image.read(), channels=3)
        image = tf.image.resize(image, model.input_shape[1:-1])
        adjusted_images.append(image)
    predictions = model.predict(tf.stack(adjusted_images) / 255, **kwargs)
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
    src, dest, model, colors, class_names, codec='mp4v', display=False, **kwargs
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
            np.expand_dims(frame, 0), model, class_names, **kwargs
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
