import numpy as np
import pandas as pd


def get_true_positives(actual, predictions, iou_threshold):
    if 'detection_key' not in predictions.columns:
        predictions['detection_key'] = np.random.default_rng().choice(
            predictions.shape[0], size=predictions.shape[0], replace=False
        )
    merged = actual.merge(predictions, on=['image', 'object_name'])
    merged['x0'] = merged[['x0_x', 'x0_y']].max(1)
    merged['x1'] = merged[['x1_x', 'x1_y']].min(1)
    merged['y0'] = merged[['y0_x', 'y0_y']].max(1)
    merged['y1'] = merged[['y1_x', 'y1_y']].min(1)
    true_intersect = (merged['x1'] > merged['x0']) & (merged['y1'] > merged['y0'])
    merged = merged[true_intersect]
    actual_areas = (merged['x1_x'] - merged['x0_x']) * (merged['y1_x'] - merged['y0_x'])
    predicted_areas = (merged['x1_y'] - merged['x0_y']) * (
        merged['y1_y'] - merged['y0_y']
    )
    intersection_areas = (merged['x1'] - merged['x0']) * (merged['y1'] - merged['y0'])
    merged['iou'] = intersection_areas / (
        actual_areas + predicted_areas - intersection_areas
    )
    merged['true_positive'] = True
    merged['false_positive'] = False
    merged = merged[merged['iou'] >= iou_threshold]
    return merged.drop_duplicates(subset='detection_key')


def get_false_positives(predictions, true_positives):
    keys_before = predictions['detection_key'].values
    keys_after = true_positives['detection_key'].values
    false_keys = np.where(np.isin(keys_before, keys_after, invert=True))
    false_keys = keys_before[false_keys]
    false_positives = predictions.set_index('detection_key').loc[false_keys]
    false_positives['true_positive'] = False
    false_positives['false_positive'] = True
    return false_positives.reset_index()


def calculate_ap(combined, total_actual):
    combined = combined.sort_values(by='score', ascending=False).reset_index(drop=True)
    combined['acc_tp'] = combined['true_positive'].cumsum()
    combined['acc_fp'] = combined['false_positive'].cumsum()
    combined['precision'] = combined['acc_tp'] / (
        combined['acc_tp'] + combined['acc_fp']
    )
    combined['recall'] = combined['acc_tp'] / total_actual
    combined['m_pre1'] = combined['precision'].shift(1, fill_value=0)
    combined['m_pre'] = combined[['m_pre1', 'precision']].max(axis=1)
    combined['m_rec1'] = combined['recall'].shift(1, fill_value=0)
    combined.loc[combined['m_rec1'] != combined['recall'], 'valid_m_rec'] = 1
    combined['average_precision'] = (
        combined['recall'] - combined['m_rec1']
    ) * combined['m_pre']
    return combined


def calculate_stats(
    actual,
    predictions,
    true_positives,
    false_positives,
    combined,
):
    class_stats = []
    for object_name in actual['object_name'].drop_duplicates().values:
        stats = dict()
        stats['object_name'] = object_name
        stats['average_precision'] = (
            combined[combined['object_name'] == object_name]['average_precision'].sum()
            * 100
        )
        stats['actual'] = len(actual[actual['object_name'] == object_name])
        stats['detections'] = len(
            predictions[predictions['object_name'] == object_name]
        )
        stats['true_positives'] = len(
            true_positives[true_positives['object_name'] == object_name]
        )
        stats['false_positives'] = len(
            false_positives[false_positives['object_name'] == object_name]
        )
        stats['combined'] = len(combined[combined['object_name'] == object_name])
        class_stats.append(stats)
    total_stats = pd.DataFrame(class_stats).sort_values(
        by='average_precision', ascending=False
    )
    return total_stats


def calculate_map(actual, predictions, iou_threshold):
    class_counts = actual['object_name'].value_counts().to_dict()
    true_positives = get_true_positives(actual, predictions, iou_threshold)
    false_positives = get_false_positives(predictions, true_positives)
    true_positives = true_positives[
        [*set(true_positives.columns) & set(false_positives.columns)]
    ]
    false_positives = false_positives[
        [*set(true_positives.columns) & set(false_positives.columns)]
    ]
    combined = pd.concat([true_positives, false_positives])
    combined = pd.concat(
        [
            calculate_ap(group, class_counts.get(object_name))
            for object_name, group in combined.groupby('object_name')
        ]
    )
    return calculate_stats(
        actual, predictions, true_positives, false_positives, combined
    )
