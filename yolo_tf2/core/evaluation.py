import numpy as np
import pandas as pd


def get_true_positives(actual, detections, iou_threshold):
    """
    Identify and flag true positive detections.
    Args:
        actual: Ground truth data as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`
            as columns.
        detections: Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.

    Returns:
        pd.DataFrame containing filtered out true positives.
    """
    if 'detection_key' not in detections.columns:
        detections['detection_key'] = np.random.default_rng().choice(
            detections.shape[0], size=detections.shape[0], replace=False
        )
    merged = actual.merge(detections, on=['image', 'object_name'])
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


def get_false_positives(detections, true_positives):
    """
    Filter out False positives.
    Args:
        detections: Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        true_positives: `pd.DataFrame` of true positive detections, the result
            of `get_true_positives`.

    Returns:
        `pd.DataFrame` containing filtered out false positives.
    """
    keys_before = detections['detection_key'].values
    keys_after = true_positives['detection_key'].values
    false_keys = np.where(np.isin(keys_before, keys_after, invert=True))
    false_keys = keys_before[false_keys]
    false_positives = detections.set_index('detection_key').loc[false_keys]
    false_positives['true_positive'] = False
    false_positives['false_positive'] = True
    return false_positives.reset_index()


def calculate_ap(combined, total_actual):
    """
    Calculate single object average precision.
    Args:
        combined: `pd.DataFrame` containing true positives + false positives.
        total_actual: Total instances of an object in the dataset.

    Returns:
        Updated combined with average precision calculated.
    """
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
    detections,
    true_positives,
    false_positives,
    combined,
):
    """
    Calculate display data including total actual, total true positives, total false
    positives and sort resulting `pd.DataFrame` by object average precision.
    Args:
        actual: Ground truth data as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`
            as columns.
        detections: Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        true_positives: `pd.DataFrame` of true positive detections, the result
            of `get_true_positives`.
        false_positives: `pd.DataFrame` of false positive detections, the result
            of `get_false_positives`.
        combined: `pd.DataFrame` containing true positives + false positives.

    Returns:
        `pd.DataFrame` with calculated average precisions per dataset object.
    """
    class_stats = []
    for object_name in actual['object_name'].drop_duplicates().values:
        stats = dict()
        stats['object_name'] = object_name
        stats['average_precision'] = (
            combined[combined['object_name'] == object_name]['average_precision'].sum()
            * 100
        )
        stats['actual'] = actual[actual['object_name'] == object_name].shape[0]
        stats['detections'] = detections[
            detections['object_name'] == object_name
        ].shape[0]
        stats['true_positives'] = true_positives[
            true_positives['object_name'] == object_name
        ].shape[0]
        stats['false_positives'] = false_positives[
            false_positives['object_name'] == object_name
        ].shape[0]
        stats['combined'] = combined[combined['object_name'] == object_name].shape[0]
        class_stats.append(stats)
    total_stats = pd.DataFrame(class_stats).sort_values(
        by='average_precision', ascending=False
    )
    return total_stats


def calculate_map(actual, detections, iou_threshold):
    """
    Calculate average precision per dataset object. The mean of the resulting
    `pd.DataFrame` `average_precision` column is the mAP score.
    Args:
        actual: Ground truth data as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`
            as columns.
        detections: Detections as `pd.DataFrame` having
            `image`, `object_name`, `object_index`, `x0`, `y0`, `x1`, `y1`,
            `score` as columns.
        iou_threshold: Percentage above which detections overlapping with
            ground truths are considered true positive.
    Returns:
        `pd.DataFrame`, the result of `calculate_stats`.
    """
    class_counts = actual['object_name'].value_counts().to_dict()
    true_positives = get_true_positives(actual, detections, iou_threshold)
    false_positives = get_false_positives(detections, true_positives)
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
        actual, detections, true_positives, false_positives, combined
    )
