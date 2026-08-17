from typing import Union, Tuple

import numpy as np
import pandas as pd


def find_best_split(
        feature_vector: Union[np.ndarray, pd.DataFrame],
        target_vector: Union[np.ndarray, pd.Series],
        task: str = "classification",
        feature_type: str = "real"
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)
    :param task: либо `classification`, либо `regression`
    :param feature_type: либо `real`, либо `categorical`

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis/variances: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best/variance_best: оптимальное значение критерия Джини/дисперсии (число)
    """

    def gini(targets):
        classes, counts = np.unique(targets, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    if task == 'regression':
        sorted_idx = np.argsort(feature_vector)
        target_vector = target_vector[sorted_idx]
        feature_vector = feature_vector[sorted_idx]
        thresholds = []
        i = 0
        while i < len(feature_vector) - 1:
            if feature_vector[i] != feature_vector[i + 1]:
                thresholds.append((feature_vector[i] + feature_vector[i + 1]) / 2)
            i += 1
        # Cначала посчитаем дисперсию в изначальном узле, а потом посчитаем информационный выигрыш
        variance_head = np.var(target_vector)
        gains = []
        variances = []
        for t in range(len(thresholds)):
            left_mask = feature_vector < thresholds[t]
            right_mask = feature_vector >= thresholds[t]
            left_targets = target_vector[left_mask]
            right_targets = target_vector[right_mask]
            if len(left_targets) == 0 or len(right_targets) == 0:
                continue  # пропускаем такой порог
            variance_left = np.var(left_targets)
            variance_right = np.var(right_targets)
            variance_total = (len(left_targets) / len(target_vector)) * variance_left + (
                    len(right_targets) / len(target_vector)) * variance_right
            variances.append(variance_total)
            gain = variance_head - variance_total
            gains.append(gain)
        argmax = np.argmax(gains)
        threshold_best = thresholds[argmax]
        variance_best = variances[argmax]
        return (thresholds, variances, threshold_best, variance_best)

    elif task == 'classification':  # рассматриваем бинарную классификацию
        sorted_idx = np.argsort(feature_vector)
        target_vector = target_vector[sorted_idx]
        feature_vector = feature_vector[sorted_idx]
        thresholds = []
        i = 0
        while i < len(feature_vector) - 1:
            if feature_vector[i] != feature_vector[i + 1]:
                thresholds.append((feature_vector[i] + feature_vector[i + 1]) / 2)
            i += 1
        ginis = []
        for t in range(len(thresholds)):
            left_mask = feature_vector < thresholds[t]
            right_mask = feature_vector >= thresholds[t]
            left_targets = target_vector[left_mask]
            right_targets = target_vector[right_mask]
            if len(left_targets) == 0 or len(right_targets) == 0:
                continue  # пропускаем такой порог
            gini_right = gini(right_targets)
            gini_left = gini(left_targets)
            gini_total = (len(left_targets) / len(target_vector)) * gini_left + (
                    len(right_targets) / len(target_vector)) * gini_right
            ginis.append(gini_total)
        argmin = np.argmin(ginis)
        threshold_best = thresholds[argmin]
        gini_best = ginis[argmin]
    return (thresholds, ginis, threshold_best, gini_best)
