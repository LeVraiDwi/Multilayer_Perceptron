import numpy as np

def recall_precision(y_pred, y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))

    recall = TP / (TP + FN)

    precision = TP / (TP + FP)

    f1_score =  2 * ((precision * recall) / (precision + recall))

    print(f"True positive {TP}, True negative: {TN}, false positive: {FP}, False Negative: {FN}")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print(f"f1_score: {f1_score}")