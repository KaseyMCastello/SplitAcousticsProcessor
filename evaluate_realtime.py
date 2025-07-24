import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for 2D time-frequency boxes.P
    Each box = [start_time, end_time, low_freq, high_freq]
    """
    t1_start, t1_end, f1_low, f1_high = box1
    t2_start, t2_end, f2_low, f2_high = box2

    t_inter = max(0, (min(t1_end, t2_end) - max(t1_start, t2_start)).total_seconds())
    f_inter = max(0, (min(f1_high, f2_high) - max(f1_low, f2_low)))
    intersection = t_inter * f_inter

    area1 = (t1_end - t1_start).total_seconds() * (f1_high - f1_low)
    area2 = (t2_end - t2_start).total_seconds() * (f2_high - f2_low)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def match_predictions_to_gt(pred_df, gt_df, iou_threshold=0.5):
    """
    Match predictions to ground truth using IoU.
    Returns TP, FP, FN per label.
    """
    labels = sorted(set(gt_df['label']) | set(pred_df['label']))
    stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    #I want a specific value for each label, and one for overall
    for label in labels:
        #Get all the instances of the label in both the ground truth and in the predictions
        gt_label = gt_df[gt_df['label'] == label].copy()
        pred_label = pred_df[pred_df['label'] == label].copy()

        gt_label['matched'] = False
        pred_label['matched'] = False

        for pred_idx, pred_row in pred_label.iterrows():
            best_iou = 0
            best_gt_idx = None

            pred_box = [ pred_row['start_time'], pred_row['end_time'], pred_row["min_frequency"], pred_row["max_frequency"]]
            
            for gt_idx, gt_row in gt_label.iterrows():
                if gt_label.at[gt_idx, 'matched']:
                    continue

                gt_box = [gt_row['start_time_abs'], gt_row['end_time_abs'], gt_row['low_f'], gt_row['high_f']]
                
                iou = compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                stats[label]['TP'] += 1
                gt_label.at[best_gt_idx, 'matched'] = True
                pred_label.at[pred_idx, 'matched'] = True
            else:
                stats[label]['FP'] += 1
        tp_pred = pred_label[pred_label['matched']]
        print("Maximum time for true positive prediction of type", label, "is", tp_pred['end_time'].max())
        stats[label]['FN'] += (~gt_label['matched']).sum()

    return stats

def compute_metrics(stats):
    """
    Compute precision, recall, F1 per label and overall.
    """
    results = {}
    total_TP = total_FP = total_FN = 0

    for label, s in stats.items():
        TP, FP, FN = s['TP'], s['FP'], s['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[label] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': TP, 'FP': FP, 'FN': FN
        }
        total_TP += TP
        total_FP += FP
        total_FN += FN

    # Overall metrics
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results['Overall'] = {
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1': overall_f1,
        'TP': total_TP, 'FP': total_FP, 'FN': total_FN
    }

    return results


if __name__ == "__main__":
    # Load CSVs
    ground_truth = pd.read_excel(r"C:\Users\kasey\OneDrive - UC San Diego\Lab Work\RealTimeBaleen\Ground_Truth_SOCAL34N_sitN_090722_200000.xlsx")
    predictions = pd.read_excel(r"C:\Users\Kasey\Desktop\TimingTestSimUDP2\output.xlsx")

    # Limit ground truth to events that start before end of predictions
    max_pred_time = predictions['end_time'].max()
    ground_truth = ground_truth[ground_truth['start_time_abs'] <= max_pred_time]
    stats = match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.0001)
    metrics = compute_metrics(stats)

    # Print results
    for label, result in metrics.items():
        print(f"\nLabel: {label}")
        for k, v in result.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    
    
    
    


