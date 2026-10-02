"""Object detection metrics implemented from first principles (precision, recall, F1, average precision)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyraguard.schemas import BBox, Detection


@dataclass
class ClassStats:
    scores: list[float] = field(default_factory=list)
    hits: list[bool] = field(default_factory=list)
    ground_truth: int = 0

    def add_image(self, detections: list[Detection], truths: list[BBox], iou_threshold: float) -> list[bool]:
        """Greedy matching in confidence order. Returns which ground truth boxes were found."""
        self.ground_truth += len(truths)
        matched = [False] * len(truths)
        for det in sorted(detections, key=lambda d: -d.confidence):
            best, best_iou = -1, iou_threshold
            for j, truth in enumerate(truths):
                if matched[j]:
                    continue
                overlap = det.bbox.iou(truth)
                if overlap >= best_iou:
                    best, best_iou = j, overlap
            self.scores.append(det.confidence)
            self.hits.append(best >= 0)
            if best >= 0:
                matched[best] = True
        return matched

    def summary(self) -> dict[str, float]:
        tp = int(sum(self.hits))
        fp = len(self.hits) - tp
        fn = self.ground_truth - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / self.ground_truth if self.ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "ground_truth": self.ground_truth, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "average_precision": round(self.average_precision(), 4),
        }

    def average_precision(self) -> float:
        """Area under the precision recall curve with the monotone precision envelope (COCO and VOC 2010 style)."""
        if not self.scores or not self.ground_truth:
            return 0.0
        order = np.argsort(-np.asarray(self.scores))
        hits = np.asarray(self.hits, dtype=np.float64)[order]
        tp = np.cumsum(hits)
        fp = np.cumsum(1.0 - hits)
        recall = np.concatenate([[0.0], tp / self.ground_truth, [1.0]])
        precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1e-12), [0.0]])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        steps = np.where(recall[1:] != recall[:-1])[0]
        return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def binary_summary(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
        "accuracy": round((tp + tn) / max(tp + fp + tn + fn, 1), 4),
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
    }
