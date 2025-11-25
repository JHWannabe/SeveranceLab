import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from xml.etree import ElementTree
from zipfile import ZipFile


NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _load_shared_strings(z: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    with z.open("xl/sharedStrings.xml") as handle:
        root = ElementTree.parse(handle).getroot()
    strings: List[str] = []
    for si in root.findall(f"{NAMESPACE}si"):
        text_node = si.find(f"{NAMESPACE}t")
        if text_node is not None and text_node.text is not None:
            strings.append(text_node.text)
            continue
        parts = [node.text or "" for node in si.findall(f"{NAMESPACE}r/{NAMESPACE}t")]
        strings.append("".join(parts))
    return strings


def _parse_cell(cell, shared_strings: List[str]) -> str:
    value_node = cell.find(f"{NAMESPACE}v")
    if value_node is None or value_node.text is None:
        return ""
    cell_type = cell.get("t")
    if cell_type == "s":
        index = int(value_node.text)
        return shared_strings[index]
    return value_node.text


def load_xlsx(path: Path) -> Tuple[List[str], List[List[str]]]:
    with ZipFile(path) as z:
        shared_strings = _load_shared_strings(z)
        with z.open("xl/worksheets/sheet1.xml") as handle:
            root = ElementTree.parse(handle).getroot()
        rows = []
        for row in root.findall(f".//{NAMESPACE}row"):
            values = [_parse_cell(cell, shared_strings) for cell in row.findall(f"{NAMESPACE}c")]
            rows.append(values)
    header, *records = rows
    return header, records


def _parse_value(name: str, raw: str) -> float:
    if name == "진료년월일":
        # Convert date to ordinal for a numeric representation
        try:
            date_obj = datetime.strptime(raw, "%Y-%m-%d").date()
            return float(date_obj.toordinal())
        except Exception:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


@dataclass
class Dataset:
    feature_names: List[str]
    X: List[List[float]]
    y: List[int]


def prepare_dataset(path: Path) -> Dataset:
    header, rows = load_xlsx(path)
    target_col = "Osteoporosis"
    skip_cols = {"File Analyzed", target_col}
    indices = [i for i, name in enumerate(header) if name not in skip_cols]
    feature_names = [header[i] for i in indices]
    target_index = header.index(target_col)

    X: List[List[float]] = []
    y: List[int] = []
    for row in rows:
        if len(row) <= target_index:
            continue
        try:
            label = int(float(row[target_index]))
        except ValueError:
            continue
        features = [_parse_value(header[i], row[i] if i < len(row) else "") for i in indices]
        if len(features) != len(feature_names):
            continue
        X.append(features)
        y.append(label)
    return Dataset(feature_names, X, y)


def train_test_split(X: List[List[float]], y: List[int], test_size: float = 0.2, seed: int = 42):
    combined = list(zip(X, y))
    random.Random(seed).shuffle(combined)
    split = int(len(combined) * (1 - test_size))
    train = combined[:split]
    test = combined[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)


class StandardScaler:
    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []

    def fit(self, X: List[List[float]]):
        cols = len(X[0])
        self.means = []
        self.stds = []
        for j in range(cols):
            col_values = [row[j] for row in X]
            mean = sum(col_values) / len(col_values)
            variance = sum((v - mean) ** 2 for v in col_values) / len(col_values)
            std = math.sqrt(variance) or 1.0
            self.means.append(mean)
            self.stds.append(std)

    def transform(self, X: List[List[float]]):
        transformed = []
        for row in X:
            transformed.append([(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))])
        return transformed

    def fit_transform(self, X: List[List[float]]):
        self.fit(X)
        return self.transform(X)


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.01, epochs: int = 50, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: List[float] = []
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0 for _ in range(n_features)]
            grad_b = 0.0
            for xi, yi in zip(X, y):
                z = self.bias + sum(w * v for w, v in zip(self.weights, xi))
                pred = 1 / (1 + math.exp(-z))
                error = pred - yi
                for j in range(n_features):
                    grad_w[j] += error * xi[j]
                grad_b += error
            for j in range(n_features):
                grad_w[j] = (grad_w[j] / n_samples) + self.l2 * self.weights[j]
                self.weights[j] -= self.lr * grad_w[j]
            self.bias -= self.lr * (grad_b / n_samples)
        return self

    def predict_proba(self, X: List[List[float]]):
        probs = []
        for xi in X:
            z = self.bias + sum(w * v for w, v in zip(self.weights, xi))
            prob = 1 / (1 + math.exp(-z))
            probs.append(prob)
        return probs

    def predict(self, X: List[List[float]], threshold: float = 0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


@dataclass
class GridSearchResult:
    best_estimator_: LogisticRegressionGD
    best_params_: Dict[str, float]
    cv_scores_: Dict[float, float]


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def kfold_indices(n_samples: int, k: int = 2, seed: int = 42):
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size
    return folds


def grid_search_l2(X: List[List[float]], y: List[int], l2_values: List[float]) -> GridSearchResult:
    folds = kfold_indices(len(X))
    scores: Dict[float, float] = {}
    for l2_value in l2_values:
        fold_scores = []
        for i in range(len(folds)):
            val_idx = set(folds[i])
            train_idx = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
            X_train = [X[idx] for idx in train_idx]
            y_train = [y[idx] for idx in train_idx]
            X_val = [X[idx] for idx in val_idx]
            y_val = [y[idx] for idx in val_idx]
            model = LogisticRegressionGD(l2=l2_value)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            fold_scores.append(f1_score(y_val, preds))
        scores[l2_value] = sum(fold_scores) / len(fold_scores)
    best_l2 = max(scores, key=scores.get)
    best_model = LogisticRegressionGD(l2=best_l2)
    best_model.fit(X, y)
    return GridSearchResult(best_model, {"l2": best_l2}, scores)


def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, fp, tn, fn


def precision_recall(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def classification_report(y_true: List[int], y_pred: List[int]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for label in [0, 1]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for yt in y_true if yt == label)
        report[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
    return report


def roc_curve(y_true: List[int], probs: List[float]):
    thresholds = sorted(set(probs), reverse=True)
    thresholds = [float("inf")] + thresholds + [0.0]
    points = []
    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in probs]
        tp, fp, tn, fn = confusion_matrix(y_true, preds)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        points.append((fpr, tpr, thr))
    return points


def auc(points: List[Tuple[float, float, float]]):
    sorted_points = sorted(points, key=lambda x: x[0])
    area = 0.0
    for (x1, y1, _), (x2, y2, _) in zip(sorted_points[:-1], sorted_points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2
    return area


def precision_recall_curve(y_true: List[int], probs: List[float]):
    thresholds = sorted(set(probs), reverse=True)
    thresholds = thresholds + [0.0]
    points = []
    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in probs]
        precision, recall = precision_recall(y_true, preds)
        points.append((recall, precision, thr))
    return points


def sweep_thresholds(y_true: List[int], probs: List[float], thresholds: List[float]):
    sweep = []
    for thr in thresholds:
        preds = [1 if p >= thr else 0 for p in probs]
        tp, fp, tn, fn = confusion_matrix(y_true, preds)
        precision, recall = precision_recall(y_true, preds)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = f1_score(y_true, preds)
        youden = recall + specificity - 1
        sweep.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "youden_j": youden,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    sweep.sort(key=lambda row: (row["youden_j"], row["f1"], row["recall"]), reverse=True)
    return sweep


def save_svg_curve(points: List[Tuple[float, float, float]], x_label: str, y_label: str, title: str, path: Path):
    width, height = 640, 480
    margin = 60
    svg_points = []
    for x, y, _ in points:
        px = margin + x * (width - 2 * margin)
        py = height - margin - y * (height - 2 * margin)
        svg_points.append((px, py))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in svg_points)
    ticks = [i / 5 for i in range(6)]
    with path.open("w") as handle:
        handle.write("<?xml version='1.0' encoding='UTF-8'?>\n")
        handle.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>\n")
        handle.write(f"  <rect width='{width}' height='{height}' fill='white' stroke='none'/>\n")
        # Axes
        handle.write(
            f"  <line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black'/>\n"
        )
        handle.write(f"  <line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='black'/>\n")
        # Ticks and labels
        for t in ticks:
            x = margin + t * (width - 2 * margin)
            y = height - margin - t * (height - 2 * margin)
            handle.write(f"  <line x1='{x}' y1='{height - margin}' x2='{x}' y2='{height - margin + 5}' stroke='black'/>\n")
            handle.write(f"  <text x='{x}' y='{height - margin + 20}' font-size='12' text-anchor='middle'>{t:.1f}</text>\n")
            handle.write(f"  <line x1='{margin - 5}' y1='{y}' x2='{margin}' y2='{y}' stroke='black'/>\n")
            handle.write(f"  <text x='{margin - 10}' y='{y + 4}' font-size='12' text-anchor='end'>{t:.1f}</text>\n")
        # Labels and title
        handle.write(f"  <text x='{width/2}' y='{height - 15}' font-size='14' text-anchor='middle'>{x_label}</text>\n")
        handle.write(
            f"  <text x='{15}' y='{height/2}' font-size='14' text-anchor='middle' transform='rotate(-90 {15},{height/2})'>{y_label}</text>\n"
        )
        handle.write(f"  <text x='{width/2}' y='{30}' font-size='16' text-anchor='middle' font-weight='bold'>{title}</text>\n")
        # Curve
        handle.write(f"  <polyline points='{polyline}' fill='none' stroke='blue' stroke-width='2'/>\n")
        handle.write("</svg>\n")


def evaluate():
    dataset = prepare_dataset(Path("breast_data_250812.xlsx"))
    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    search = grid_search_l2(X_train_scaled, y_train, [0.0, 0.01, 0.1])
    model = search.best_estimator_

    test_probs = model.predict_proba(X_test_scaled)
    test_preds = model.predict(X_test_scaled)

    report = classification_report(y_test, test_preds)
    cm_tp, cm_fp, cm_tn, cm_fn = confusion_matrix(y_test, test_preds)
    print("Best params:", search.best_params_)
    print("CV scores:", search.cv_scores_)
    print("Classification report:", report)
    print("Confusion matrix (tp, fp, tn, fn):", (cm_tp, cm_fp, cm_tn, cm_fn))

    roc_points = roc_curve(y_test, test_probs)
    roc_auc = auc(roc_points)
    print("ROC AUC:", roc_auc)

    pr_points = precision_recall_curve(y_test, test_probs)
    # Area under PR using trapezoid on recall sorted
    sorted_pr = sorted(pr_points, key=lambda x: x[0])
    pr_auc = 0.0
    for (r1, p1, _), (r2, p2, _) in zip(sorted_pr[:-1], sorted_pr[1:]):
        pr_auc += (r2 - r1) * (p1 + p2) / 2
    print("PR AUC:", pr_auc)

    thresholds = [round(t, 2) for t in [0.1 * i for i in range(1, 10)]]
    sweep = sweep_thresholds(y_test, test_probs, thresholds)
    best_threshold = sweep[0]["threshold"]
    print("Top threshold candidates:")
    for entry in sweep[:5]:
        print(entry)

    # Re-evaluate at best threshold
    best_preds = [1 if p >= best_threshold else 0 for p in test_probs]
    best_report = classification_report(y_test, best_preds)
    best_tp, best_fp, best_tn, best_fn = confusion_matrix(y_test, best_preds)
    print("Best threshold:", best_threshold)
    print("Metrics at best threshold:", {
        "classification_report": best_report,
        "confusion_matrix": (best_tp, best_fp, best_tn, best_fn),
    })

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    save_svg_curve(roc_points, "False Positive Rate", "True Positive Rate", "ROC Curve", reports_dir / "roc_curve.svg")
    save_svg_curve(pr_points, "Recall", "Precision", "Precision-Recall Curve", reports_dir / "pr_curve.svg")


if __name__ == "__main__":
    evaluate()
