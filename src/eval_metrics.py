"""Evaluation metrics that do not revolve around plain accuracy.

This module computes per-task metrics over the 27 SIDER tasks (ROC-AUC, AUPR,
balanced accuracy, macro F1/precision/recall), marks single-class tasks as NaN
and excludes them from the averages, and provides molecule-level bootstrap
confidence intervals plus a paired significance test.

Contract: `y_true` and `y_prob` have shape (n_molecules, n_tasks). In `y_true`,
-1 means "unknown/unlabelled" (as in the literature labels of the 70-molecule
test); those cells are dropped from that task's computation, including the
single-class check.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class TaskMetric:
    task: int
    n: int
    n_pos: int
    excluded: bool  # single-class task (AUC/AUPR undefined) -> NaN, left out of the averages
    auc: float = float("nan")
    aupr: float = float("nan")
    balanced_acc: float = float("nan")
    f1: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")


@dataclass
class MacroReport:
    threshold: float
    n_tasks_total: int
    n_tasks_excluded: int  # tasks dropped because AUC/AUPR is undefined
    macro_auc: float
    macro_aupr: float
    balanced_accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    # kept only to compare against older reports; no longer a primary metric
    accuracy_legacy: float
    per_task: list = field(default_factory=list)


def _valid_mask(y_true_col):
    return y_true_col != -1


def per_task_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> list[TaskMetric]:
    """Metrics computed separately for each task. For a single-class task
    (only positives or only negatives in the test slice) AUC and AUPR are
    undefined, so they are set to NaN and the task is marked `excluded=True`;
    such tasks do not enter the macro averages, rather than letting sklearn
    return a quietly wrong value.
    """
    out = []
    n_tasks = y_true.shape[1]
    for t in range(n_tasks):
        mask = _valid_mask(y_true[:, t])
        yt, yp = y_true[mask, t], y_prob[mask, t]
        n, n_pos = len(yt), int(yt.sum())
        if n == 0 or n_pos == 0 or n_pos == n:
            out.append(TaskMetric(task=t, n=n, n_pos=n_pos, excluded=True))
            continue
        pred = (yp >= threshold).astype(int)
        out.append(TaskMetric(
            task=t, n=n, n_pos=n_pos, excluded=False,
            auc=roc_auc_score(yt, yp),
            aupr=average_precision_score(yt, yp),
            balanced_acc=balanced_accuracy_score(yt, pred),
            f1=f1_score(yt, pred, zero_division=0),
            precision=precision_score(yt, pred, zero_division=0),
            recall=recall_score(yt, pred, zero_division=0),
        ))
    return out


def macro_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> MacroReport:
    tasks = per_task_metrics(y_true, y_prob, threshold)
    included = [t for t in tasks if not t.excluded]
    n_excl = len(tasks) - len(included)

    def avg(attr):
        vals = [getattr(t, attr) for t in included]
        return float(np.mean(vals)) if vals else float("nan")

    mask_all = y_true != -1
    tp = int(((y_prob >= threshold) & (y_true == 1) & mask_all).sum())
    tn = int(((y_prob < threshold) & (y_true == 0) & mask_all).sum())
    n_valid = int(mask_all.sum())
    accuracy_legacy = (tp + tn) / n_valid if n_valid else float("nan")

    return MacroReport(
        threshold=threshold, n_tasks_total=len(tasks), n_tasks_excluded=n_excl,
        macro_auc=avg("auc"), macro_aupr=avg("aupr"), balanced_accuracy=avg("balanced_acc"),
        f1_macro=avg("f1"), precision_macro=avg("precision"), recall_macro=avg("recall"),
        accuracy_legacy=accuracy_legacy, per_task=tasks,
    )


def _macro_single(y_true, y_prob, sklearn_fn, needs_threshold, threshold=0.5):
    """A fast macro average for a single metric. Unlike macro_report, it does
    not compute all six metrics on every bootstrap or paired-test replicate,
    only the requested sklearn function, which is up to 6x faster.
    """
    vals = []
    for t in range(y_true.shape[1]):
        mask = _valid_mask(y_true[:, t])
        yt = y_true[mask, t]
        n_pos = int(yt.sum())
        if len(yt) == 0 or n_pos == 0 or n_pos == len(yt):
            continue
        yp = y_prob[mask, t]
        if needs_threshold:
            vals.append(sklearn_fn(yt, (yp >= threshold).astype(int)))
        else:
            vals.append(sklearn_fn(yt, yp))
    return float(np.mean(vals)) if vals else float("nan")


def macro_auc(y_true, y_prob):
    return _macro_single(y_true, y_prob, roc_auc_score, needs_threshold=False)


def macro_aupr(y_true, y_prob):
    return _macro_single(y_true, y_prob, average_precision_score, needs_threshold=False)


def macro_balanced_accuracy(y_true, y_prob, threshold=0.5):
    return _macro_single(y_true, y_prob, balanced_accuracy_score, needs_threshold=True, threshold=threshold)


def macro_f1(y_true, y_prob, threshold=0.5):
    f = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    return _macro_single(y_true, y_prob, f, needs_threshold=True, threshold=threshold)


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 2000,
                  ci: float = 0.95, seed: int = 0) -> dict:
    """A `ci` confidence interval from molecule-level (row) resampling.

    `metric_fn(y_true_sub, y_prob_sub) -> float`, applied to a single scalar
    metric (for example `lambda yt, yp: macro_report(yt, yp).macro_auc`). On a
    small set such as the 70 molecules the interval is expected to be wide:
    that is precisely the reason not to trust a single point estimate, so it is
    reported rather than hidden.
    """
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    point = metric_fn(y_true, y_prob)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_prob[idx])
        except Exception:
            continue
        if not np.isnan(val):
            boots.append(val)
    if len(boots) < n_boot * 0.5:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot_valid": len(boots), "warning": "more than half of the bootstrap replicates were NaN (small or imbalanced set)"}
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot_valid": len(boots)}


def paired_bootstrap_pvalue(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                             metric_fn, n_boot: int = 2000, seed: int = 0) -> dict:
    """A two-sided paired bootstrap p-value for the difference between two
    models on the same molecules. H0: the difference is 0. Molecules are
    resampled together, with a shared index, so the pairing is preserved.
    """
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    obs_diff = metric_fn(y_true, prob_a) - metric_fn(y_true, prob_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            d = metric_fn(y_true[idx], prob_a[idx]) - metric_fn(y_true[idx], prob_b[idx])
        except Exception:
            continue
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"observed_diff": obs_diff, "p_value": float("nan"), "n_boot_valid": 0}
    # Two-sided p-value: the share of the bootstrap distribution on the far
    # side of zero (percentile method), symmetric in the sign of the observed
    # difference.
    p_two_sided = 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    p_two_sided = min(p_two_sided, 1.0)
    return {"observed_diff": float(obs_diff), "p_value": p_two_sided, "n_boot_valid": len(diffs)}


def mcnemar_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test, comparing the two models' binary (correct/incorrect)
    predictions. All (molecule, task) cells are flattened and treated as one
    sample.

    Caveat: different tasks on the same molecule, and different molecules on
    the same task, are not independent, which violates classical McNemar's
    assumption. The p-value is therefore approximate. The primary evidence
    should be `paired_bootstrap_pvalue`, which resamples at the molecule level
    and preserves the dependence; McNemar is a secondary cross-check.
    """
    mask = y_true != -1
    correct_a = (pred_a == y_true) & mask
    correct_b = (pred_b == y_true) & mask
    n01 = int((~correct_a & correct_b).sum())  # A wrong, B right
    n10 = int((correct_a & ~correct_b).sum())  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": float("nan"), "note": "the two models agree everywhere (no difference)"}
    # McNemar with continuity correction; the exact binomial is more accurate for small n
    from scipy import stats
    if n < 25:
        p = stats.binomtest(min(n01, n10), n, 0.5).pvalue
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        p = float(1 - stats.chi2.cdf(chi2, df=1))
    return {"n01": n01, "n10": n10, "p_value": float(p)}
