#!/usr/bin/env python3
"""Plot per-epoch train vs. validation loss from a run's stats CSV.

Reads the <model_name>.csv that save_checkpoint() writes alongside every
checkpoint (mean_loss, val_loss, valtest_loss, ... per epoch) and plots train
vs. validation loss so you can eyeball overfitting. Named, titled, and saved
the same way as plot_augmented_cost_estimation.py's output, so the two are
easy to pair up for the same run.
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/polaris_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Defaults: edit these to change what plain `python plot_loss_curves.py`
# ---- plots, or override any of them at the command line (see --help). ----
CSV_PATH = (
    "saved/models/"
    "est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_employee_bs512_ep50_maxr30/"
    "est_complex_dd_pulluppushdown_ddestfonudf_liboh_gradnorm_mldupl_loopend_loopedge_employee_bs512_ep50_maxr30_20260819_184934_053.csv"
)
TEST_DB = "employee"
TIME_STAMP = "20260819_184928"
OUTPUT_DIR = "results/augmented_plots"
SEED = 42
CARDINALITY = "est"
AUGMENT = "False"
TEST_AUGMENT = "True"
AUGMENT_POOLING = "attention"
AUGMENT_REFINEMENT = "gated_residual"
AUGMENT_COARSE_LAYERS = "1"
AUGMENT_INCLUDE_INV = "False"
AUGMENT_REFINE_RET = "False"
LAMBDA_STRUCT = "0"
SHOW_VALTEST = False  # valtest_loss can spike orders of magnitude above train/val (generalization gap)
LOG_SCALE = False     # log-scale the loss axis; helps when spikes would otherwise dominate the y-range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV_PATH, help="Path to the run's per-epoch stats CSV")
    parser.add_argument("--test-db", default=TEST_DB, help="Held-out test database")
    parser.add_argument("--time-stamp", default=TIME_STAMP, help="Run's GROUP_RUN_TIME (used for the output filename)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save the plot into")
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--cardinality", default=CARDINALITY)
    parser.add_argument("--augment", default=AUGMENT)
    parser.add_argument("--test-augment", default=TEST_AUGMENT)
    parser.add_argument("--augment-pooling", default=AUGMENT_POOLING)
    parser.add_argument("--augment-refinement", default=AUGMENT_REFINEMENT)
    parser.add_argument("--augment-coarse-layers", default=AUGMENT_COARSE_LAYERS)
    parser.add_argument("--augment-include-inv", default=AUGMENT_INCLUDE_INV)
    parser.add_argument("--augment-refine-ret", default=AUGMENT_REFINE_RET)
    parser.add_argument("--lambda-struct", default=LAMBDA_STRUCT)
    parser.add_argument("--show-valtest", action="store_true", default=SHOW_VALTEST,
                        help="Also plot valtest_loss (the actual held-out DB)")
    parser.add_argument("--log-scale", action="store_true", default=LOG_SCALE,
                        help="Log-scale the loss axis instead of clipping the view to a robust range")
    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "unknown"


def format_settings(args: argparse.Namespace) -> str:
    settings = f"SEED={args.seed} | CARDINALITY={args.cardinality} | AUGMENT={args.augment}"
    if str(args.augment).strip().lower() == "true":
        settings += (
            f" | TEST_AUGMENT={args.test_augment}\n"
            f"AUGMENT_POOLING={args.augment_pooling} | AUGMENT_REFINEMENT={args.augment_refinement} | "
            f"AUGMENT_COARSE_LAYERS={args.augment_coarse_layers}\n"
            f"AUGMENT_INCLUDE_INV={args.augment_include_inv} | AUGMENT_REFINE_RET={args.augment_refine_ret} | "
            f"LAMBDA_STRUCT={args.lambda_struct}"
        )
    return settings


def robust_ylim(*series: pd.Series, upper_pct: float = 90, pad_frac: float = 0.15):
    """Clip to the Nth percentile of the data (not min/max), so rare loss spikes don't
    stretch the axis and flatten the part of the curve that actually matters. With small
    epoch counts, percentile clipping needs to be well below 99 to have any effect at all
    (99th percentile of ~50 points is essentially the max)."""
    values = pd.concat([s.dropna() for s in series])
    lo = values.min()
    hi = np.percentile(values, upper_pct)
    pad = (hi - lo) * pad_frac or hi * pad_frac or 1.0
    return max(0.0, lo - pad * 0.3), hi + pad


def create_plot(csv_path: str, test_db: str, time_stamp: str, output_dir: str, args: argparse.Namespace) -> Path:
    df = pd.read_csv(csv_path)

    figure, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(13, 5.5))

    loss_ax.plot(df["epoch"], df["mean_loss"], label="Train loss", color="#4C78A8", linewidth=1.8)
    loss_ax.plot(df["epoch"], df["val_loss"], label="Val loss", color="#F58518", linewidth=1.8)
    if args.show_valtest and "valtest_loss" in df.columns:
        loss_ax.plot(df["epoch"], df["valtest_loss"], label="Valtest loss (held-out DB)",
                    color="#E45756", linewidth=1.4, linestyle="--")

    if args.log_scale:
        loss_ax.set_yscale("log")
        loss_ax.set_ylabel("Loss (log scale)")
    else:
        series = [df["mean_loss"], df["val_loss"]]
        if args.show_valtest and "valtest_loss" in df.columns:
            series.append(df["valtest_loss"])
        loss_ax.set_ylim(robust_ylim(*series))
        loss_ax.set_ylabel("Loss")

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_title("Train vs. Validation Loss", fontsize=12, fontweight="bold")
    loss_ax.legend(frameon=False)
    loss_ax.grid(axis="y", linestyle="--", alpha=0.3)
    loss_ax.set_axisbelow(True)

    # "Accuracy" proxy: median q-error (q50). val is a frozen end-of-epoch snapshot via
    # validate_model(); train is computed cheaply inline in train_epoch() from the
    # predictions/labels the training batches already produce (no extra pass) -- see
    # models/training/train.py. Train reflects the model as it evolved *during* the
    # epoch, so it's not a perfectly like-for-like snapshot vs. val, same caveat as
    # mean_loss vs. val_loss on the left.
    acc_series = [df[col] for col in ("train_median_q_error_50", "val_median_q_error_50") if col in df.columns]
    if "train_median_q_error_50" in df.columns:
        acc_ax.plot(df["epoch"], df["train_median_q_error_50"], label="Train q50", color="#4C78A8", linewidth=1.8)
    if "val_median_q_error_50" in df.columns:
        acc_ax.plot(df["epoch"], df["val_median_q_error_50"], label="Val q50", color="#F58518", linewidth=1.8)
    if acc_series:
        if not args.log_scale:
            acc_ax.set_ylim(robust_ylim(*acc_series))
        else:
            acc_ax.set_yscale("log")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Q-error (log scale)" if args.log_scale else "Q-error")
    acc_ax.set_title("Train vs. Validation Q-error", fontsize=12, fontweight="bold")
    acc_ax.legend(frameon=False)
    acc_ax.grid(axis="y", linestyle="--", alpha=0.3)
    acc_ax.set_axisbelow(True)

    figure.suptitle(f"Train vs. Validation Loss & Accuracy — Test DB: {test_db} | Cardinality: {args.cardinality}",
                    fontsize=13, fontweight="bold", y=1.06)
    figure.text(0.5, 0.98, format_settings(args), ha="center", va="top", fontsize=8, linespacing=1.5)
    figure.tight_layout(rect=[0, 0, 1, 0.78])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{safe_filename_part(test_db)}_{safe_filename_part(time_stamp)}_loss.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = create_plot(args.csv, args.test_db, args.time_stamp, args.output_dir, args)
    print(f"Loss curve plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
