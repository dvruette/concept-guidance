import argparse
import glob
import json
import functools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


CONCEPT_TO_METRIC_KEY = {
    "compliance": "is_compliant",
    "appropriateness": "accuracy",
    "truthfulness": "truth",
    "humor": "pr_pred",
    "creativity": "pr_pred",
    "quality": "y_pred",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None, help="If not provided, will save in the same folder as input_dir")
    parser.add_argument("--concept", type=str, default="compliance", choices=CONCEPT_TO_METRIC_KEY.keys())
    parser.add_argument("--metric_key", type=str, default=None, help="If not provided, will use the default metric key for the concept")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args


def ppl_func(x, b):
    return b * np.abs(x)**2

def pne_func(x, a, c, b):
    return (np.tanh(a*x) + c) / np.exp(b * np.abs(x)**2)

def compute_pnes(xs, effects, ppls, ppl_cutoff=2e3):
    baseline_idx = np.argmin(np.abs(xs))
    base_value = effects[baseline_idx]
    base_ppl = ppls[baseline_idx]
    ys = (effects - base_value) / (ppls / base_ppl)

    ppl_ids = np.where(ppls < ppl_cutoff)
    ppl_xs = xs[ppl_ids]
    ppl_ys = ppls[ppl_ids] / base_ppl
    p0 = [0.05]
    fn = functools.partial(ppl_func)
    popt, pcov = curve_fit(fn, ppl_xs, np.log(ppl_ys), p0=p0, maxfev=20000)
    b = popt[0]

    p0 = [0, 0]
    fn = functools.partial(pne_func, b=b)
    popt, pcov = curve_fit(fn, xs, ys, p0=p0, maxfev=20000)
    a, c = popt

    residuals = ys - fn(xs, *popt)
    rss = np.sum(residuals**2)
    ss_tot = np.sum((ys - np.mean(ys))**2)
    if ss_tot > 0:
        r_squared = 1 - (rss / ss_tot)
        xs_ = np.linspace(xs.min(), xs.max(), 200)
        ys_ = pne_func(xs_, a=a, b=b, c=c)
        x_min = xs_[np.argmin(ys_)]
        x_max = xs_[np.argmax(ys_)]
        y_min = ys_.min()
        y_max = ys_.max()
    else:
        r_squared = float("nan")
        x_min = float("nan")
        x_max = float("nan")
        y_min = 0
        y_max = 0

    pnes = y_max - y_min

    min_idx = np.argmin(xs * (ppls < 10))
    max_idx = np.argmax(xs * (ppls < 10))
    alpha_min = xs[min_idx]
    alpha_max = xs[max_idx]
    p_low = ys[min_idx]
    p_high = ys[max_idx]

    return {
        "pne": pnes,
        "r_squared": r_squared,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "p_low": p_low,
        "p_high": p_high,
        "params": (a, c, b),
    }


def plot_pne(xs, effects, ppls, params=None, figsize=(10, 5), color="tab:blue", label=None):
    baseline_idx = np.argmin(np.abs(xs))
    base_value = effects[baseline_idx]
    base_ppl = ppls[baseline_idx]
    ys = (effects - base_value) / (ppls / base_ppl)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, marker="s", color=color)

    xs_ = np.linspace(xs.min(), xs.max(), 200)
    ys_ = pne_func(xs_, *params)

    ax.plot(xs_, ys_, color=color, label=label)

    ax.set_xlabel("Guidance Scale")
    ax.set_ylabel("PNE")

    if label:
        ax.legend()

    return fig, ax


def main(args):
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise ValueError(f"Invalid input folder: {input_path}")
    
    all_files = sorted(glob.glob(str(input_path / "**" / "metrics.json")))
    if len(all_files) == 0:
        raise ValueError(f"No metrics found in {input_path}")

    if args.metric_key is not None:
        metric_key = args.metric_key
    else:
        metric_key = CONCEPT_TO_METRIC_KEY[args.concept]
    
    if args.output_dir is None:
        output_path = input_path
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

    all_metrics = []
    for input_file in all_files:
        input_file = Path(input_file)

        with open(input_file, "r") as f:
            metrics = json.load(f)

        if "guidance_scale" not in metrics:
            print(f"Skipping {input_file}: no guidance_scale")
            continue

        if metric_key not in metrics:
            print(f"Skipping {input_file}: no {metric_key}")
            continue

        all_metrics.append(metrics)

    if len(all_metrics) == 0:
        raise ValueError(f"No valid metrics found in {input_path}")
    
    df = pd.DataFrame(all_metrics)

    all_top_ks = df["guidance_top_k"].unique()

    for top_k, dfi in df.groupby("guidance_top_k"):
        dfi = dfi.sort_values("guidance_scale")
        xs = dfi["guidance_scale"].values
        ys = dfi[metric_key].values
        ppls = dfi["ppl"].values

        metrics = {"top_k": top_k}
        metrics.update(compute_pnes(xs, ys, ppls))
        print(json.dumps(metrics, indent=4))

        metrics.update({"data": {"x": xs.tolist(), "y": ys.tolist(), "ppl": ppls.tolist()}})

        if len(all_top_ks) > 1:
            output_file = output_path / f"results_topk={top_k}.json"
        else:
            output_file = output_path / "results.json"

        fig, ax = plot_pne(xs, ys, ppls, params=metrics["params"], label="$r^2 = {:.3f}$".format(metrics["r_squared"]))
        fig.savefig(output_path / f"pne_topk={top_k}.png")

        if output_file.exists():
            with open(output_file, "r") as f:
                old_metrics = json.load(f)
            old_metrics.update(metrics)
            metrics = old_metrics

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
