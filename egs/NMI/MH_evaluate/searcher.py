# -*- coding: utf-8 -*-
"""
Created on Wed July 08 11:00:00 2026
"""
import numpy as np
import torch
import os
import pandas as pd


def inspect_model_quantities(model, name_filter=None):
    """
    Automatically prints every numerical quantity stored inside the
    MAG2305 model.
    """
    print("\n" + "=" * 75)
    print(f"{'DISCOVERING AVAILABLE MODEL QUANTITIES':^75}")
    print("=" * 75)

    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        if name_filter and name_filter.lower() not in name.lower():
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        
        # Skip callable methods (functions) to focus only on data fields
        if callable(value):
            continue

        # Type mapping
        if isinstance(value, torch.Tensor):
            type_str = f"Tensor({value.device})"
            info_str = f"shape={tuple(value.shape)}"
        elif isinstance(value, np.ndarray):
            type_str = "ndarray"
            info_str = f"shape={value.shape}"
        elif np.isscalar(value):
            type_str = "scalar"
            # Format floats so scientific values read cleanly
            info_str = f"value={value:.4g}" if isinstance(value, (float, np.floating)) else f"value={value}"
        else:
            type_str = type(value).__name__
            val_str = str(value)
            info_str = f"value={val_str[:40]}..." if len(val_str) > 40 else f"value={val_str}"

        print(f"{'name: ' + name:32s} | type: {type_str:15s} | {info_str}")
    print("=" * 75)

def extract_candidate_predictors(model):
    """
    Automatically extracts every scalar predictor from the model.

    Returns
    -------
    predictors : dict
    """
    predictors = {}

    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue

        # Skip callable methods (functions) to focus only on data fields
        if callable(value):
            continue

        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().numpy()
            if x.size == 0:
                continue
            if x.ndim >= 2:
                predictors[f"{name}_mean"] = np.mean(x)
                predictors[f"{name}_std"] = np.std(x)
                predictors[f"{name}_max"] = np.max(x)
                predictors[f"{name}_min"] = np.min(x)
                predictors[f"{name}_rms"] = np.sqrt(np.mean(x**2))
            else:
                predictors[name] = np.mean(x)
        elif np.isscalar(value):
            predictors[name] = value

    return predictors


def analyze_transition_peaks(time, trajectory_error, parameter, parameter_name="Parameter", peak_threshold=None, window=20):
    """
    Analyze whether a parameter predicts trajectory error peaks.
    """
    error = np.asarray(trajectory_error)
    parameter = np.asarray(parameter)

    if peak_threshold is None:
        peak_threshold = error.mean() + error.std()

    peak_indices = np.where(error > peak_threshold)[0]

    if len(peak_indices) == 0:
        print("No peaks detected.")
        return []

    # Split into separate peaks
    groups = []
    current = [peak_indices[0]]

    for idx in peak_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    results = []

    print("=" * 60)
    print(f"Peak analysis for {parameter_name}")
    print("=" * 60)

    for i, group in enumerate(groups):
        peak = group[np.argmax(error[group])]
        start = max(0, peak - window)
        end = min(len(error), peak + window + 1)

        local_error = error[start:end]
        local_parameter = parameter[start:end]
        corr = np.corrcoef(local_error, local_parameter)[0, 1]

        results.append({"peak_number": i + 1, 
                        "peak_frame": peak,
                        "peak_time": time[peak],
                        "peak_error": error[peak],
                        "correlation": corr,
                        "parameter_mean": np.mean(local_parameter),
                        "parameter_max": np.max(local_parameter)})
        print(f"\nPeak {i+1}")
        print(f"Frame: {peak}")
        print(f"Time : {time[peak]:.4f}")
        print(f"Error: {error[peak]:.6f}")
        print(f"Local correlation: {corr:.3f}")
        print(f"Parameter mean: {np.mean(local_parameter):.6f}")
        print(f"Parameter max : {np.max(local_parameter):.6f}")

    return results


def compare_transition_predictors(time, trajectory_error, predictors, peak_threshold=None, 
                                  window=20, csv_path=None, verbose=True, parameter_names=None):
    """
    Compare multiple physical quantities as predictors of trajectory-error peaks.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per (parameter, peak).
    summary_df : pandas.DataFrame
        Average absolute correlation ranking.
    """

    error = np.asarray(trajectory_error)

    if parameter_names is not None:
        predictors = {k: predictors[k] for k in parameter_names if k in predictors}

    if peak_threshold is None:
        peak_threshold = error.mean() + error.std()

    peak_indices = np.where(error > peak_threshold)[0]

    if len(peak_indices) == 0:
        raise ValueError("No trajectory-error peaks detected.")

    groups = []
    current = [peak_indices[0]]

    for idx in peak_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    peak_centers = []

    for group in groups:
        peak = group[np.argmax(error[group])]
        peak_centers.append(peak)

    rows = []

    for parameter_name, values in predictors.items():
        values = np.asarray(values)
        for peak_number, peak in enumerate(peak_centers, start=1):
            start = max(0, peak - window)
            end = min(len(error), peak + window + 1)

            local_error = error[start:end]
            local_values = values[start:end]
            corr = np.corrcoef(local_error, local_values)[0, 1]

            if np.isnan(corr):
                corr = 0.0

            rows.append({"Parameter": parameter_name,
                         "Peak": peak_number,
                         "Frame": int(peak),
                         "Time": float(time[peak]),
                         "Peak Error": float(error[peak]),
                         "Correlation": float(corr),
                         "Abs Correlation": abs(float(corr)),
                         "Mean Parameter": float(np.mean(local_values)),
                         "Max Parameter": float(np.max(local_values)),
                         "Std Parameter": float(np.std(local_values))})

    results_df = pd.DataFrame(rows)

    summary_df = (results_df.groupby("Parameter")["Abs Correlation"].mean().sort_values(ascending=False).reset_index()
                  .rename(columns={"Abs Correlation": "Average |Correlation|"}))

    if verbose:
        print("\n")
        print("=" * 80)
        print("TRANSITION PREDICTOR ANALYSIS")
        print("=" * 80)

        for peak in sorted(results_df["Peak"].unique()):
            table = (results_df[results_df["Peak"] == peak].sort_values("Abs Correlation", ascending=False))

            print("\n")
            print("-" * 80)
            print(f"Peak {peak}")
            print("-" * 80)
            print(table[["Parameter", "Correlation", "Mean Parameter", "Max Parameter"]].to_string(index=False))

        print("\n")
        print("=" * 80)
        print("OVERALL RANKING")
        print("=" * 80)
        print(summary_df.to_string(index=False))

    if csv_path is not None:
        os.makedirs(csv_path, exist_ok=True)
        results_df.to_csv(os.path.join(csv_path, "transition_predictor_results.csv"),index=False)
        summary_df.to_csv(os.path.join(csv_path, "transition_predictor_summary.csv"),index=False)

        if verbose:
            print("\nCSV files written to:")
            print(csv_path)

    return results_df, summary_df