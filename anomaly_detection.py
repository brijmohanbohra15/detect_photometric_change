import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

import yaml
from src.load_data import load_data
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
from src.outlier_handling_with_glint import process_data_with_outlier_detection
from utility import plot_collective_anomalies
import json


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def add_phase_bin(df, width=5):
    df = df.copy()
    df['phase_bin'] = (df['equatorial_phase'] // width) * width
    return df

def detect_collective_anomalies(
    df_ref,
    df_cur,
    mag_col='magnitude_filled',
    phase_bin_col='phase_bin',
    min_samples=5,
    ks_alpha=0.15,
    effect_thresh=0.25,
    min_mean_shift=0.1
):
    results = []

    common_bins = set(df_ref[phase_bin_col]).intersection(
        set(df_cur[phase_bin_col])
    )

    for pb in sorted(common_bins):
        ref_vals = df_ref[df_ref[phase_bin_col] == pb][mag_col].dropna()
        cur_vals = df_cur[df_cur[phase_bin_col] == pb][mag_col].dropna()

        if len(ref_vals) < min_samples or len(cur_vals) < min_samples:
            continue

        # Robust central tendency
        ref_mean = ref_vals.mean()
        cur_mean = cur_vals.mean()
        delta_mu = cur_mean - ref_mean

        # KS test
        ks_stat, ks_p = ks_2samp(ref_vals, cur_vals)

        # Robust pooled spread (protects against small variance)
        pooled_std = np.sqrt(
            (ref_vals.var(ddof=1) + cur_vals.var(ddof=1)) / 2
        )
        cohens_d = delta_mu / pooled_std if pooled_std > 1e-6 else 0

        is_anomaly = (
            ks_p < ks_alpha and
            abs(cohens_d) > effect_thresh and
            abs(delta_mu) > min_mean_shift
        )

        results.append({
            'phase_bin': pb,
            'ref_mean': ref_mean,
            'cur_mean': cur_mean,
            'delta_mean': delta_mu,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'cohens_d': cohens_d,
            'is_collective_anomaly': is_anomaly
        })

    return pd.DataFrame(results)


def deviation_score(row):
    # Lower p-value → higher deviation
    p_component = 1 - min(row['ks_pvalue'], 1)

    # Normalize effect size
    d_component = min(abs(row['cohens_d']) / 2.0, 1)

    return 0.5 * p_component + 0.5 * d_component

def phase_bin_bounds(phase_bin, bin_width):
    return phase_bin, phase_bin + bin_width

def build_anomaly_records(
    df_collective,
    df_cur,
    norad_id,
    phase_bin_col='phase_bin',
    time_col='timestamp',
    phase_col='equatorial_phase',
    bin_width=10
):
    anomalies = []

    df_anom_bins = df_collective[
        df_collective['is_collective_anomaly']
    ]

    for _, row in df_anom_bins.iterrows():
        pb = row[phase_bin_col]
        pmin, pmax = phase_bin_bounds(pb, bin_width)

        df_bin_cur = df_cur[
            (df_cur[phase_col] >= pmin) &
            (df_cur[phase_col] <  pmax)
        ]

        if df_bin_cur.empty:
            continue

        anomalies.append({
            "norad_id": int(norad_id),
            "equatorial_phase": [
                float(df_bin_cur[phase_col].min()),
                float(df_bin_cur[phase_col].max())
            ],
            "timestamp": [
                df_bin_cur[time_col].min().strftime("%Y-%m-%d %H:%M:%S"),
                df_bin_cur[time_col].max().strftime("%Y-%m-%d %H:%M:%S")
            ],
            "deviation_score": float(row['deviation_score']),
            # "delta_mean_mag": float(row['delta_mean']),
            # "ks_pvalue": float(row['ks_pvalue']),
            # "cohens_d": float(row['cohens_d'])
        })

    return anomalies


if __name__ == "__main__":
    # Load date strings from config and convert to datetime
    Reference_Start = pd.to_datetime(
        config['dates']['Reference_Start'], utc=True
    )
    Reference_End = pd.to_datetime(
        config['dates']['Reference_End'], utc=True
    )
    Current_Start = pd.to_datetime(
        config['dates']['Current_Start'], utc=True
    )
    Current_End = pd.to_datetime(
        config['dates']['Current_End'], utc=True
    )

    df_all = load_data(config['data']['raw_path'])
    output_path = config['data']['output_path']
    os.makedirs(output_path, exist_ok=True)
    # create empty json file for collective anomalies 
    with open(os.path.join(output_path, "collective_anomalies.json"), "w") as f:
        json.dump([], f)
    for norad_id in df_all['norad_id'].unique():
        df_context = df_all[df_all['norad_id'] == norad_id]

        df_ref_annotated, df_cur_annotated, _, _ = process_data_with_outlier_detection(
            df_context=df_context,
            Reference_Start=Reference_Start,
            Reference_End=Reference_End,
            Current_Start=Current_Start,
            Current_End=Current_End,
        )

        try:
            df_collective = detect_collective_anomalies(df_ref_annotated, df_cur_annotated)
            df_collective['deviation_score'] = df_collective.apply(deviation_score, axis=1)
            os.makedirs(output_path, exist_ok=True)

            anomaly_json = build_anomaly_records(
                df_collective=df_collective,
                df_cur=df_cur_annotated,
                norad_id=norad_id,
                bin_width=10)
            # append to collective_anomalies.json
            with open(os.path.join(output_path, "collective_anomalies.json"), "a") as f:
                json.dump(anomaly_json, f, indent=4, ensure_ascii=False)
            
            # Plot collective anomalies for this NORAD ID and save to output_path
            plot_collective_anomalies(df_ref_annotated, df_cur_annotated, df_collective, norad_id, output_path)
        except Exception as e:
            print(f"Error for NORAD ID {norad_id}: {e}")
            continue
