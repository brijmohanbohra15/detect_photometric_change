import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

# Imports after sys.path modification for local modules
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import median_abs_deviation 
from src.load_data import load_data
from utility import plot_and_save_outlier_analysis, plot_contextual_outliers_no_fill

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Phase binning
def add_phase_bin(df, bin_width=10):
    """
    Adds a phase bin column to the dataframe.
    """
    df = df.copy()
    df['phase_bin'] = (df['equatorial_phase'] / bin_width).round() * bin_width
    return df


# Glint identification
def flag_glint(df, glint_threshold=5):
    """
    Flags observations near zero SEPA as potential glints.
    """
    df = df.copy()
    df['is_glint'] = df['equatorial_phase'].abs() <= glint_threshold
    return df


# contextual statistic per bin
def compute_phase_stats(df):
    """
    Computes robust baseline per phase bin.
    """
    stats = (
        df.groupby('phase_bin')['magnitude']
        .agg(
            median_mag='median',
            mad_mag=lambda x: median_abs_deviation(x, scale=1)
        )
        .reset_index()
    )
    return stats


# Contextual Outlier Detection
def detect_contextual_outliers(df, phase_stats, z_thresh=3):
    """
    Detects contextual outliers using robust z-score.
    """
    df = df.copy()
    df = df.merge(phase_stats, on='phase_bin', how='left')

    # Robust z-score
    df['robust_z'] = (
        df['magnitude'] - df['median_mag']
    ) / (1.4826 * df['mad_mag'])

    df['is_contextual_outlier'] = df['robust_z'].abs() > z_thresh

    return df


# Phase-bin median imputation function
def fill_contextual_outliers(
    df,
    mag_col='magnitude',
    phase_bin_col='phase_bin',
    outlier_col='is_contextual_outlier',
    glint_col='is_glint'
):
    df = df.copy()

    # Compute robust phase-bin medians using NON-outliers
    phase_median = (
        df[~df[outlier_col] & ~df[glint_col]]
        .groupby(phase_bin_col)[mag_col]
        .median()
    )

    # Fill values
    def replace_mag(row):
        if row[outlier_col] or row[glint_col]:
            return phase_median.get(row[phase_bin_col], row[mag_col])
        return row[mag_col]

    df['magnitude_filled'] = df.apply(replace_mag, axis=1)

    return df


# Cleaning Pipeline
def contextual_outlier_cleaning(df, bin_width=5, z_thresh=3):
    """
    Removes contextual outliers and handles glinting.
    """
    df = df.copy()

    # 1. Define context
    df = add_phase_bin(df, bin_width)

    # 2. Flag glints
    df = flag_glint(df)

    # 3. Compute stats using NON-glint points only
    phase_stats = compute_phase_stats(df[~df['is_glint']])

    # 4. Detect contextual outliers
    df = detect_contextual_outliers(df, phase_stats, z_thresh)

    # 5. Remove:
    #    - contextual outliers
    #    - glint-dominated points
    cleaned_df = df[
        (~df['is_contextual_outlier']) &
        (~df['is_glint'])
    ]

    return cleaned_df, df


def process_data_with_outlier_detection(
    df_context,
    Reference_Start,
    Reference_End,
    Current_Start,
    Current_End,
    bin_width=10,
    z_thresh=3
):
    """
    Outlier detection and imputation of it for outlier points for reference and current periods for a specific NORAD ID.

    Returns
    -------
    tuple
        (df_ref_annotated, df_cur_annotated, df_ref_not_filled, df_cur_not_filled) - Cleaned dataframes or None
        if empty
        df_ref_not_filled - Dataframe with outlier points for reference period
        df_cur_not_filled - Dataframe with outlier points for current period
    """
    # Split
    df_ref = df_context[
        (df_context['timestamp'] >= Reference_Start) &
        (df_context['timestamp'] <= Reference_End)
    ]

    df_cur = df_context[
        (df_context['timestamp'] >= Current_Start) &
        (df_context['timestamp'] <= Current_End)
    ]

    # Skip completely empty cases
    if df_ref.empty and df_cur.empty:
        return None, None

    # Clean (only if not empty)
    if not df_ref.empty:
        _, df_ref_annotated = contextual_outlier_cleaning(
            df_ref, bin_width=bin_width, z_thresh=z_thresh
        )
        df_ref_not_filled = df_ref_annotated.copy()
        df_ref_annotated = fill_contextual_outliers(df_ref_annotated)

    else:
        df_ref_not_filled = None
        df_ref_annotated = None

    if not df_cur.empty:
        _, df_cur_annotated = contextual_outlier_cleaning(
            df_cur, bin_width=bin_width, z_thresh=z_thresh
        )
        df_cur_not_filled = df_cur_annotated.copy()
        df_cur_annotated = fill_contextual_outliers(df_cur_annotated)
    else:
        df_cur_not_filled = None
        df_cur_annotated = None

    return df_ref_annotated, df_cur_annotated, df_ref_not_filled, df_cur_not_filled


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

    for norad_id in df_all['norad_id'].unique():
        df_context = df_all[df_all['norad_id'] == norad_id]

        df_ref_annotated, df_cur_annotated, df_ref_not_filled, df_cur_not_filled = process_data_with_outlier_detection(
            df_context=df_context,
            Reference_Start=Reference_Start,
            Reference_End=Reference_End,
            Current_Start=Current_Start,
            Current_End=Current_End,
        )
        plot_contextual_outliers_no_fill(
            norad_id=norad_id,
            df_ref_annotated=df_ref_not_filled,
            df_cur_annotated=df_cur_not_filled,
            output_path=output_path
        )

        plot_and_save_outlier_analysis(
            norad_id=norad_id,
            df_ref_annotated=df_ref_annotated,
            df_cur_annotated=df_cur_annotated,
            output_path=output_path
        )
