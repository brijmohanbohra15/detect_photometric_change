import json
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

import pandas as pd  # noqa: E402

from src.load_data import load_data  # noqa: E402
from src.outlier_handling_with_glint import (  # noqa: E402
    process_data_with_outlier_detection
)
from src.anomaly_detection import (  # noqa: E402
    detect_collective_anomalies,
    deviation_score,
    add_phase_bin,
    build_anomaly_records
)
from utility import plot_collective_anomalies  # noqa: E402


class PhotometricAnomalyDetector:
    """
    A class for detecting photometric anomalies in satellite data.

    This class processes satellite photometric data, detects outliers,
    and identifies collective anomalies between reference and current periods.
    """

    def __init__(self, config_json_path):
        """
        Initialize the PhotometricAnomalyDetector with configuration from JSON.

        Parameters
        ----------
        config_json_path : str
            Path to the JSON configuration file containing:
            - data: {raw_path, output_path}
            - dates: {Reference_Start, Reference_End, Current_Start,
              Current_End}
            - bin_width (optional)
        """
        self.config_json_path = config_json_path
        self.config = self._load_config()

    def _load_config(self):
        """
        Load configuration from JSON file.

        Returns
        -------
        dict
            Configuration dictionary
        """
        with open(self.config_json_path, 'r') as f:
            config = json.load(f)

        # Set default bin_width if not provided
        if 'bin_width' not in config:
            config['bin_width'] = 10

        return config

    def run(self):
        """
        Run the photometric anomaly detection pipeline.

        This method:
        1. Loads data from the configured raw_path
        2. Processes each NORAD ID with outlier detection
        3. Detects collective anomalies
        4. Generates and saves plots

        Returns
        -------
        None
            Saves plots and processed data to output_path
        """
        # Convert date strings to datetime
        reference_start = pd.to_datetime(
            self.config['dates']['Reference_Start'], utc=True
        )
        reference_end = pd.to_datetime(
            self.config['dates']['Reference_End'], utc=True
        )
        current_start = pd.to_datetime(
            self.config['dates']['Current_Start'], utc=True
        )
        current_end = pd.to_datetime(
            self.config['dates']['Current_End'], utc=True
        )
        

        # Load data
        df_all = load_data(self.config['data']['raw_path'])
        output_path = self.config['data']['output_path']
        bin_width = self.config['bin_width']
        # create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # create empty json file for collective anomalies 
        with open(os.path.join(output_path, "collective_anomalies.json"), "w") as f:
            json.dump([], f)
        # Process each NORAD ID
        for norad_id in df_all['norad_id'].unique():
            df_context = df_all[df_all['norad_id'] == norad_id]

            # Process data with outlier detection
            # output reference and current dataframe for each norad_id 
            df_ref_annotated, df_cur_annotated, _, _ = (
                process_data_with_outlier_detection(
                    df_context=df_context,
                    Reference_Start=reference_start,
                    Reference_End=reference_end,
                    Current_Start=current_start,
                    Current_End=current_end,
                    bin_width=bin_width
                )
            )

            # Skip if both datasets are empty
            if df_ref_annotated is None and df_cur_annotated is None:
                continue

            # Detect collective anomalies
            try:

                df_collective = detect_collective_anomalies(
                    df_ref_annotated, df_cur_annotated
                )
                df_collective['deviation_score'] = df_collective.apply(
                    deviation_score, axis=1
                )
                anomaly_json = build_anomaly_records(
                    df_collective=df_collective,
                    df_cur=df_cur_annotated,
                    norad_id=norad_id,
                    bin_width=bin_width
                )
                # append to collective_anomalies.json
                with open(os.path.join(output_path, "collective_anomalies.json"), "a") as f:
                    json.dump(anomaly_json, f, indent=4, ensure_ascii=False)

                # Plot and save collective anomalies
                plot_collective_anomalies(
                    df_ref_annotated,
                    df_cur_annotated,
                    df_collective,
                    norad_id,
                    output_path,
                    phase_bin_width=bin_width
                )
            except Exception as e:
                print(f"Error processing NORAD ID {norad_id}: {e}")
                continue


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description='Run photometric anomaly detection'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to JSON configuration file'
    )

    args = parser.parse_args()

    detector = PhotometricAnomalyDetector(args.config)
    detector.run()
