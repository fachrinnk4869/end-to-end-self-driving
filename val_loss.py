import os
import pandas as pd
import csv


def calculate_loss(base_path):
    results = {}

    for root, dirs, files in os.walk(base_path):
        # Check for 'log' in root, followed by any subfolder, and then 'predict_expert' and 'ADVERSARIAL'
        path_parts = root.split(os.sep)
        if (
            len(path_parts) >= 1 and  # Ensure the path has enough levels
            # Check if 'log' is in the correct position
            path_parts[-2] == 'log' and
            'trainval_log.csv' in files  # Ensure the file exists
        ):
            # Extract the root name
            # Assuming root/*/log/*/predict_expert structure
            root_name = path_parts[1]

            # Construct the file path
            file_path = os.path.join(root, 'trainval_log.csv')
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                akhir = df.loc[df['best_model'] == 'BEST'].iloc[-1]
                # Calculate the mean of the 'elapsed_time' column
                results[root_name] = akhir
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Print the results
    output_data = []
    results = dict(sorted(results.items()))
    for root_name, akhir in results.items():
        output_data.append({
            "root_name": root_name,
            "val_loss": akhir["val_loss"],
            "val_ss_loss": akhir["val_ss_loss"],
            "val_wp_loss": akhir["val_wp_loss"],
            "val_str_loss": akhir["val_str_loss"],
            "val_thr_loss": akhir["val_thr_loss"],
            "val_brk_loss": akhir["val_brk_loss"],
            "val_redl_loss": akhir["val_redl_loss"],
            "val_stops_loss": akhir["val_stops_loss"],
            "train_loss": akhir["train_loss"],
            "train_ss_loss": akhir["train_ss_loss"],
            "train_wp_loss": akhir["train_wp_loss"],
            "train_str_loss": akhir["train_str_loss"],
            "train_thr_loss": akhir["train_thr_loss"],
            "train_brk_loss": akhir["train_brk_loss"],
            "train_redl_loss": akhir["train_redl_loss"],
            "train_stops_loss": akhir["train_stops_loss"],
        })
    # Save results to a CSV file
    output_csv = 'results_loss.csv'
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Replace '/path/to/root' with the actual path to the root directory
calculate_loss('./')
