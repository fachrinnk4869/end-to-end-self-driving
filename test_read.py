import os
import pandas as pd


def calculate_mean_elapsed_time(base_path):
    results = {}

    for root, dirs, files in os.walk(base_path):
        # Check for 'log' in root, followed by any subfolder, and then 'predict_expert' and 'ADVERSARIAL'
        path_parts = root.split(os.sep)
        if (
            len(path_parts) >= 4 and  # Ensure the path has enough levels
            # Check if 'log' is in the correct position
            path_parts[-4] == 'log' and
            # Check for 'predict_expert' in the last folder
            'ADVERSARIAL' in path_parts[-1] and
            # Check for 'ADVERSARIAL' in the second-to-last folder
            'predict_expert' in path_parts[-2] and
            'test_log_ClearNoon-fix.csv' in files  # Ensure the file exists
        ):
            print(path_parts)
            # Extract the root name
            # Assuming root/*/log/*/predict_expert structure
            root_name = path_parts[1]

            # Construct the file path
            file_path = os.path.join(root, 'test_log_ClearNoon-fix.csv')

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Store the result
                results[root_name] = df
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Print the results
    output_data = []
    results = dict(sorted(results.items()))
    for root_name, df in results.items():
        output_data.append(
            {"root_name": root_name,
             "test_metric": df['test_metric'].mean(),
             "test_ss_metric": df['test_ss_metric'].mean(),
             "test_wp_metric": df['test_wp_metric'].mean(),
             "test_str_metric": df['test_str_metric'].mean(),
             "test_thr_metric": df['test_thr_metric'].mean(),
             "test_brk_metric": df['test_brk_metric'].mean(),
             "test_redl_metric": df['test_redl_metric'].mean(),
             "test_stops_metric": df['test_stops_metric'].mean(),
             })
    # Save results to a CSV file
    output_csv = 'results_test.csv'
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Replace '/path/to/root' with the actual path to the root directory
calculate_mean_elapsed_time('./')
