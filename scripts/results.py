import os
import pandas as pd
import numpy as np
from scipy import stats

# Define base directory and model names
base_dir = "/cs/cs_groups/cliron_group/Calibrato/Fashion_MNIST"
random_states = range(200, 231)
models = ["cnn", "GB", "RF"]
distance_metric = "L2"  # Assuming L2 is the distance metric used

def compute_mean_ci(data):
    """Compute mean and 95% confidence interval."""
    if len(data) == 0:
        return None, None  # Return None if no data
    mean_value = np.mean(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean_value, scale=stats.sem(data))
    return mean_value, mean_value - ci[0], len(data)  # Return mean, ±CI value, and count

# Dictionary to store results
samples_per_sec = {}

# Iterate through models and random states
for model in models:
    all_samples = []
    
    for state in random_states:
        file_path = os.path.join(base_dir, str(state), model, distance_metric, "separation", "results.csv")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")  # Debugging
            continue
        
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}, columns: {df.columns}")  # Debugging

        # Ensure "Metric" column exists and locate "Separation" row
        if "Metric" in df.columns and "Samples per Second" in df.columns:
            separation_row = df[df["Metric"] == "Separation"]
            if separation_row.empty:
                print(f"No 'Separation' row found in {file_path}")  # Debugging
            else:
                sample_value = separation_row["Samples per Second"].values[0]
                print(f"Extracted {sample_value} from {file_path}")  # Debugging
                all_samples.append(sample_value)
        else:
            print(f"Missing required columns in {file_path}")  # Debugging
    
    if all_samples:
        mean_value, ci_value, count = compute_mean_ci(all_samples)
        if mean_value is not None:
            samples_per_sec[model] = (mean_value, ci_value, count)

# Print results, including the number of values used for calculation
if samples_per_sec:
    for model, (mean, ci, count) in samples_per_sec.items():
        print(f"{model.upper()}: Mean = {mean:.3f}, 95% CI = ±{ci:.3f}, Count = {count}")
else:
    print("No valid data found for any model.")
