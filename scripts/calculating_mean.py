#!/usr/bin/env python
# coding: utf-8

# 1. Import Required Libraries
# Start by importing the necessary libraries.

# In[1]:


import os
import fireducks.pandas as pd
import kagglehub


# 2. Define Paths
# Set up the base path where your data is stored and the output paths for combined and aggregated results.

# In[2]:


# Paths
base_path = "/cs/cs_groups/cliron_group/Calibrato/"
datasets = ["CIFAR100"]  # Add other datasets if needed
models = ["GB", "RF"]  # Models to include
distance_metrics = [ "L2"]  # Distance metrics
combined_files = {
    # "transformed": "/cs/cs_groups/cliron_group/Calibrato/Combined/combined_results_transformed_{dataset}_{model}_{metric}.csv",
    "non_transformed": "/cs/cs_groups/cliron_group/Calibrato/Combined/combined_results_non_transformed_{dataset}_{model}_{metric}.csv"
}
aggregate_files = {
    # "transformed": "/cs/cs_groups/cliron_group/Calibrato/Aggregate/{dataset}/{model}/aggregate_results_transformed_{metric}.csv",
    "non_transformed": "/cs/cs_groups/cliron_group/Calibrato/Aggregate/{dataset}/{model}/aggregate_results_non_transformed_{metric}.csv"
}


# 3. Combine Results
# Create a cell to iterate through your folders, extract metadata, and combine all CSV files.

# In[3]:


def combine_results(base_path, dataset_filter, model_filter, metric_filter, transformed_filter):
    """
    Combines results from multiple CSV files into a single DataFrame, ensuring the 'Samples per Second' column exists.

    Parameters:
        base_path (str): The base path to search for files.
        dataset_filter (str): Dataset to filter (e.g., 'MNIST', 'CIFAR100').
        model_filter (str): Model to filter (e.g., 'GB', 'RF', 'cnn').
        metric_filter (str): Distance metric to filter (e.g., 'L1', 'L2').
        transformed_filter (bool): Whether to filter for transformed paths.

    Returns:
        pd.DataFrame: Combined DataFrame of results.
    """
    combined_data = []

    for root, _, files in os.walk(base_path):
        # Filter directories based on dataset, model, and metric
        if dataset_filter not in root or model_filter not in root or metric_filter not in root:
            continue

        # Check for transformed or non-transformed filter
        is_transformed = "transformed" in root
        if is_transformed != transformed_filter:
            continue

        for file in files:
            if file.endswith("all_results.csv"):
                # Extract metadata
                path_parts = root.split('/')
                try:
                    random_state = path_parts[-5 if is_transformed else -4]
                except IndexError:
                    print(f"Unexpected path structure: {root}. Skipping.")
                    continue

                # Attempt to read the CSV file
                file_path = os.path.join(root, file)
                try:
                    data = pd.read_csv(file_path)
                    if data.empty:
                        print(f"Empty data in file: {file_path}. Skipping.")
                        continue
                except pd.errors.EmptyDataError:
                    print(f"File is empty or unreadable: {file_path}. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}. Skipping.")
                    continue

                # Ensure 'Samples per Second' column exists
                if 'Samples per Second' not in data.columns:
                    print(f"'Samples per Second' column missing in file: {file_path}. Skipping.")
                    continue

                # Add metadata columns
                data['Dataset'] = dataset_filter
                data['Random State'] = int(random_state)
                data['Model'] = model_filter
                data['Distance Metric'] = metric_filter
                data['Transformed'] = is_transformed

                combined_data.append(data)

    # Combine all collected data
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    else:
        print(f"No valid data found for dataset: {dataset_filter}, model: {model_filter}, metric: {metric_filter}, transformed: {transformed_filter}")
        return pd.DataFrame()
    
# Ensure the combined_data folder exists
output_dir = os.path.join(base_path, "combined_data_")
os.makedirs(output_dir, exist_ok=True)


# In[4]:


import scipy.stats as stats

# Function to aggregate results
def aggregate_results(file_path, group_by_columns, aggregate_columns):
    """
    Aggregates results by calculating mean, standard deviation, count, and 95% confidence intervals.

    Parameters:
        file_path (str): Path to the combined CSV file.
        group_by_columns (list): Columns to group by (e.g., ['Metric', 'Dataset', 'Distance Metric']).
        aggregate_columns (list): Columns to aggregate (e.g., ['ECE', 'MCE', 'Brier Score']).

    Returns:
        pd.DataFrame: Aggregated DataFrame with mean, standard deviation, count, and confidence intervals.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Ensure Distance Metric is part of grouping columns
    if 'Distance Metric' not in group_by_columns:
        group_by_columns.append('Distance Metric')
    
    # Aggregate results
    aggregated = (
        data.groupby(group_by_columns)[aggregate_columns]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    
    # Flatten the multi-level columns
    aggregated.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in aggregated.columns]

    # Calculate 95% confidence intervals
    for col in aggregate_columns:
        mean_col = f"{col} mean"
        std_col = f"{col} std"
        count_col = f"{col} count"
        ci_lower_col = f"{col} 95% CI Lower"
        ci_upper_col = f"{col} 95% CI Upper"

        # Add confidence interval columns
        aggregated[ci_lower_col] = aggregated[mean_col] - stats.t.ppf(0.975, aggregated[count_col] - 1) * (aggregated[std_col] / (aggregated[count_col] ** 0.5))
        aggregated[ci_upper_col] = aggregated[mean_col] + stats.t.ppf(0.975, aggregated[count_col] - 1) * (aggregated[std_col] / (aggregated[count_col] ** 0.5))

    return aggregated

# Define columns to group by and aggregate
group_by_columns = ['Metric', 'Dataset', 'Distance Metric', 'Model', 'Transformed']
aggregate_columns = ['ECE', 'Samples per Second', 'Calibration Time (s)']

# Aggregate results for each dataset, model, and metric
output_dir = "aggregated_results_"  # Directory to save the aggregated tables
os.makedirs(output_dir, exist_ok=True)



# In[ ]:


from tqdm import tqdm

def process_and_aggregate_results(base_path, datasets, models, distance_metrics):
    """
    Combines results from multiple CSV files, aggregates them, and saves both combined and aggregated results.

    Parameters:
        base_path (str): The base path to search for files.
        datasets (list): List of datasets to process (e.g., ['MNIST', 'CIFAR100']).
        models (list): List of models to process (e.g., ['GB', 'RF', 'cnn']).
        distance_metrics (list): List of distance metrics (e.g., ['L1', 'L2']).
    """
    # Iterate over datasets, models, and distance metrics
    for dataset in tqdm(datasets, desc="Processing Datasets"):
        for model in tqdm(models, desc=f"Processing Models for {dataset}", leave=False):
            for metric in tqdm(distance_metrics, desc=f"Processing Metrics for {model}", leave=False):
                # Define file paths dynamically for each combination
                aggregate_files = {
                    "transformed": f"/cs/cs_groups/cliron_group/Calibrato/Aggregate/{dataset}/{model}/aggregate_results_transformed_{metric}.csv",
                    "non_transformed": f"/cs/cs_groups/cliron_group/Calibrato/Aggregate/{dataset}/{model}/aggregate_results_non_transformed_{metric}.csv"
                }
                combined_files = {
                    "transformed": f"/cs/cs_groups/cliron_group/Calibrato/Combined/{dataset}/{model}/combined_results_transformed_{metric}.csv",
                    "non_transformed": f"/cs/cs_groups/cliron_group/Calibrato/Combined/{dataset}/{model}/combined_results_non_transformed_{metric}.csv"
                }

                # Ensure directories exist
                os.makedirs(os.path.dirname(aggregate_files["transformed"]), exist_ok=True)
                os.makedirs(os.path.dirname(combined_files["transformed"]), exist_ok=True)

                # Transformed results
                combined_df_transformed = combine_results(base_path, dataset, model, metric, transformed_filter=True)
                if not combined_df_transformed.empty:
                    # Save combined results
                    transformed_file = combined_files["transformed"]
                    combined_df_transformed.to_csv(transformed_file, index=False)
                    tqdm.write(f"Transformed results for {dataset}, {model}, {metric} saved to {transformed_file}")

                    # Aggregate results
                    aggregated_transformed = aggregate_results(
                        transformed_file,
                        group_by_columns=['Metric', 'Dataset', 'Distance Metric', 'Model', 'Transformed'],
                        aggregate_columns=['ECE', 'Samples per Second', 'Calibration Time (s)']
                    )
                    aggregated_file = aggregate_files["transformed"]
                    aggregated_transformed.to_csv(aggregated_file, index=False)
                    tqdm.write(f"Aggregated Transformed Results saved to {aggregated_file}")
                else:
                    tqdm.write(f"No transformed data for {dataset}, {model}, {metric}.")

                # Non-transformed results
                combined_df_non_transformed = combine_results(base_path, dataset, model, metric, transformed_filter=False)
                if not combined_df_non_transformed.empty:
                    # Save combined results
                    non_transformed_file = combined_files["non_transformed"]
                    combined_df_non_transformed.to_csv(non_transformed_file, index=False)
                    tqdm.write(f"Non-transformed results for {dataset}, {model}, {metric} saved to {non_transformed_file}")

                    # Aggregate results
                    aggregated_non_transformed = aggregate_results(
                        non_transformed_file,
                        group_by_columns=['Metric', 'Dataset', 'Distance Metric', 'Model', 'Transformed'],
                        aggregate_columns=['ECE', 'Samples per Second', 'Calibration Time (s)']
                    )
                    aggregated_file = aggregate_files["non_transformed"]
                    aggregated_non_transformed.to_csv(aggregated_file, index=False)
                    tqdm.write(f"Aggregated Non-Transformed Results saved to {aggregated_file}")
                else:
                    tqdm.write(f"No non-transformed data for {dataset}, {model}, {metric}.")




import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine and aggregate results for a specific dataset and metric.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--metric", type=str, required=True, help="Metric to use for processing (e.g., L1, L2).")

    args = parser.parse_args()

    process_and_aggregate_results(
        base_path="/cs/cs_groups/cliron_group/Calibrato",
        datasets=[args.dataset_name],
        models=["cnn", "RF", "GB", "pretrained_efficientnet"],
        distance_metrics=[args.metric]
    )

if __name__ == "__main__":
    main()
