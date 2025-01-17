#!/bin/bash

# Declare datasets and metrics to iterate over
declare -a arr_data=("SignLanguage" "MNIST" "Fashion_MNIST" "CIFAR10" "GTSRB" "CIFAR100")
declare -a arr_metrics=( "L2" )

# Base SLURM parameters
RAM_DEFAULT=16
PARTITION="gpu_partition"
OUTPUT_DIR="/home/itayab/output"

# Loop over datasets and metrics
for dataset in "${arr_data[@]}"; do
    # Set RAM dynamically based on the dataset
    if [ "$dataset" == "SignLanguage" ]; then
        RAM=10
    elif [ "$dataset" == "MNIST" ] || [ "$dataset" == "Fashion_MNIST" ]; then
        RAM=10
    elif [ "$dataset" == "CIFAR10" ] || [ "$dataset" == "GTSRB" ]; then
        RAM=10
    elif [ "$dataset" == "tiny_imagenet" ] || [ "$dataset" == "CIFAR100" ]; then
        RAM=10
    else
        RAM=$RAM_DEFAULT
    fi

    for metric in "${arr_metrics[@]}"; do
        # Generate a unique SLURM batch script for each combination
        job_file="combine_aggregate_${dataset}_${metric}.slurm"

        echo "Creating SLURM job file: $job_file"
        cat << EOF > $job_file
#!/bin/bash

#SBATCH --partition=${PARTITION}
#SBATCH --time=1-00:00:00
#SBATCH --job-name=${dataset}_${metric}
#SBATCH --output=${OUTPUT_DIR}/${dataset}_${metric}_%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --gpus=1
##SBATCH --exclude=cs-1080-02
#SBATCH --mem=${RAM}G
#SBATCH --mail-user=itayab@post.bgu.ac.il
#SBATCH --mail-type=NONE

echo "Job started at: \$(date)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"

module load anaconda
source activate itay_geometric

cd /cs/cs_groups/cliron_group/Calibrato/ || { echo "Failed to change directory"; exit 1; }
echo "Current working directory: \$(pwd)"

# Execute Python script for the given dataset and metric
python scripts/calculating_mean.py \\
    --dataset_name "${dataset}" \\
    --metric "${metric}"

echo "Job completed at: \$(date)"
EOF

        # Submit the job
        sbatch $job_file
        echo "Submitted job: $job_file"

        # Remove the job file after submission to keep the directory clean
        rm $job_file
        echo "Removed job file: $job_file"
    done
done
