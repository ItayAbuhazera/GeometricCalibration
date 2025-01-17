#!/bin/bash

# Declare the datasets and metrics to iterate over
declare -a arr_data=("SignLanguage" "Fashion_MNIST" "MNIST" "CIFAR10" "GTSRB")  # Add your desired datasets here
declare -a arr_model=("cnn" "RF" "GB")  # Add your desired models here
declare -a arr_metrics=("L2")  # Add your desired metrics here

# Loop through each dataset
for dataset in "${arr_data[@]}"; do
    # Set RAM allocation based on the dataset
    if [ "$dataset" == "SignLanguage" ]; then
        RAM="10"
    elif [ "$dataset" == "MNIST" ] || [ "$dataset" == "Fashion_MNIST" ]; then
        RAM="25"
    elif [ "$dataset" == "CIFAR10" ] || [ "$dataset" == "GTSRB" ]; then
        RAM="26"
    elif [ "$dataset" == "tiny_imagenet" ] || [ "$dataset" == "CIFAR100" ]; then
        RAM="64"
    fi

    # Loop through shuffle numbers
    for shuffle_num in {220..230}; do
        # Loop through metrics
        for metric in "${arr_metrics[@]}"; do
            for model in "${arr_model[@]}"; do
                # Create unique job name and file name
                File="sbatch_${dataset}_${model}_${shuffle_num}_${metric}.slurm"

                if [ ! -e "$File" ]; then
                    echo "Creating file $File"
                    touch $File
                fi

                cat << EOF > $File
#!/bin/bash

#SBATCH --partition=gpu_partition
#SBATCH --time=5-00:00:00
#SBATCH --job-name=${dataset}_${model}_${shuffle_num}_${metric}
#SBATCH --output=/home/itayab/output/${model}_${dataset}_${shuffle_num}_${metric}_%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --gpus=1
##SBATCH --exclude=cs-1080-02
#SBATCH --mem=64G
#SBATCH --mail-user=itayab@post.bgu.ac.il
#SBATCH --mail-type=NONE
#SBATCH --nodelist=cs-cpu128-[01-05],dt-cpu128-[01-02],ise-cpu128-[01-13,15]
#SBATCH --nodes=1



echo "Job started at: \$(date)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "SLURM_JOB_NODELIST: \$SLURM_JOB_NODELIST"

module load anaconda
source activate itay_geometric

cd /cs/cs_groups/cliron_group/Calibrato/ || { echo "Failed to change directory"; exit 1; }
echo "Current working directory: \$(pwd)"

# Execute with additional metric parameter
python scripts/main.py \\
    --dataset_name ${dataset} \\
    --random_state ${shuffle_num} \\
    --model_type ${model} \\
    --metric ${metric} \\
    #--transformed  # Add this flag if transformed dataset is needed

echo "Job completed at: \$(date)"
EOF

            sbatch $File
            rm $File
            done
        done
    done
done
