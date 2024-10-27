#!/bin/bash
#SBATCH --job-name=master_job              # Job name
#SBATCH --output=logs/master_job_%j.out    # Standard output log file
#SBATCH --error=logs/master_job_%j.err     # Standard error log file
#SBATCH --time=48:00:00                    # Total time limit for all jobs combined
#SBATCH --cpus-per-task=4                  # Adjust based on your needs
#SBATCH --mem=16G                          # Adjust memory per task
#SBATCH --partition=kern,preempt,kerngpu   # Partitions to submit the job to
#SBATCH --account=kernlab                  # Account to use
#SBATCH --requeue                          # Requeue on preemption

LOG_DIR="/projects/kernlab/akapoor/Demographic_Inference/logs"
ERROR_LOG="${LOG_DIR}/error_log.txt"
SUCCESS_LOG="${LOG_DIR}/success_log.txt"

# Ensure log directory exists
mkdir -p $LOG_DIR

# Function to unlock Snakemake if a lock exists
unlock_snakemake() {
    if [ -e "/projects/kernlab/akapoor/Demographic_Inference/.snakemake/locks" ]; then
        echo "Unlocking Snakemake directory..."
        snakemake --unlock
        if [ $? -ne 0 ]; then
            echo "Error unlocking Snakemake." >> $ERROR_LOG
            exit 1
        fi
    fi
}

# Function to run Snakemake with unlocking and logging
run_snakemake() {
    local snakemake_command=$1
    local description=$2

    # Unlock Snakemake before running
    unlock_snakemake

    # Run Snakemake command
    echo "Running Snakemake: $description"
    eval $snakemake_command

    # Check for errors and log the outcome
    if [ $? -ne 0 ]; then
        echo "Error: Snakemake failed during $description" >> $ERROR_LOG
        exit 1
    else
        echo "Snakemake completed successfully during $description" >> $SUCCESS_LOG
    fi
}

# Step 1: Run setup.sh
echo "Submitting setup.sh job..."
setup_job=$(sbatch /projects/kernlab/akapoor/Demographic_Inference/bash_scripts/setup.sh)
setup_job_id=$(echo $setup_job | awk '{print $4}')
echo "Setup job submitted with ID: $setup_job_id"

# Step 2: Run running_simulation.sh after setup.sh completes
echo "Submitting running_simulation.sh job..."
simulation_job=$(sbatch --dependency=afterok:$setup_job_id /projects/kernlab/akapoor/Demographic_Inference/bash_scripts/running_simulation.sh)
simulation_job_id=$(echo $simulation_job | awk '{print $4}')
echo "Simulation job submitted with ID: $simulation_job_id"

# Step 3: Run momentsLD_inference.sh after running_simulation.sh completes
echo "Submitting momentsLD_inference.sh job..."
echo "Path: /projects/kernlab/akapoor/Demographic_Inference/bash_scripts/momentsLD_inferences.sh"

momentsLD_job=$(sbatch --dependency=afterok:$simulation_job_id /projects/kernlab/akapoor/Demographic_Inference/bash_scripts/momentsLD_inferences.sh)
momentsLD_job_id=$(echo $momentsLD_job | awk '{print $4}')
echo "MomentsLD job submitted with ID: $momentsLD_job_id"

# Wait for job completion
scontrol show jobid $momentsLD_job_id

# Snakemake commands to be run sequentially
run_snakemake "snakemake --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER window_number=$WINDOW_NUMBER --rerun-incomplete '${SIM_DIRECTORY}/sampled_genome_windows/sim_${SIM_NUMBER}/ld_stats_window.${WINDOW_NUMBER}.pkl'" \
              "Calculating LD stats for sim_number $SIM_NUMBER and window_number $WINDOW_NUMBER"

run_snakemake "snakemake --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER --rerun-incomplete '${SIM_DIRECTORY}/sampled_genome_windows/sim_${SIM_NUMBER}/combined_LD_stats_sim_${SIM_NUMBER}.pkl'" \
              "Gathering LD stats for sim_number $SIM_NUMBER"

run_snakemake "snakemake --config sim_directory=$SIM_DIRECTORY sim_number=$SIM_NUMBER --rerun-incomplete '${SIM_DIRECTORY}/simulation_results/momentsLD_inferences_sim_${SIM_NUMBER}.pkl'" \
              "Running MomentsLD analysis for sim_number $SIM_NUMBER"
