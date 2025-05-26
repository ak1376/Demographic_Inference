#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=72:00:00  # Increase overall time for the pipeline
#SBATCH --mem=64G
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akapoor@uoregon.edu
#SBATCH --verbose

# Ensure logs directory exists
mkdir -p logs

# Function to unlock Snakemake if a lock is present
unlock_snakemake() {
    echo "Checking for Snakemake lock..."
    snakemake --unlock
    if [ $? -eq 0 ]; then
        echo "Snakemake unlocked successfully."
    else
        echo "Snakemake unlock failed or was not needed."
    fi
}

# Call unlock_snakemake function at the start
unlock_snakemake

# Function to submit jobs and retrieve job IDs
submit_job() {
    local script="$1"
    local dependency="$2"
    local job_id=""

    if [ -n "$dependency" ]; then
        job_id=$(sbatch --dependency=afterok:$dependency "$script" | awk '{print $4}')
    else
        job_id=$(sbatch "$script" | awk '{print $4}')
    fi

    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "$job_id"
    else
        echo "Error submitting job for script $script. Exiting."
        exit 1
    fi
}

# Function to wait for job array completion and log time and memory
wait_for_job_array_completion() {
    local job_id="$1"
    local job_name="$2"
    local start_time="$3"

    echo "Waiting for completion of job array $job_id ($job_name)..."
    
    # Add sleep here to allow SLURM to process job completions
    sleep 5
    
    while squeue -j "$job_id" -h -o "%T" | grep -q .; do
        sleep 30
    done

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Query memory usage for all tasks in the job array
    local max_memory=$(sacct -j "${job_id}" --format=JobID,MaxRSS --noheader | \
        awk '{if ($2 ~ /K$/) sum += $2 / 1024; else if ($2 ~ /M$/) sum += $2; else if ($2 ~ /G$/) sum += $2 * 1024} END {print sum " MB"}')

    echo "$job_name,$job_id,$start_time,$end_time,$elapsed,$max_memory" >> logs/job_stats.txt

    echo "$job_name completed: Start Time = $start_time, End Time = $end_time, Elapsed = $elapsed seconds, Max Memory = $max_memory"
}

# Initialize the stats log file with a header
echo "JobName,JobID,StartTime,EndTime,Elapsed,MaxMemory" > logs/job_stats.txt

# Submit jobs and start tracking times
echo "Submitting all jobs and capturing start times..."

# 1. Setup
setup_id=$(submit_job "bash_scripts/setup.sh")
setup_start=$(date +%s)
wait_for_job_array_completion "$setup_id" "Setup" "$setup_start"

# 2. Simulation
sim_id=$(submit_job "bash_scripts/running_simulation.sh" "$setup_id")
sim_start=$(date +%s)
wait_for_job_array_completion "$sim_id" "Simulation" "$sim_start"

# -- Removed genome windows and combine metadata steps --

# 3. LD Stats (depends directly on Simulation now)
ld_stats_id=$(submit_job "bash_scripts/LD_stats_windows.sh" "$sim_id")
ld_stats_start=$(date +%s)
wait_for_job_array_completion "$ld_stats_id" "LD Stats Window" "$ld_stats_start"

# 4. momentsLD inferences
momentsld_id=$(submit_job "bash_scripts/momentsLD_inferences.sh" "$ld_stats_id")
momentsld_start=$(date +%s)
wait_for_job_array_completion "$momentsld_id" "MomentsLD" "$momentsld_start"

# 5. moments/dadi
moments_dadi_id=$(submit_job "bash_scripts/moments_dadi.sh" "$momentsld_id")
moments_dadi_start=$(date +%s)
wait_for_job_array_completion "$moments_dadi_id" "Moments/Dadi" "$moments_dadi_start"

# 6. Aggregate features
aggregate_features_id=$(submit_job "bash_scripts/aggregate_features.sh" "$moments_dadi_id")
aggregate_features_start=$(date +%s)
wait_for_job_array_completion "$aggregate_features_id" "Aggregate Features" "$aggregate_features_start"

echo "Job statistics have been written to logs/job_stats.txt"
column -t -s, logs/job_stats.txt
