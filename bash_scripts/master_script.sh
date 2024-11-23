#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=10:00:00  # Increase overall time for the pipeline
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

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

# Function to wait for job array completion and log time
wait_for_job_array_completion() {
    local job_id="$1"
    local job_name="$2"
    local start_time="$3"

    echo "Waiting for completion of job array $job_id ($job_name)..."
    while squeue -j "$job_id" -h -o "%T" | grep -q .; do
        sleep 30
    done

    # Capture end time and calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "$job_name,$job_id,$start_time,$end_time,$elapsed" >> logs/job_stats.txt

    echo "$job_name completed: Start Time = $start_time, End Time = $end_time, Elapsed = $elapsed seconds"
}

# Initialize the stats log file with a header
echo "JobName,JobID,StartTime,EndTime,Elapsed" > logs/job_stats.txt

# Submit jobs and start tracking times
echo "Submitting all jobs and capturing start times..."

setup_id=$(submit_job "bash_scripts/setup.sh")
setup_start=$(date +%s)
wait_for_job_array_completion "$setup_id" "Setup" "$setup_start"

sim_id=$(submit_job "bash_scripts/running_simulation.sh" "$setup_id")
sim_start=$(date +%s)
wait_for_job_array_completion "$sim_id" "Simulation" "$sim_start"

genome_id=$(submit_job "bash_scripts/genome_windows.sh" "$sim_id")
genome_start=$(date +%s)
wait_for_job_array_completion "$genome_id" "Genome Windows" "$genome_start"

ld_stats_id=$(submit_job "bash_scripts/LD_stats_windows.sh" "$genome_id")
ld_stats_start=$(date +%s)
wait_for_job_array_completion "$ld_stats_id" "LD Stats Window" "$ld_stats_start"

momentsld_id=$(submit_job "bash_scripts/momentsLD_inferences.sh" "$ld_stats_id")
momentsld_start=$(date +%s)
wait_for_job_array_completion "$momentsld_id" "MomentsLD" "$momentsld_start"

moments_dadi_id=$(submit_job "bash_scripts/moments_dadi.sh" "$momentsld_id")
moments_dadi_start=$(date +%s)
wait_for_job_array_completion "$moments_dadi_id" "Moments/Dadi" "$moments_dadi_start"

# Submit aggregate_features.sh after moments_dadi.sh
aggregate_features_id=$(submit_job "bash_scripts/aggregate_features.sh" "$moments_dadi_id")
aggregate_features_start=$(date +%s)
wait_for_job_array_completion "$aggregate_features_id" "Aggregate Features" "$aggregate_features_start"

remaining_id=$(submit_job "bash_scripts/remaining_rules.sh" "$aggregate_features_id")
remaining_start=$(date +%s)
wait_for_job_array_completion "$remaining_id" "Remaining Rules" "$remaining_start"

echo "Job statistics have been written to logs/job_stats.txt"
column -t -s, logs/job_stats.txt
