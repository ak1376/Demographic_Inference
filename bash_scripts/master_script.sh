#!/bin/bash
#SBATCH --job-name=main_pipeline
#SBATCH --output=logs/pipeline_main.out
#SBATCH --error=logs/pipeline_main.err
#SBATCH --time=00:10:00
#SBATCH --partition=kern,preempt,kerngpu
#SBATCH --account=kernlab
#SBATCH --requeue

mkdir -p logs

# Submit all jobs first with dependencies
echo "Submitting all jobs..."

# Submit setup
setup_id=$(sbatch bash_scripts/setup.sh | awk '{print $4}')
echo "Setup job submitted: $setup_id"

# Submit rest with dependencies
sim_id=$(sbatch --dependency=afterok:$setup_id bash_scripts/running_simulation.sh | awk '{print $4}')
echo "Simulation job submitted: $sim_id"

genome_id=$(sbatch --dependency=afterok:$sim_id bash_scripts/genome_windows.sh | awk '{print $4}')
echo "Genome windows job submitted: $genome_id"

momentsld_id=$(sbatch --dependency=afterok:$genome_id bash_scripts/momentsLD_inferences.sh | awk '{print $4}')
echo "MomentsLD job submitted: $momentsld_id"

moments_dadi_id=$(sbatch --dependency=afterok:$momentsld_id bash_scripts/moments_dadi.sh | awk '{print $4}')
echo "Moments/dadi job submitted: $moments_dadi_id"

remaining_id=$(sbatch --dependency=afterok:$moments_dadi_id bash_scripts/remaining_rules.sh | awk '{print $4}')
echo "Remaining rules job submitted: $remaining_id"

# Function to check if any jobs are still running or pending
check_jobs_status() {
    local jobs="$1"
    local running=0
    
    for job in $jobs; do
        state=$(sacct -j "$job" --format=State --noheader | head -n1 | tr -d ' ')
        case $state in
            PENDING|RUNNING|REQUEUED)
                running=1
                break
                ;;
        esac
    done
    
    return $running
}

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
all_jobs="$setup_id $sim_id $genome_id $momentsld_id $moments_dadi_id $remaining_id"

while check_jobs_status "$all_jobs"; do
    echo "Jobs still running... checking again in 30 seconds"
    sleep 30
done

# Wait a bit more to ensure job completion is registered
sleep 10

# Get runtime statistics for all jobs
echo "Getting runtime statistics..."

{
    echo "JobName|JobID|Elapsed|MaxRSS|NCPUS"
    sacct -j "$setup_id,$sim_id,$genome_id,$momentsld_id,$moments_dadi_id,$remaining_id" \
        --format=JobName,JobID,Elapsed,MaxRSS,NCPUS \
        --parsable2 | grep -v "batch" | grep -v "extern"
} > logs/job_stats.csv

# Convert the | separated file to a proper CSV
sed 's/|/,/g' logs/job_stats.csv > logs/job_stats_temp.csv
mv logs/job_stats_temp.csv logs/job_stats.csv

echo "Job statistics have been written to logs/job_stats.csv"
echo "Summary of job runs:"
column -t -s, logs/job_stats.csv