#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting pipeline execution..."

# 1. Submit setup job
echo "Submitting setup job..."
setup_job=$(sbatch bash_scripts/setup.sh | awk '{print $4}')

# 2. Submit simulation job with dependency
echo "Submitting simulation job..."
sim_job=$(sbatch --dependency=afterok:$setup_job bash_scripts/running_simulation.sh | awk '{print $4}')

# 3. Submit genome windows job with dependency
echo "Submitting genome windows job..."
genome_job=$(sbatch --dependency=afterok:$sim_job bash_scripts/genome_windows.sh | awk '{print $4}')

# 4. Submit momentsLD inferences job with dependency
echo "Submitting momentsLD inferences job..."
momentsld_job=$(sbatch --dependency=afterok:$genome_job bash_scripts/momentsLD_inferences.sh | awk '{print $4}')

# 5. Submit moments and dadi job with dependency
echo "Submitting moments and dadi job..."
moments_dadi_job=$(sbatch --dependency=afterok:$momentsld_job bash_scripts/moments_dadi.sh | awk '{print $4}')

# 6. Submit remaining rules job with dependency
echo "Submitting remaining rules job..."
sbatch --dependency=afterok:$moments_dadi_job bash_scripts/remaining_rules.sh

echo "All jobs submitted with dependencies!"