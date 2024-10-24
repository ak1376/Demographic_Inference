for i in {1..40}; 
do
    snakemake \
    --batch all=${i}/40 --executor slurm --workflow-profile /home/akapoor/.config/snakemake/slurm/ \
    --rerun-incomplete \
    -j 1000 \
    --slurm-requeue  # No need to specify True
done