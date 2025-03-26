#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main 
#SBATCH --time 0-01:00:00 ### limit the time of job running. Format: D-H:MM:SS
#SBATCH --job-name initial_protein_evaluation 
#SBATCH --output initial_protein_evaluation-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=meytav@post.bgu.ac.il ### user’s email for sending job status notifications
#SBATCH --mail-type=END,FAIL ### conditions for sending the email. 
#SBATCH --gpus=0 ### number of GPUs. 
##SBATCH --tasks=1 # 1 process 
### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda ### load anaconda module
source activate my_env ### activate a conda environment, replace my_env with your conda environment
python protein_initial_evaluation.csv