#!/bin/bash
#SBATCH --job-name fsdp_compare         # job name
#SBATCH --output log_slurm.log     # log file name (%j expands to jobID) use log_slurm.o%j
#SBATCH -n 1                 # total number of tasks requested
#SBATCH -N 1                 # number of nodes you want to run on
#SBATCH --cpus-per-task 48
#SBATCH --gres=gpu:8         # request 8 gpu
#SBATCH -p nam-bio           # queue (partition)
#SBATCH -t 24:00:00          # run time (hh:mm:ss)

. ~/.bashrc

# set -x #echo on
# set -e #exit on error

# module load conda

module load cudnn8.7-cuda11/8.7.0.84
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0

mamba activate mistral

export PYTHONUNBUFFERED=TRUE
cd $HOME/scratch/code/fsdp-compare-finetune-mistral

# cmd="echo 'Running...' && \
# echo 'Done'"
# srun --pty $cmd

echo -e "STARTING..."
echo -e "\n===\n"
echo -e "==>CMD: $cmd"
echo -e "\n"
echo -e "==>PWD:$(pwd)"
echo -e "==>PYTHON: $(which python)"

python strain.py --wandb_mode=online --wandb_group="finetune/fuyu-8b" --model_name="adept/fuyu-8b" --decoder_layer_import="transformers.models.persimmon.modeling_persimmon,PersimmonDecoderLayer" --max_length=2048 --n_epochs=4
echo -e "==>DONE with fuyu-8b"

python strain.py --wandb_mode=online --wandb_group="finetune/mistral-7b" --max_length=2048 --n_epochs=4
echo -e "==>DONE with mistral-7b"
