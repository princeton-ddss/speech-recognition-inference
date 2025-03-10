#!/bin/bash
#SBATCH --job-name=whisper_tiny_batch # create a short name for your job
#SBATCH --output=whisper_tiny_batch.out # stdout file
#SBATCH --error=whisper_tiny_batch.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1            # number of gpu
#SBATCH --partition=mig         #mig partition
#SBATCH --mem-per-cpu=32G        # memory per cpu-core (4G is default)
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jf3375@princeton.edu

cd /scratch/gpfs/jf3375/projects/speech-recognition-inference/src/batch_processing

source /scratch/gpfs/jf3375/virtualenvs/localml/bin/activate

python sample_run.py
