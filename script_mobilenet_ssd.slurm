#!/bin/sh
#SBATCH --account=ie-idi      # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=mobilenet_ssd_job
#SBATCH --time=0-12:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --cpus-per-task=4 
#SBATCH --nodes=1
#SBATCH --output=output_mobilenet_ssd.txt      # Specifying 'stdout'
#SBATCH --error=output_mobilenet_ssd.err        # Specifying 'stderr'

#SBATCH --mail-user=magnsjak@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1

# Install packages if not already installed (with --user flag)
pip install --user torch torchvision pillow matplotlib pyyaml pandas

# Run with unbuffered output so print statements appear immediately
python -u models/lightweight/mobilenet_ssd_script.py