#!/usr/bin/env bash
#
# Useful documentation: https://docs.google.com/document/d/1-a0vWkz7x5JNSlDRdf5EW2wRLviC4F__g4S4YGexHZA/edit#
#
#SBATCH --time=48:00:00
#SBATCH --account=def-adurand
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --array=0
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=betur57@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
VENV_DIR=${HOME}/projects/def-adurand/hazeless/neurips/gym_sted_env

module load python/3.8

[ -d $VENV_DIR ] || virtualenv --no-download $VENV_DIR

source $VENV_DIR/bin/activate

#### RUNNING STUFF

# Moves to working folder
cd ~/projects/def-adurand/hazeless/neurips/gym_sted

echo "**** STARTED TRAINING ****"

python gym-sted-pfrl/analysis/gym2_results_analysis_1.py --main_dir ./data/sample_env_training_v4_hard/20210820-125709_088550d6/ --analysis_n_runs 50

echo "**** ENDED TRAINING ****"