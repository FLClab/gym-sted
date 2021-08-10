#!/usr/bin/env bash
#
# Useful documentation: https://docs.google.com/document/d/1-a0vWkz7x5JNSlDRdf5EW2wRLviC4F__g4S4YGexHZA/edit#
#
#SBATCH --time=00:20:00
#SBATCH --account=def-adurand
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --array=0-4
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=betur57@ulaval.ca
#SBATCH --mail-type=ALL
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
VENV_DIR=$~/projects/def-adurand/hazeless/neurips/gym_sted_env

module load python/3.8

[ -d $VENV_DIR ] || virtualenv --no-download $VENV_DIR

source $VENV_DIR/bin/activate

#### RUNNING STUFF

# Moves to working folder
cd ~/projects/def-adurand/hazeless/neurips/gym_sted/gym-sted-pfrl

echo "**** STARTED TRAINING ****"

python main.py --env gym_sted:STEDtimed-v2 --outdir ./data/test2 --steps 10 --eval-interval 10 --eval-n-runs 2

echo "**** ENDED TRAINING ****"