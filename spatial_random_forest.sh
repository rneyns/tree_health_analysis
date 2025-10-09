#!/bin/bash

#SBATCH -l nodes=1:ppn:1

#SBATCH --mem-per-cpu=64G

#SBATCH --time=4:00:00

cd /data/brussel/104/vsc10421/

ml purge

ml load Python/3.12.3-GCCcore-13.3.0
ml load scikit-learn/1.6.1-gfbf-2024a

python3 -m venv venv-geographicalRegression --system-site-packages
source venv-geographicalRegression/bin/activate

python3 -m pip install pyGRF

cd /user/brussel/104/vsc10421/Tree_health_classification/

python -u Train_model.py  -i "/data/brussel/104/vsc10421/tree health classification/output_patches/output_patches_hydra/" -multi "/data/brussel/104/vsc10421/tree health classification/output_patches/5_species_other/" -lh species_code -idh tree_id -w 120 -ts 17 -nc 6 --epochs 10 --alpha 0.5 --task 'multiclass' --undersample --spatio_temp --batch_size 16
