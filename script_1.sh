#!/bin/bash
#SBATCH -w node4
#
source /home/tao/.bashrc
source activate tao

module load python
python /data_hdd/tao/treecover_segmentation/treecover/active2.py

