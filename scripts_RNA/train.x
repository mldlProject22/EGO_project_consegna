#!/bin/bash
#SBATCH -p m100_usr_prod
#SBATCH --time 4:16:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1 # 8 tasks out of 128
#SBATCH --gres=gpu:4     # 1 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 246000MB
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mirco.planamente@polito.it


source=$1 #D1
target=$2 #D1
target1=$3 #D1
target2=$4 #D2
target3=$5 #D3
file=$6
radius=$7

module load profile/deeplrn
module load autoload /epic-kitchens

chmod +x /m100/home/userexternal/abottin1/Mirco_ActivityRecognition_DA/scripts_mirco_Journal_RNA/$file
srun /m100/home/userexternal/abottin1/Mirco_ActivityRecognition_DA/scripts_mirco_Journal_RNA/$file  $source $target $target1 $target2 $target3 $radius


