# all exp

# DG SINGLE SOURCE
path=/path/../RNA-Relative-Norm-Alignment/scripts_RNA

# file train.x (for cineca) resource to required for the job
# source target target1 target2 target3
# (source - target) are for the training and (target1 target2 target3) for the test phase
# name script.sh to run
# radius rna --> 1
sbatch $path/train.x D1 D1 D1 D2 D3 NOME_FILE_SH.sh 1
sbatch $path/train.x D2 D2 D2 D1 D3 NOME_FILE_SH.sh 1
sbatch $path/train.x D3 D3 D3 D1 D2 NOME_FILE_SH.sh 1


# UDA
path=/path/../RNA-Relative-Norm-Alignment/scripts_RNA/UDA
# file train.x (for cineca) resource to required for the job
# source target for UDA
# radius rna --> 1
sbatch $path/train.x D1 D2 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1
sbatch $path/train.x D1 D3 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1

sbatch $path/train.x D2 D1 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1
sbatch $path/train.x D2 D3 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1

sbatch $path/train.x D3 D1 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1
sbatch $path/train.x D3 D2 RGB-FLOW_UDA_RNA1_learnFC_lr0.001_train.sh 1