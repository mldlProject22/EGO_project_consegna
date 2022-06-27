#!/bin/bash

#source ~/~/.local/env_rna/bin/activate # virtual env
#cd /m100_scratch/userexternal/abottin1/Journal_RNA_mirco

set -ex

# complete with correct paths
path_train=/../RNA-Relative-Norm-Alignment/train.py
path_test=/../RNA-Relative-Norm-Alignment/test.py
path_log=/../TEST_RESULTS/log_output
path_resume=/../

source1=$1 #DX
source2=$2 #DY
target=$3 #DZ
radius=$4 #1

N=MultiDG_Flow_DeepAll-radius$radius # name exp

python -u $path_train \
--modality  Flow  \
--model  i3d \
--base_arch  bninception \
--dropout 0.5 -b 32 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 5000 \
--eval-freq 50 \
--name $N-$source1-$source2-$target \
--shift $source1-$source2-$target \
--num_frames_per_clip_train  16 \
--num_frames_per_clip_test  16 \
--num_clips_test 1 \
--verbose \
--dense_sampling_test  True \
--dense_sampling_train  True \
--last \
--radius $radius \
--channels_events 3 >> "$path_log"_"$N"_"$source1"_"$source2"_"$target".txt

#--resume_from $path_resume/saved_models/$N-$source/*  \


python -u $path_test \
--modality  Flow  \
--model  i3d \
--base_arch  bninception \
--dropout 0.5 -b 8 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 9000 \
--eval-freq 50 \
--name $N-$source1-$source2-$target \
--shift $source1-$source2-$target  \
--num_frames_per_clip_train  16 \
--num_frames_per_clip_test  16 \
--verbose \
--dense_sampling_test  True \
--dense_sampling_train  True \
--num_clips_test 5 \
--last \
--resume_from  $path_resume/saved_models/$N-$source1-$source2-$target/ >> "$path_log"_"$N"_"$source1"_"$source2"_"$target".txt
