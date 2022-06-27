#!/bin/bash

#source ~/~/.local/env_rna/bin/activate # virtual env
#cd /m100_scratch/userexternal/abottin1/Journal_RNA_mirco

set -ex

# complete with correct paths
path_train=/../RNA-Relative-Norm-Alignment/train.py
path_test=/../RNA-Relative-Norm-Alignment/test.py
path_log=/../TEST_RESULTS/log_output
path_resume=/../


source=$1 #D1
target=$2 #D1

target1=$3 #D1
target2=$4 #D2
target3=$5 #D3

radius=$6

N=RGB-Flow_RNA-radius$radius # name exp


python -u $path_train \
--modality RGB Flow  \
--model i3d i3d \
--base_arch bninception bninception \
--dropout 0.5 -b 32 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 9000 \
--eval-freq 50 \
--name $N-$source \
--shift $source-$target \
--num_frames_per_clip_train 16 16 \
--num_frames_per_clip_test 16 16 \
--num_clips_test 1 \
--verbose \
--dense_sampling_test True True \
--dense_sampling_train True True \
--weight_rna 1 --rna \
--last \
--radius $radius \
--resume_from $path_resume/saved_models/$N-$source/*  \
--channels_events 3 >> "$path_log"_"$N"_"$source".txt




python -u $path_test \
--modality RGB Flow  \
--model i3d i3d \
--base_arch bninception bninception \
--dropout 0.5 -b 32 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 9000 \
--eval-freq 50 \
--name $N-$source-$target1 \
--shift $source-$target1  \
--num_frames_per_clip_train 16 16 \
--num_frames_per_clip_test 16 16 \
--verbose \
--dense_sampling_test True True \
--dense_sampling_train True True \
--num_clips_test 5 \
--last \
--resume_from  $path_resume/saved_models/$N-$source/ >> "$path_log"_"$N"_"$source"-"$target1".txt

python -u $path_test \
--modality RGB Flow  \
--model i3d i3d \
--base_arch bninception bninception \
--dropout 0.5 -b 32 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 9000 \
--eval-freq 50 \
--name $N-$source-$target2 \
--shift $source-$target2  \
--num_frames_per_clip_train 16 16 \
--num_frames_per_clip_test 16 16 \
--verbose \
--dense_sampling_test True True \
--dense_sampling_train True True \
--num_clips_test 5 \
--last \
--resume_from $path_resume/saved_models/$N-$source/ >> "$path_log"_"$N"_"$source"-"$target2".txt

python -u $path_test \
--modality RGB Flow  \
--model i3d i3d \
--base_arch bninception bninception \
--dropout 0.5 -b 32 \
--weight-decay 1e-7 \
--total_batch 128 \
--lr 0.01 \
--lr_steps 3000 -j 10 \
--num_iter 9000 \
--eval-freq 50 \
--name $N-$source-$target3 \
--shift $source-$target3  \
--num_frames_per_clip_train 16 16 \
--num_frames_per_clip_test 16 16 \
--verbose \
--dense_sampling_test True True \
--dense_sampling_train True True \
--num_clips_test 5 \
--last \
--resume_from $path_resume/saved_models/$N-$source/ >> "$path_log"_"$N"_"$source"-"$target3".txt
