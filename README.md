# Domain Adaptation in First Person Action Recognition


## Requirements

* Install project's requirements in a separate conda environment. In your terminal:<br> `$conda create --name <env> --file requirements.txt`. 

## Data preparation

### Data

This step assumes that you've downloaded the RGB and Flow frames of EPIC dataset using [this script](https://github.com/jonmun/MM-SADA_Domain_Adaptation_Splits/blob/master/download_script.sh). Also you should untar each video's frames in its corresponding folder. 

`loaders.py` uses a unified folder structure for all datasets, which is the same as the one used in the [TSN code](https://github.com/yjxiong/tsn-pytorch). Example of the folder structure for RGB and Flow:

```
├── dataset_root
|   ├── P01_01
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000100.jpg
|   |   ├── x_0000000100.jpg
|   |   ├── y_0000000100.jpg
|   ├── .
|   ├── .
|   ├── .
|   ├── P22_17
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000250.jpg
|   |   ├── x_0000000250.jpg
|   |   ├── y_0000000250.jpg
```
        
To map the folder structure of EPIC to the above folder structure you can use symlinks. Use the following script to convert
the original folder structure of EPIC to the folder structure above:

```
python ops/symlinks.py /path/to/rgb_flow/ /path/to/output
```

## Training

To train different combinations of input and modalities, run:
```
usage: train.py [-h] [--modality {RGB,Flow,Event} [{RGB,Flow,Event} ...]]
                [--train_list TRAIN_LIST] 
                [--val_list VAL_LIST] 
                [--visual_path VISUAL_PATH] 
                [--flow_path FLOW_PATH] 
                [--event_path EVENT_PATH] #only for the event data
                [--model MODEL [MODEL ...]] # i3d | TSN  (one for each modality) 
                [--base_arch BASE_ARCH [BASE_ARCH ...]] #bninception (one for each modality)
                [--num_frames_per_clip_train NUM_FRAMES_PER_CLIP_TRAIN [NUM_FRAMES_PER_CLIP_TRAIN ...]] (one for each modality)
                [--num_frames_per_clip_test NUM_FRAMES_PER_CLIP_TEST [NUM_FRAMES_PER_CLIP_TEST ...]] (one for each modality)
                [--dropout DO]
                [--num_iter N] 
                [--total_batch N] #total batch size used for the training 
                [-b N] # if b < total_batch --> batch accumulation 
                [--lr LR] 
                [--lr_steps LRSteps] 
                [--weight-decay W]
                [--eval-freq N] [--verbose]
                [--shift SHIFT] [--name NAME]
                [--gpus GPUS]
                [--resume_from RESUME_FROM] [--last] #to restart a training from LAST checkpoint or validate it
                [--resume_from_iteration RESUME_FROM_ITERATION] [--iteration ITERATION] #to restart a training from ITERATION checkpoint or validate it
                [--dense_sampling_train DENSE_SAMPLING_TRAIN [DENSE_SAMPLING_TRAIN ...]] # True (dense) | False (uniform) (one for each modality)
                [--dense_sampling_test DENSE_SAMPLING_TEST [DENSE_SAMPLING_TEST ...]] # True (dense) | False (uniform) (one for each modality)
                [--num_clips_test {1,5}]
                [--rna] [--weight_rna 1] #rna

```

The training and validation sets used are the ones present [here](https://github.com/jonmun/MM-SADA_Domain_Adaptation_Splits) 

## Testing

For testing the same usage shown before is adopted. ```test.py``` will test the last 9 models saved every 50 iterations
and average the results.


## Predefined scripts

[Here](./scripts_RNA) we have reported some predifined scripts for the settings shown in the paper.


