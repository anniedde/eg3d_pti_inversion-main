#!/bin/bash

# Get the list of celebs
celeb_list=('Harry' 'IU' 'Margot' 'Michael')
experiments=('lower_bound' 'random' 'ransac' 'upper_bound')

# Set the initial device number
device=0

# Loop through each celeb and run the script
for celeb in "${celeb_list[@]}"; do
    (
        for experiment in "${experiments[@]}"; do
            MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$device python batch_synthesize_celeb.py --celeb $celeb --experiment $experiment --which_gpus $device
        done
    ) &
    ((device++))
done

# Wait for all the scripts to finish
wait
