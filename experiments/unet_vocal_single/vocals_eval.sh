#!/bin/bash

OUTPUT_DIR="./unet_vocal_single/result"
original_path="./dataset/orig_mono16000/test"
ESTIMATES_DIR="./unet_vocal_single/result/Experiment/test"
subsets="test"
list_path="./list.txt"
save_name="01.vocals_eval.sh"
inst="vocals"
python3 eval.py --estimates_dir $ESTIMATES_DIR --subsets $subsets --output_dir $OUTPUT_DIR  --list_path $list_path --original_path $original_path --savename $save_name --inst $inst
