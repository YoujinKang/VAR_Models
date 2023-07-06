# bash scripts/finetune_var.bash 0 0 9590 1
# The name of this experiment.
name=snap/var4

# Save logs and models under snap/var; make backup.
output=$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP
export PYTHONPATH=$PYTHONPATH

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/var.py \
    --train train --valid val \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --batchSize 16 --optim bert --lr 1e-6 --epochs 15 --gradient_accumulation_steps 1 \
    --tqdm --output $output ${@:3}
