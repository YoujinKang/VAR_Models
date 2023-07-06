# bash scripts/finetune_var.bash 0 0 9590 1
# The name of this experiment.
name=./snap/var2/$2


# export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP
export PYTHONPATH=$PYTHONPATH

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/var_eval.py \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --load $name \
    --batchSize 400 --tqdm
