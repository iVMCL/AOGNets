#!/usr/bin/env bash

# Usage: train_fp16.sh arch_name config_filename name_tag

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$#" -ne 3 ]; then
    echo "Usage: train_fp16.sh arch_name relative_config_filename name_tag"
    exit
fi

ARCH_NAME=$1
CONFIG_FILE=$DIR/../$2

### Change accordingly
GPUS=0,1,2,3,4,5,6,7
NUM_GPUS=8
NUM_WORKERS=8

CONFIG_FILENAME="$(cut -d'/' -f2 <<<$2)"
CONFIG_BASE="${CONFIG_FILENAME%.*}"
NAME_TAG=$3
SAVE_DIR=$DIR/../results/$ARCH_NAME-$CONFIG_BASE-$NAME_TAG
if [ -d $SAVE_DIR ]; then
    echo "$SAVE_DIR --- Already exists, try a different name tag or delete it first"
    exit
else
    mkdir -p $SAVE_DIR
fi

# backup for reproducing results
cp $CONFIG_FILE $SAVE_DIR/config.yaml
cp -r $DIR/../backbones $SAVE_DIR
cp $DIR/../tools/main_fp16.py $SAVE_DIR

# ImageNet
DATA=$DIR/../datasets/ILSVRC2015/Data/CLS-LOC/

# train
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    $DIR/../tools/main_fp16.py -a $ARCH_NAME --cfg $CONFIG_FILE --workers $NUM_WORKERS \
    --fp16 --static-loss-scale 128 \
    -p 100 --save-dir $SAVE_DIR $DATA  \
    2>&1 | tee $SAVE_DIR/log.txt

