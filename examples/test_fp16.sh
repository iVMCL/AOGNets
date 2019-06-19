#!/usr/bin/env bash

# Usage: test_fp16.sh arch_name pretrained_model_folder

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$#" -ne 2 ]; then
    echo "Usage: test_fp16.sh arch_name pretrained_model_folder"
    exit
fi

ARCH_NAME=$1
PRETRAINED_MODEL_PAHT=$2
CONFIG_FILE=$PRETRAINED_MODEL_PAHT/config.yaml
PRETRAINED_MODEL_FILE=$PRETRAINED_MODEL_PAHT/model.pth.tar

### Change accordingly
GPUS=2,3,4,5,6,7
NUM_GPUS=6
NUM_WORKERS=8

# ImageNet
DATA=$DIR/../datasets/ILSVRC2015/Data/CLS-LOC/

# test
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    $DIR/../tools/main_fp16.py -a $ARCH_NAME --cfg $CONFIG_FILE --workers $NUM_WORKERS \
    --fp16 \
    -p 100 --save-dir $PRETRAINED_MODEL_PAHT --pretrained --evaluate $DATA \
    2>&1 | tee $PRETRAINED_MODEL_PAHT/log_test.txt



