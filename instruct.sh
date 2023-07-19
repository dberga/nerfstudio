#!/bin/bash
if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export MODEL="nerfacto"
else
export MODEL=$2
fi
if [ -z $3 ]
then
export DATASET="data/nerfstudio/kitchen"
else
export DATASET=$3
fi
if [ -z "$4" ]
then
export PROMPT="replace humans by aliens"
else
export PROMPT="$4"
fi
if [ -z $5 ]
then
export GSCALE=7.5
else
export GSCALE=$5
fi
if [ -z $6 ]
then
export ISCALE=1.5
else
export ISCALE=$6
fi
export SCENE=$(echo $(basename $DATASET))

echo $PROMPT
CKPT_PATH=$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CKPT_DATE=$(basename $MODEL_PATH)
CFG_PATH=$MODEL_PATH/config.yml
OUTPUT_PATH=$MODEL_PATH/results.json

# RUN
echo "ns-train in2n --data $DATASET --load-dir $MODEL_PATH --pipeline.prompt '$PROMPT' --pipeline.guidance-scale $GSCALE --pipeline.image-guidance-scale $ISCALE --vis viewer"
ns-train in2n --data $DATASET --load-dir $MODEL_PATH --pipeline.prompt '$PROMPT' --pipeline.guidance-scale $GSCALE --pipeline.image-guidance-scale $ISCALE --vis viewer

