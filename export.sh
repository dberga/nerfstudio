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
if [ -z "$3" ]
then
export DATASET="data/nerfstudio/kitchen"
else
export DATASET="$3"
fi
if [ -z $4 ]
then
export SCALE=1
else
export SCALE=$4
fi

if [ "${MODEL}" = "generfacto" ] # for model generation
then
SCENE=`echo ${DATASET// /_}`  # output folder name, using _ instead of spaces
  if ! [ -e outputs/$SCENE ]
  then
  export SCENE=""
  fi
else # for any nerf model
export SCENE=$(echo $(basename $DATASET))
fi

CKPT_PATH=$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CFG_PATH=$MODEL_PATH/config.yml

# RUN
ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL --num-points 1000000 --remove-outliers True --normal-method open3d --use-bounding-box True --bounding-box-min -$SCALE -$SCALE -$SCALE --bounding-box-max $SCALE $SCALE $SCALE;
ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL --target-num-faces 50000 --num-pixels-per-side 2048 --normal-method open3d --num-points 1000000 --remove-outliers True --use-bounding-box True --bounding-box-min -$SCALE -$SCALE -$SCALE --bounding-box-max $SCALE $SCALE $SCALE

