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

export EXTRAFLAGS=""

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

CKPT_LIST=$(ls -d $PWD/outputs/$SCENE/$MODEL/*/*/*.ckpt)
for CKPT in $CKPT_LIST
do
CKPT_PATH=$CKPT #$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CKPT_DATE=$(basename $MODEL_PATH)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CFG_PATH=$MODEL_PATH/config.yml

echo $MODEL_PATH;
echo $CKPT_PATH;

if [ -d exports/pcd/$SCENE/$MODEL-$CKPT_DATE ] && [ -d exports/mesh/$SCENE/$MODEL-$CKPT_DATE ]
then
echo "$SCENE/$MODEL-$CKPT_DATE already exported"
continue
fi

# RUN
if [ $MODEL = "vanilla-nerf" ] # vanilla nerf
then
  echo "ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --normal-method open3d --rgb-output-name rgb_fine --depth-output-name depth_fine --num-rays-per-batch 8192";
  ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --normal-method open3d --rgb-output-name rgb_fine --depth-output-name depth_fine --num-rays-per-batch 8192;
  echo "ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --normal-method open3d --rgb-output-name rgb_fine --depth-output-name depth_fine --num-rays-per-batch 8192";
  ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --normal-method open3d --rgb-output-name rgb_fine --depth-output-name depth_fine --num-rays-per-batch 8192;
elif [ $MODEL = "splatfacto" ] # for gaussian splatting
then
  echo "ns-export gaussian-splat --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE";
  ns-export gaussian-splat --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE;
  echo "ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --rgb-output-name rgb --depth-output-name depth --num-rays-per-batch 8192 --normal-method open3d";
  ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --rgb-output-name rgb --depth-output-name depth --num-rays-per-batch 8192 --normal-method open3d;
elif [ $MODEL = "nerfacto" ] # nerfacto calcs normals (no need to use open3d if trained with predicted normals)
then
  echo "ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --rgb-output-name rgb --depth-output-name depth --normal-method open3d ";
  ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --rgb-output-name rgb --depth-output-name depth --normal-method open3d ;
  echo "ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --rgb-output-name rgb --depth-output-name depth --normal-method open3d ";
  ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --rgb-output-name rgb --depth-output-name depth --normal-method open3d ;
else # all other methods
  echo "ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --normal-method open3d --rgb-output-name rgb --depth-output-name depth --normal-method open3d ";
  ns-export pointcloud --load-config $CFG_PATH --output-dir exports/pcd/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --normal-method open3d --rgb-output-name rgb --depth-output-name depth;
  echo "ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --normal-method open3d --rgb-output-name rgb --depth-output-name depth";
  ns-export poisson --load-config $CFG_PATH --output-dir exports/mesh/$SCENE/$MODEL-$CKPT_DATE --num-rays-per-batch 8192 --normal-method open3d --rgb-output-name rgb --depth-output-name depth;
fi
done
