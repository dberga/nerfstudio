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
export RESOL=1
else
export RESOL=$4
fi
if [ -z $5 ]
then
export OVERWRITE=false
else
export OVERWRITE=$5
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
CKPT_DATE=$(basename $MODEL_PATH)
CFG_PATH=$MODEL_PATH/config.yml

if [ -e renders/$SCENE/$MODEL/$CKPT_DATE.mp4 ] && ! $OVERWRITE; then
echo "renders/$SCENE/$MODEL/$CKPT_DATE.mp4 already rendered"
exit
fi

PATH_CAMERA_JSON=${DATASET}/camera_paths/*.json
if [ -e $PATH_CAMERA_JSON ]; then
CAM_PATH=$(ls $DATASET/camera_paths/*.json | sort -n | tail -n 1)
echo "ns-render camera-path --load-config $CFG_PATH --camera-path-filename $DATASET/camera_paths/$CKPT_DATE.json --output-path renders/$SCENE/$MODEL/$CKPT_DATE.mp4 --downscale-factor $RESOL  --rendered_output_name rgb_fine"
ns-render camera-path --load-config $CFG_PATH --camera-path-filename $CAM_PATH --output-path renders/$SCENE/$MODEL/$CKPT_DATE.mp4 --downscale-factor $RESOL --rendered_output_names rgb_fine
else

echo "ns-render interpolate --load-config $CFG_PATH --output-path renders/$SCENE/$MODEL/$CKPT_DATE.mp4 --downscale-factor $RESOL --rendered_output_names rgb"
ns-render interpolate --load-config $CFG_PATH --output-path renders/$SCENE/$MODEL/$CKPT_DATE.mp4 --downscale-factor $RESOL --rendered_output_names rgb

echo "ns-render spiral --load-config $CFG_PATH --output-path renders/$SCENE/$MODEL/spiral-$CKPT_DATE.mp4 --downscale-factor $RESOL --rendered_output_names rgb"
ns-render spiral --load-config $CFG_PATH --output-path renders/$SCENE/$MODEL/spiral-$CKPT_DATE.mp4 --downscale-factor $RESOL --rendered_output_names rgb
fi

# mp4 video to gif
#echo "ffmpeg -y -i renders/$SCENE/$MODEL/$CKPT_DATE.mp4 renders/$SCENE/$MODEL/$CKPT_DATE.gif"
#ffmpeg -y -i renders/$SCENE/$MODEL/$CKPT_DATE.mp4 renders/$SCENE/$MODEL/$CKPT_DATE.gif -filter_complex "fps=30,scale=480:-1"


