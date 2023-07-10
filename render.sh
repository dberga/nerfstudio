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
if [ -z $4 ]
then
export RESOL=1
else
export RESOL=$4
fi

export SCENE=$(echo $(basename $DATASET))
CKPT_PATH=$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CKPT_DATE=$(basename $MODEL_PATH)
CFG_PATH=$MODEL_PATH/config.yml
OUTPUT_PATH=$MODEL_PATH/results.json
CAM_PATH=$(ls $DATASET/camera_paths/*.json | sort -n | tail -n 1)

# RUN
ns-render camera-path --load-config $CFG_PATH --camera-path-filename $CAM_PATH --output-path renders/$SCENE/$CKPT_DATE.mp4 --downscale-factor $RESOL
echo "ns-render camera-path --load-config $CFG_PATH --camera-path-filename $DATASET/camera_paths/$CKPT_DATE.json --output-path renders/$SCENE/$CKPT_DATE.mp4 --downscale-factor $RESOL"