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
export PORT="7007"
else
export PORT=$4
fi

export SCENE=$(echo $(basename $DATASET))
CKPT_PATH=$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CFG_PATH=$MODEL_PATH/config.yml
OUTPUT_PATH=$MODEL_PATH/results.json

# RUN
ns-viewer --load-config $CFG_PATH --viewer.websocket-port $PORT;
echo "ns-viewer --load-config $CFG_PATH --viewer.websocket-port $PORT"

