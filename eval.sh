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

export SCENE=$(echo $(basename $DATASET))
CKPT_PATH=$(ls outputs/$SCENE/$MODEL/*/*/*.ckpt | sort -n | tail -n 1)
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CFG_PATH=$MODEL_PATH/config.yml
OUTPUT_PATH=$MODEL_PATH/results.json

# RUN
ns-eval --load-config=$CFG_PATH --output-path=$OUTPUT_PATH
echo "ns-eval --load-config=$CFG_PATH --output-path=$OUTPUT_PATH;";

