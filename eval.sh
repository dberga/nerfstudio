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
CKPT_NAME=$(basename $(dirname $(dirname $CKPT)))
MODEL_PATH=$(dirname $(dirname $CKPT_PATH))
CKPT_DATE=$(basename $MODEL_PATH)
CFG_PATH=$MODEL_PATH/config.yml
OUTPUT_PATH=$MODEL_PATH/results.json
RENDER_PATH=$MODEL_PATH/renders

if [ -e $RENDER_PATH ]
then
rm -rf $RENDER_PATH/*
fi

# RUN
echo "ns-eval --load-config=$CFG_PATH --output-path=$OUTPUT_PATH --render-output-path=$RENDER_PATH"
ns-eval --load-config=$CFG_PATH --output-path=$OUTPUT_PATH --render-output-path=$RENDER_PATH

done
