if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export FOLDER="data/nerfstudio"
else
export FOLDER=$2
fi
if [ -z $3 ]
then
export MODEL="nerfacto"
else
export MODEL=$3
fi
if [ -z $4 ]
then
export OVERWRITE=false
else
export OVERWRITE=$4
fi
#if [ -z $4 ]
#then
#export VIS="wandb"
#else
#export VIS=$4
#fi

for SCENE_FOLDER in $FOLDER/*
do

if [ ! -d "${SCENE_FOLDER}" ]; then 
continue;
fi

SCENE=$(basename $SCENE_FOLDER)
DATASET=$FOLDER/$SCENE

PATH_TRANSFORMS=${DATASET}/transforms*.json
if [ ! $(ls -a ${PATH_TRANSFORMS} | wc -l) ]; then 
echo "$DATASET not processed, discard training."
continue
fi

if [ -e "outputs/${SCENE}/${MODEL}" ]
then
if [ `ls -a outputs/$SCENE/$MODEL | wc -l` ]
#if [[ ! -z $(ls -A outputs/$SCENE/$MODEL) ]]
then
for DATE in outputs/$SCENE/$MODEL/*
do
CKPT_DATE=$(basename $DATE)
if [ -e "outputs/$SCENE/$MODEL/$CKPT_DATE/nerfstudio_models" ]
then
EXIST_CKPT=$(ls -a outputs/$SCENE/$MODEL/$CKPT_DATE/nerfstudio_models/*.ckpt)
NUM_EXIST_CKPT=$( ls $EXIST_CKPT | wc -l)
if [ $NUM_EXIST_CKPT ] && ! $OVERWRITE
then
if [ $NUM_EXIST_CKPT -gt 1 ]; then
CKPT=$( ls $EXIST_CKPT | sort -n 1 | tail -n 1)
else
CKPT=$EXIST_CKPT
fi
echo "$CKPT already trained (ckpt exists)"
break
fi
fi
done
fi
fi
echo "sh train.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET"
sh train.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET
done


