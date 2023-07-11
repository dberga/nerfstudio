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
export VIS="wandb"
else
export VIS=$4
fi

for SCENE_FOLDER in $FOLDER/*
do
SCENE=$(basename $SCENE_FOLDER)
echo $SCENE
if [ -e "outputs/$SCENE/$MODEL/*/nerfstudio_models" ]
then
if [ -e "outputs/$SCENE/$MODEL/*/nerfstudio_models/*.ckpt" ]
then
CKPT=$( ls outputs/$SCENE/$MODEL/*/nerfstudio_models/*.ckpt | sort -n 1 | tail -n 1)
echo "$CKPT already trained (ckpt exists)"
fi
else
echo "sh train.sh $CUDA_VISIBLE_DEVICES $MODEL $PACK/$SCENE $VIS"
sh train.sh $CUDA_VISIBLE_DEVICES $MODEL $PACK/$SCENE $VIS
fi
done


