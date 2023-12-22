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
export OVERWRITE=false
else
export OVERWRITE=$3
fi
#if [ -z $3 ]
#then
#export VIS="wandb"
#else
#export VIS=$3
#fi


for FOLDER in data/*; 
do
echo "sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER $MODEL $OVERWRITE"
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER $MODEL $OVERWRITE
done



