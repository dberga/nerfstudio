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
#if [ -z $4 ]
#then
#export VIS="wandb"
#else
#export VIS=$4
#fi

if [ $MODEL == "nerfacto" ] || [ $MODEL == "refnerfacto" ]
then
export EXTRAFLAGS="--pipeline.model.predict-normals"
else
export EXTRAFLAGS=""
fi

# RUN
echo "ns-train ${MODEL} --data ${DATASET} --vis viewer --viewer.quit-on-train-completion True $EXTRAFLAGS";
ns-train ${MODEL} --data ${DATASET} --vis viewer --viewer.quit-on-train-completion True $EXTRAFLAGS;


