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
export VIS="viewer"
else
export VIS=$4
fi
if [ "${MODEL}" = "refnerfacto" ] # [ "${MODEL}" = "nerfacto" ] || 
then
export EXTRAFLAGS="--pipeline.model.predict-normals True"
elif [ "${MODEL}" = "generfacto" ]
then
export EXTRAFLAGS="--pipeline.model.diffusion_model stablediffusion"
else
export EXTRAFLAGS=""
fi

# RUN
if [ "${MODEL}" = "generfacto" ] # for generfacto, dataset is prompt (string)
then

export SCENE=`echo ${DATASET// /_}` # replace spaces (for 'ls' path readout)
echo "ns-train ${MODEL} --prompt '${DATASET}' --experiment-name '${SCENE}' --vis $VIS --viewer.quit-on-train-completion True $EXTRAFLAGS";
ns-train ${MODEL} --prompt "${DATASET}" --experiment-name "${SCENE}" --vis $VIS --viewer.quit-on-train-completion True $EXTRAFLAGS;

else # for nerf models, specify dataset

echo "ns-train ${MODEL} --data ${DATASET} --vis $VIS --viewer.quit-on-train-completion True $EXTRAFLAGS";
ns-train ${MODEL} --data ${DATASET} --vis $VIS --viewer.quit-on-train-completion True $EXTRAFLAGS;

fi

