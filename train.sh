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
export VIS="wandb"
else
export VIS=$4
fi

# RUN
echo "ns-train ${MODEL} --data ${DATASET} --vis $VIS;";
ns-train ${MODEL} --data ${DATASET} --vis $VIS;

############################################
#export CUDA_VISIBLE_DEVICES=0;
#export MODEL='nerfacto';
#ns-train nerfacto --data data/nerfstudio/poster --vis wandb;
#ns-train nerfacto --data data/nerfstudio/bww_entrance --vis wandb;
#ns-train nerfacto --data data/nerfstudio/campanile --vis wandb;
#ns-train nerfacto --data data/nerfstudio/desolation --vis wandb;
#ns-train nerfacto --data data/nerfstudio/library --vis wandb;
#ns-train nerfacto --data data/nerfstudio/redwoods2 --vis wandb;
#ns-train nerfacto --data data/nerfstudio/storefront --vis wandb;
#ns-train nerfacto --data data/nerfstudio/vegetat --vis wandb;
#ns-train nerfacto --data data/nerfstudio/ion --vis wandb;
#ns-train nerfacto --data data/nerfstudio/Egypt --vis wandb;
#ns-train nerfacto --data data/nerfstudio/person --vis wandb;
#ns-train nerfacto --data data/nerfstudio/kitchen --vis wandb;
#ns-train nerfacto --data data/nerfstudio/plane --vis wandb;
#ns-train nerfacto --data data/nerfstudio/dozer --vis wandb;
#ns-train nerfacto --data data/nerfstudio/floating-tree --vis wandb;
#ns-train nerfacto --data data/nerfstudio/aspen --vis wandb;
#ns-train nerfacto --data data/nerfstudio/stump --vis wandb;
#ns-train nerfacto --data data/nerfstudio/sculpture --vis wandb;
#ns-train nerfacto --data data/nerfstudio/Giannini-Hall --vis wandb;

