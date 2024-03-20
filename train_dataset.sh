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


sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER vanilla-nerf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER mipnerf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER nerfacto;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER nerfacto-big;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER nerfacto-huge;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER instant-ngp;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER instant-ngp-bounded;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER dnerf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER neus;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER neus-facto;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER tensorf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER volinga;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER kplanes;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER kplanes-dynamic;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER tetra-nerf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER tetra-nerf-original;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER phototourism;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER splatfacto;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER pynerf;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER pynerf-synthetic;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER pynerf-occupancy-grid;
sh all_train.sh $CUDA_VISIBLE_DEVICES $FOLDER nerfbusters;

