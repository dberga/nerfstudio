if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export RESOL=1
else
export RESOL=$2
fi
if [ -z $3 ]
then
export OVERWRITE=false
else
export OVERWRITE=$3
fi

ALL_CKPTS=$(ls outputs/*/*/*/*/*.ckpt)
for CKPT in $ALL_CKPTS
do

DATE=$(basename $(dirname $(dirname $CKPT)))
MODEL=$(basename $(dirname $(dirname $(dirname $CKPT))))
SCENE=$(basename $(dirname $(dirname $(dirname $(dirname $CKPT)))))

#echo "$DATE"
#echo "$MODEL"
#echo "$SCENE"

if [ "${MODEL}" = "generfacto" ] # for model generation
then
	export DATASET=$SCENE
        PATH_EXPORTS_MESH=exports/mesh/${SCENE}/${MODEL}/*.obj
        PATH_EXPORTS_PCD=exports/pcd/${SCENE}/${MODEL}/*.ply
	if [ -e $PATH_EXPORTS_MESH ] && [ -e $PATH_EXPORTS_PCD ] && ! $OVERWRITE
	then
	echo "$SCENE already exported to mesh/pcd"
	else
	echo "sh export.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1"
	sh export.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1
	fi
else # for any nerf model
	for FOLDER in data/*;
	do
	PACK=$(basename $FOLDER)
	if [ -e data/$PACK/$SCENE ];
	then
	DATASET=data/$PACK/$SCENE
        PATH_EXPORTS_MESH=exports/mesh/${SCENE}/${MODEL}/*.obj
        PATH_EXPORTS_PCD=exports/pcd/${SCENE}/${MODEL}/*.ply
	if [ -e $PATH_EXPORTS_MESH ] && [ -e $PATH_EXPORTS_PCD ] && ! $OVERWRITE
	then
	echo "$SCENE already exported to mesh/pcd"
	else
	echo "sh export.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1"
	sh export.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1
	fi
	fi
	done
fi



done
