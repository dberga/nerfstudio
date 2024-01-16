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


if [ "${MODEL}" = "generfacto" ] # for model generation
then
	export DATASET=$SCENE
	echo "sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1 $OVERWRITE"
	sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1 $OVERWRITE
else # for any nerf model
	for FOLDER in data/*;
	do
	PACK=$(basename $FOLDER)
	if [ -e data/$PACK/$SCENE ]; then
	DATASET=data/$PACK/$SCENE
	echo "sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1 $OVERWRITE"
	sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1 $OVERWRITE
	fi
	done
fi


##################### old: force existing camera path
: '
for FOLDER in data/*;
do
PACK=$(basename $FOLDER)
if [ -d "data/$PACK/$SCENE/camera_paths" ];
then
PATH_SUB_CAMERA=data/${PACK}/${SCENE}/camera_paths/*.json
if [ -e $PATH_SUB_CAMERA ];
then
DATASET=data/$PACK/$SCENE
PATH_SUB_MP4=renders/${SCENE}/*/*.mp4
if [ -e $PATH_SUB_MP4 ]
then
echo "$SCENE already rendered"
else
echo "sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1"
#sh render.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET 1
fi
fi
fi
done
'

done

