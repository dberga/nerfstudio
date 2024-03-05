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
export OVERWRITE=false
else
export OVERWRITE=$3
fi
if [ -z $4 ]
then
export SFM=colmap
else
export SFM=$4 #colmap or hloc
fi

for SCENE_FOLDER in $FOLDER/*
do
SCENE=$(basename $SCENE_FOLDER)
DATASET=$FOLDER/$SCENE
OUTPUT_DIR=$DATASET

TRANSFORMS=${DATASET}/transforms.json
TRANSFORMS_TRAIN=${DATASET}/transforms_train.json
TRANSFORMS_TEST=${DATASET}/transforms_test.json
TRANSFORMS_VAL=${DATASET}/transforms_val.json

# ([ -e $TRANSFORMS ] || [ -e $TRANSFORMS_TRAIN ] || [ -e $TRANSFORMS_TEST ] || [ -e $TRANSFORMS_VAL ] )
if [ -e $TRANSFORMS ] && ! $OVERWRITE; then
echo "$DATASET already processed"
continue

else

if [ ! -d $DATASET ]; then
continue
fi

# possible video paths
PATH_MP4=${DATASET}/*.mp4
PATH_AVI=${DATASET}/*.avi
PATH_MKV=${DATASET}/*.mkv
PATH_MOV=${DATASET}/*.mov
PATH_WMV=${DATASET}/*.wmv
PATH_FLV=${DATASET}/*.flv
PATH_WEBM=${DATASET}/*.webm

# check data dir for images or video
if [ -e "$DATASET/images" ]; then
TYPE="images"
DATASET="$DATASET/images"
elif [ -e "$DATASET/rgb" ]; then
TYPE="images"
DATASET="$DATASET/rgb"
elif [ -e "$DATASET/image" ]; then
TYPE="images"
DATASET="$DATASET/image"
elif [ -e "$DATASET/train" ] || [ -e "$DATASET/test" ] || [ -e "$DATASET/val" ]; then
TYPE="images"
DATASET="$DATASET/train"
elif [ -e $PATH_MP4 ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.mp4 | sort -n 1 | tail -n 1)
elif [ -e $PATH_AVI ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.avi | sort -n 1 | tail -n 1)
elif [ -e $PATH_MKV ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.mkv | sort -n 1 | tail -n 1)
elif [ -e $PATH_MOV ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.MOV | sort -n 1 | tail -n 1)
elif [ -e $PATH_WMV ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.wmv | sort -n 1 | tail -n 1)
elif [ -e $PATH_FLV ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.flv | sort -n 1 | tail -n 1)
elif [ -e $PATH_WEBM ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.webm | sort -n 1 | tail -n 1)
else
echo "$DATASET has no data"
break
fi

echo "sh process_data.sh $CUDA_VISIBLE_DEVICES $TYPE $DATASET $SFM $OUTPUT_DIR"
sh process_data.sh $CUDA_VISIBLE_DEVICES $TYPE $DATASET $SFM $OUTPUT_DIR
fi

done







