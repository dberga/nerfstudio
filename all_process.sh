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

for SCENE_FOLDER in $FOLDER/*
do
SCENE=$(basename $SCENE_FOLDER)
DATASET=$FOLDER/$SCENE

if [ -e "$DATASET/transforms*.json" ] && ! $OVERWRITE; then
echo "$DATASET already processed"
break

else
if [ -e "$DATASET/images" ]; then
TYPE="images"
elif [ -e "$DATASET/rgb" ]; then
TYPE="images"
DATASET="$DATASET/rgb"
elif [ -e "$DATASET/train" ] || [ -e "$DATASET/test" ] || [ -e "$DATASET/val" ]; then
TYPE="images"
DATASET="$DATASET/train"
elif [ -e "$DATASET/*.mp4" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.mp4 | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.avi" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.avi | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.mkv" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.mkv | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.MOV" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.MOV | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.wmv" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.wmv | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.flv" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.flv | sort -n 1 | tail -n 1)
elif [ -e "$DATASET/*.webm" ]; then
TYPE="video"
DATASET=$(ls $DATASET/*.webm | sort -n 1 | tail -n 1)
else
echo "$DATASET has no data"
break
fi

echo "sh process_data.sh $CUDA_VISIBLE_DEVICES $TYPE $DATASET"
sh process_data.sh $CUDA_VISIBLE_DEVICES $TYPE $DATASET
fi

done







