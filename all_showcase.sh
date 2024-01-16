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

# disable xcb Qt
export QT_QPA_PLATFORM=offscreen

# loop through all folders
for SCENE_FOLDER in $FOLDER/*
do
SCENE=$(basename $SCENE_FOLDER)
DATASET=$FOLDER/$SCENE

if [ -e "showcase/$SCENE.gif" ] && ! $OVERWRITE; then
echo "$DATASET already showcased"
break

else

PATH_MP4=${DATASET}/*.mp4
PATH_AVI=${DATASET}/*.avi
PATH_MKV=${DATASET}/*.mkv
PATH_MOV=${DATASET}/*.MOV
PATH_WMV=${DATASET}/*.wmv
PATH_FLV=${DATASET}/*.flv
PATH_WEBM=${DATASET}/*.webm

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

ffmpeg -i $DATASET/frame_%05d.jpeg -c:v libx264 showcase/$SCENE.mp4
ffmpeg -i $DATASET/frame_%05d.jpg -c:v libx264 showcase/$SCENE.mp4
ffmpeg -i $DATASET/frame_%05d.png -c:v libx264 showcase/$SCENE.mp4
ffmpeg -y -i showcase/$SCENE.mp4 showcase/$SCENE.gif -filter_complex "fps=5,scale=480:-1"

fi

done







