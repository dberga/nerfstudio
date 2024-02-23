if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export TYPE="images"
else
export TYPE=$2
fi
if [ -z $3 ]
then
export DATASET="data/nerfstudio/kitchen"
else
export DATASET=$3
fi
if [ -z $4 ]
then
export OUTPUT_DIR=$DATASET
else
export OUTPUT_DIR=$4
fi
if [ -z $5 ]
then
export SFM=colmap #hloc
else
export SFM=$5
fi

# disable xcb Qt
export QT_QPA_PLATFORM=offscreen

# RUN
{
echo "ns-process-data $TYPE --data $DATASET --output-dir $OUTPUT_DIR"
ns-process-data $TYPE --data $DATASET --output-dir $OUTPUT_DIR --sfm_tool $SFM
} || {
echo "ns-process-data $TYPE --data $DATASET --output-dir $OUTPUT_DIR --no-gpu" # without --no-gpu crashes
ns-process-data $TYPE --data $DATASET --output-dir $OUTPUT_DIR --no-gpu --sfm_tool $SFM
}
