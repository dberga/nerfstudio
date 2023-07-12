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

# disable xcb Qt
export QT_QPA_PLATFORM=offscreen

# RUN
{
echo "ns-process-data $TYPE --data $DATASET --output-dir $DATASET"
ns-process-data $TYPE --data $DATASET --output-dir $DATASET
} || {
echo "ns-process-data $TYPE --data $DATASET --output-dir $DATASET --no-gpu" # without --no-gpu crashes
ns-process-data $TYPE --data $DATASET --output-dir $DATASET --no-gpu
}
