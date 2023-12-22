if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export OVERWRITE=false
else
export OVERWRITE=$2
fi

ALL_CKPTS=$(ls outputs/*/*/*/*/*.ckpt)
for CKPT in $ALL_CKPTS
do

DATE=$(basename $(dirname $(dirname $CKPT)))
MODEL=$(basename $(dirname $(dirname $(dirname $CKPT))))
SCENE=$(basename $(dirname $(dirname $(dirname $(dirname $CKPT)))))

for FOLDER in data/*;
do
PACK=$(basename $FOLDER)
if [ -d data/$PACK/$SCENE ];
then
DATASET=data/$PACK/$SCENE
MODEL_PATH=$(dirname $(dirname $CKPT))
OUTPUT_PATH=$MODEL_PATH/results.json
if [ -e "$OUTPUT_PATH/results.json" ] && ! $OVERWRITE
then
echo "$OUTPUT_PATH/results.json already exists"
else
echo "sh eval.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET"
sh eval.sh $CUDA_VISIBLE_DEVICES $MODEL $DATASET
fi
fi
done

done
