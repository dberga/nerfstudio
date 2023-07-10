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



ALL_CKPTS=$(ls outputs/*/*/*/*/*.ckpt)
for CKPT in $ALL_CKPTS
do

DATE=$(basename $(dirname $(dirname $CKPT)))
MODEL=$(basename $(dirname $(dirname $(dirname $CKPT))))
SCENE=$(basename $(dirname $(dirname $(dirname $(dirname $CKPT)))))

#echo "$DATE"
#echo "$MODEL"
#echo "$SCENE"

for FOLDER in data/*;
do
PACK=$(basename $FOLDER)
if [ -d data/$PACK/$SCENE/camera_paths ];
then
if [ -e data/$PACK/$SCENE/camera_paths/*.json ];
then
DATASET=data/$PACK/$SCENE
if [ -e renders/$SCENE/*.mp4 ]
then
echo "$SCENE already rendered"
else
echo "sh render.sh $1 $MODEL $DATASET 1"
sh render.sh $1 $MODEL $DATASET 1
fi
fi
fi
done


done