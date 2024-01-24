for FOLDER in data/*
do
for SCENE_FOLDER in $FOLDER/*
do

if [ ! -d "${SCENE_FOLDER}" ]; then 
continue;
fi

TRANSFORMS_PATH=$SCENE_FOLDER/transforms.json
TRANSFORMS_TRAIN_PATH=$SCENE_FOLDER/transforms_train.json

if [ ! -e $TRANSFORMS_PATH ]
then
echo "no transforms file $SCENE_FOLDER, trying to link"

if [ -L $TRANSFORMS_PATH ]; then
unlink $TRANSFORMS_PATH
fi

if [ -e $TRANSFORMS_TRAIN_PATH ]
then
echo "linking transforms.json to ${TRANSFORMS_TRAIN_PATH}"
ln -s transforms_train.json $TRANSFORMS_PATH
else
echo "${TRANSFORMS_TRAIN_PATH} does not exist"
fi

fi
done
done


