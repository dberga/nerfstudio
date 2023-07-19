for FOLDER in data/*
do
for SCENE in $FOLDER/*
do
if ! [ -e $SCENE/transforms.json ]
then
echo "no transforms file $SCENE"
else
echo "transforms found $SCENE"
if ! [ -e $SCENE/transforms_train.json ]
then
unlink $(pwd)/$SCENE/transforms_train.json
ln -s $(pwd)/$SCENE/transforms.json $(pwd)/$SCENE/transforms_train.json
fi
if [ -e $SCENE/transforms_test.json ]
then
unlink $(pwd)/$SCENE/transforms_test.json
ln -s $(pwd)/$SCENE/transforms.json $(pwd)/transforms_test.json
fi
fi
done
done
