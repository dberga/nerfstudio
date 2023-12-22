for FOLDER in data/*
do
for SCENE in $FOLDER/*
do

if ! [ -e $SCENE/transforms.json ]
then
echo "no transforms file $SCENE, trying to link"

if [ -f $SCENE/transforms_train.json ]
then
ln -s $(pwd)/$SCENE/transforms_train.json $(pwd)/$SCENE/transforms.json
fi

if ! [ -e $SCENE/*.json ];
then
echo "no json to link for $SCENE"
else
JSON=$(ls $SCENE/*.json | sort -n | tail -n 1)
echo $JSON
fi

if [ -L $JSON ]
then
unlink $JSON
fi
if [ -L $(pwd)/$SCENE/transforms.json ]
then
unlink $(pwd)/$SCENE/transforms.json
fi
if [ -f $JSON ]
then
echo "linking $JSON to $(pwd)/$SCENE/transforms.json"
ln -s $JSON $(pwd)/$SCENE/transforms.json
fi

else
echo "transforms found $SCENE"
fi

done
done

