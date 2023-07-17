for SCENE in outputs/*
do
for MODEL in $SCENE/*
do
for DATE in $MODEL/*
do
if [ -e $DATE/nerfstudio_models/*.ckpt ]
then
echo "$DATE checkpoint exists, all ok here"
else
echo "$DATE checkpoint does not exist, removing empty folder (and config.yml)"
rm -rf $DATE/config.yml
fi
done
done
done

