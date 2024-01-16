for SCENE in outputs/*
do
for MODEL in $SCENE/*
do
for DATE in $MODEL/*
do
PATH_CKPT=${DATE}/nerfstudio_models/*.ckpt
if [ -e $PATH_CKPT ]
then
echo "$DATE checkpoint exists, all ok here"
else
echo "$DATE checkpoint does not exist, removing empty folder (and config.yml)"
#rm -rf $DATE/config.yml
rm -rf $DATE
fi
done
if [ -z "$(ls $MODEL)" ]
then
echo $MODEL ended up empty, removing whole folder.
rm -rf $MODEL
fi
done
if [ -z "$(ls $SCENE)" ]
then
echo $SCENE ended up empty, removing whole folder
rm -rf $SCENE
fi

if [ "$(ls -A $SCENE | wc -l)" -eq 1 ]
then
if [ -e $SCENE/benchmark_$(basename $SCENE).csv ]
then
echo $SCENE only has empty benchmark, removing whole folder
rm -rf $SCENE
fi
fi
done

