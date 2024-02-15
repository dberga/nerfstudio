if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export EXPORTS='exports'
else
export EXPORTS=$2
fi

for SCENE in $(ls -A $EXPORTS)
do
python compare.py --exports $EXPORTS --scene $SCENE
done