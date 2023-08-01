if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export OUTPUT='outputs'
else
export OUTPUT=$2
fi

for SCENE in $(ls -A $OUTPUT)
do
python benchmark.py --output $OUTPUT --scene $SCENE
done
