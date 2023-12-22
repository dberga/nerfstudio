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

for FOLDER in data/*; 
do
sh all_process.sh $CUDA_VISIBLE_DEVICES $FOLDER $OVERWRITE
done



