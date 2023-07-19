if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi

for FOLDER in data/*; 
do
sh all_process.sh $CUDA_VISIBLE_DEVICES $FOLDER
done



