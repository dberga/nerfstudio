
for FOLDER in data/*
do
for SCENE in $FOLDER/*
do
for VIDEO in $SCENE/*.MOV $SCENE/*.mp4 $SCENE/*.mkv
do
ffmpeg -i $VIDEO $SCENE/$(basename $VIDEO).gif -filter_complex "fps=15,scale=480:-1" -y
done

ffmpeg -pattern_type glob -i $SCENE/images/* $SCENE/images.gif -y "fps=15,scale=480:-1"
convert -monitor $SCENE/images/* $SCENE/view.gif -delay 7

#ffmpeg -y -i $SCENE/*.MOV $SCENE/video.gif -filter_complex "fps=15,scale=480:-1"
done
done
rsync -a --prune-empty-dirs --include '*/' --include '*.gif' --exclude '*' data/* data_gifs --progress

