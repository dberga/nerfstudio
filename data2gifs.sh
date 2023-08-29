
for FOLDER in data/*
do
for SCENE in $FOLDER/*
do
#ffmpeg -i $SCENE/images/*.MOV $SCENE/images.gif
#ffmpeg -i $SCENE/images/*.mp4 $SCENE/images.gif
#ffmpeg -i $SCENE/images/*.avi $SCENE/images.gif
#ffmpeg -i $SCENE/images/*.mkv $SCENE/images.gif
convert -monitor $SCENE/images/* $SCENE/view.gif
done
done

