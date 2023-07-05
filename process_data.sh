if [ -z $1 ]
then
export CUDA_VISIBLE_DEVICES=0
else
export CUDA_VISIBLE_DEVICES=$1
fi
if [ -z $2 ]
then
export TYPE="images"
else
export TYPE=$2
fi
if [ -z $3 ]
then
export DATASET="data/nerfstudio/kitchen"
else
export DATASET=$3
fi


# RUN
ns-process-data $TYPE --data $DATASET --output-dir $DATASET --no-gpu
echo "ns-process-data $TYPE --data $DATASET --output-dir $DATASET --no-gpu" # without --no-gpu crashes

#ns-process-data images --data data/sagrada/images --output-dir data/sagrada --no-gpu;
#ns-process-data images --data data/ttorrent/aniso --output-dir data/ttorrent/aniso --no-gpu
#ns-process-data video --data data/ttorrent/aniso/0001-0120_greyback.mkv --output-dir data/ttorrent/aniso --no-gpu
#ns-process-data video --data data/rafa/zapato/IMG_8737.MOV --output-dir data/rafa/zapatos --no-gpu;
#ns-process-data video --data data/rafa/recipientes/IMG_8743.MOV --output-dir data/rafa/recipientes --no-gpu;
#ns-process-data video --data data/rafa/taza/IMG_8749.MOV --output-dir data/rafa/taza --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/amulet/objecte_de_plom__mac_ullastret.mp4 --output-dir data/sketchfab-giravolt/amulet --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/dress/vestit_de_margarida_xirgu_per_a_do_a_rosita.mp4 --output-dir data/sketchfab-giravolt/dress --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/inflammatory/encenser__museu_de_lleida.mp4 --output-dir data/sketchfab-giravolt/inflammatory --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/lucrecia/lucr_cia_morta__biblioteca_museu_v_ctor_balaguer.mp4 --output-dir data/sketchfab-giravolt/lucrecia --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/meteor/meteor__museu_de_l_empord_.mp4 --output-dir data/sketchfab-giravolt/meteor --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/neptune/nept___biblioteca_museu_v_ctor_balaguer.mp4 --output-dir data/sketchfab-giravolt/neptune --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/pietat/pietat__museu_frederic_mar_s.mp4 --output-dir data/sketchfab-giravolt/pietat --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/pipa/pipa_amb_cap_de_drac.mp4 --output-dir data/sketchfab-giravolt/pipa --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/rococo/rococo_baptismal_font.mp4 --output-dir data/sketchfab-giravolt/rococo --no-gpu;
#ns-process-data video --data data/sketchfab-giravolt/tauleta/tauleta_de_nit_modernista__museu_del_disseny.mp4 --output-dir data/sketchfab-giravolt/tauleta --no-gpu;
