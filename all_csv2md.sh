for SCENE in outputs/*/benchmark_*.csv
do
python convert_csv2md.py -i $SCENE -o $SCENE.md
done
