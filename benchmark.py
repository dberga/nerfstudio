import argparse
import pandas as pd
#pd.options.display.max_rows = 999
#pd.options.display.max_columns = 999
import json
import os
import shutil

BENCHMARKS_FOLDER="benchmarks"
	
def find_all(name, path):
	# get all results.json files
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result
def parse_scene_path():
	# get folder
	parser = argparse.ArgumentParser()
	parser.add_argument(
	    "--output", nargs='?', type=str, default="outputs"
	)
	parser.add_argument(
	    "--scene", nargs='?', type=str, default="kitchen"
	)
	args, unknown_args = parser.parse_known_args()
	output_folder = args.output
	scene_name = args.scene
	scene_path = os.path.join(output_folder,scene_name)
	return scene_path, scene_name, output_folder

def read_json(json_path):
	with open(json_path, 'r') as f:
  		data = json.load(f)  		
	return data


if __name__ == "__main__":
	os.makedirs(BENCHMARKS_FOLDER,exist_ok=True)
	scene_path,scene_name, output_folder = parse_scene_path()
	list_results = find_all('results.json',scene_path)
	df = pd.DataFrame(columns=['experiment_name','ckpt_path','fps','fps_std','lpips','lpips_std','psnr','psnr_std','ssim','ssim_std']).set_index('experiment_name')

	for results_path in list_results:
		# read json
		results_data = read_json(results_path)
		df_results = pd.DataFrame.from_dict(results_data).transpose()
		
		# get results
		results_row = df_results.loc['results']

		# get experiment name
		model = df_results.loc['method_name'][0]
		ckpt_path = df_results.loc['checkpoint'][0]
		experiment_date = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
		experiment_name = model+":"+experiment_date
		
		# join experiment name and ckpt path to one single series
		series_results = results_row.append(pd.Series([ckpt_path],index=['ckpt_path']))
		series_results.name = experiment_name
		
		# create dataframe
		df = df.append(series_results)

	if not df.empty:
		# round decimals
		df = df.round({'fps': 4, 'lpips': 4, 'psnr': 4, 'ssim': 4})
		#reorder columns
		list_essential = ['fps','lpips','psnr','ssim']
		list_nonessential = [i for i in list(df.columns) if i not in list_essential]
		df = df.reindex(columns=list_essential+list_nonessential)
		# write .csv
		print(f"writing benchmark_{scene_name}")
		output_benchmark_csv = os.path.join(scene_path,f'benchmark_{scene_name}.csv')
		df.to_csv(output_benchmark_csv, sep=',')
		shutil.copy(output_benchmark_csv,os.path.join(BENCHMARKS_FOLDER,f'benchmark_{scene_name}.csv'))
		# write .md
		output_benchmark_md = os.path.join(BENCHMARKS_FOLDER,f'benchmark_{scene_name}.md')
		with open(output_benchmark_md, 'w') as md:
	  	    df.to_markdown(buf=md, tablefmt="grid")
