from pathlib import Path
from nerfstudio.process_data.colmap_utils import colmap_to_json

inputs=Path("sparse")
output=Path(".")

colmap_to_json(inputs,output)