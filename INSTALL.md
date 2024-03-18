
## Installation: Setup your Nerfstudio Package drivers and dependencies

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
python -m pip install --upgrade pip setuptools
```

For installing PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 12.3) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), you need to [Install your CUDA drivers](https://developer.nvidia.com/cuda-downloads) and the CUDA toolkit prior to building `tiny-cuda-nn`:
```
sudo apt-get install cuda-toolkit
```
Then make sure your `PATH` and `LD_LIBRARY_PATH` (you add it to your terminal source in `~/.bashrc`) include your `CUDA_HOME` paths towards `/usr/local/cuda` and `/usr/local/cuda/lib64` respectively.

Packages for CUDA 11.7
```bash
python -m pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install torchvision==0.15.2+cu117 torchaudio==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
python -m pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Packages for CUDA 12 and above
```bash
python -m pip install torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install torchvision==0.16.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

conda install -c "nvidia/label/cuda-12.3.0" cuda-toolkit
python -m pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
After installing torch, install nerfstudio's requirements:
```bash
python -m pip install -r requirements.txt --force-reinstall --ignore-installed --no-deps
```

Installing your Nerfstudio package
```bash
python -m pip install -e . --ignore-installed --force-reinstall --no-deps
```
See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies) in the Installation documentation for more.

## Installation: Colmap and Hloc data preprocessing (SfM tools) for GPU

Install prior dependencies:
```bash
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```
Build and compile Colmap (make sure you are back to base environment)
```bash
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
sudo chown -R $(whoami) .
ninja
sudo ninja install
```
Install Hloc (make sure you are in your nerfstudio's environment)
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization
python -m pip install -e .
```
See the [Colmap Installation](https://colmap.github.io/install.html) and [Hloc Installation](https://github.com/cvg/Hierarchical-Localization#installation) documentation for more.
## Installation: ffmpeg for rendering
```bash
sudo apt install ffmpeg
```
