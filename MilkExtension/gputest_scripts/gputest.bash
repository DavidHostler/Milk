//python --version 
//conda create -n gputest python 3.8.5
conda activate tensorflow_gpuenv
pip install ipykernel
python -m  ipykernel install --user --name tensorflow_gpuenv  --display-name "tensorflow_gpuenv"
conda install tensorflow-gpu
conda install jupyter
pip install keras
jupyter-notebook