install_sys:
	sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig && conda env create -f environment.yml

install_dep:
	pip install -r requirements.txt

start_board:
	mkdir -p experiments/runs && tensorboard --logdir=experiments/runs&

stop_board:
	sudo pkill tensorboard

train:
	mkdir -p experiments && python src/app/train.py

play:
	python src/app/play.py

decay_test:
	mkdir -p decay_test && python src/decay_test.py

download_datasets:
	wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy downloads/

setup: 
	mkdir -p results/movingMNIST/dae_training && mkdir -p weights/movingMNIST/dae_training && mkdir -p results/movingMNIST/dae_denoising && mkdir -p weights/movingMNIST/dae_denoising && mkdir -p results/movingMNIST/sae_training && mkdir -p weights/movingMNIST/sae_training