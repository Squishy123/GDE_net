install_sys:
	sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig && conda env create -f environment.yml

install_dep:
	pip install -r requirements.txt

start_board:
	mkdir -p results/runs && tensorboard --logdir=results/runs&

stop_board:
	sudo pkill tensorboard

train:
	mkdir -p results && python src/train.py

play:
	python src/play.py

decay_test:
	mkdir -p decay_test && python src/decay_test.py