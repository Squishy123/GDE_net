# GDE_net
Game Dynamics Estimator Network


## Installation Instructions
### Install Conda 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 

## ON WINDOWS:
### 1. Create Conda Environment
```
conda env create -f environment.yml
```

### 2. Activate Conda Environment
```
conda activate gde_net
```

### 3. Install Pip Requirements
```
pip install -r requirements.txt
```

## ON LINUX: 
### 1. Create Conda Environment and Install System Dependencies
```
sudo make install_sys
```

### 2. Activate and Install Pip Dependencies
```
conda activate gde_net
make install_dep
```

# Running the Project
## 1. Start TensorBoard
``` 
make start_board
```

## 2a. Run the Training Script
```
make train
```

## 2b. Run the Play Script
```
make play
```

## 3. Close TensorBoard
```
make stop_board
```