from pathlib import Path
import numpy as np
import torch

from torchvision import transforms

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

DATASETS_PATH = Path((Path(__file__).parent / '../../datasets/').resolve())
RESULTS_PATH = Path((Path(__file__).parent / '../../results/movingMNIST/').resolve())
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = Path((Path(__file__).parent / '../../weights/movingMNIST/').resolve())
WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)


MOVING_MNIST_DATASET = np.load(str(DATASETS_PATH) + '/mnist_test_seq.npy')
MOVING_MNIST_DATASET = np.transpose(MOVING_MNIST_DATASET, (1, 0, 2, 3))

try:
    MOVING_MNIST_DATASET_ENCODED = np.load(str(DATASETS_PATH) + '/mnist_encoded_seq.npz')
except:
    MOVING_MNIST_DATASET_ENCODED = None

MOVING_MNIST_DATASET_FLAT = MOVING_MNIST_DATASET.reshape((MOVING_MNIST_DATASET.shape[0]*MOVING_MNIST_DATASET.shape[1], MOVING_MNIST_DATASET.shape[2], MOVING_MNIST_DATASET.shape[3]))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., noise_factor=0.4):
        self.std = std
        self.mean = mean
        self.noise_factor = noise_factor
        
    def __call__(self, tensor):
        noisy = tensor + self.noise_factor*torch.randn(tensor.size()) * self.std + self.mean
        noisy = torch.clip(noisy, 0., 1.)
        return noisy
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

reg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

noisy_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1.)
])
