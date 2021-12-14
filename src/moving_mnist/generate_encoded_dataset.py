from pathlib import Path
from dae_utils import MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, DATASETS_PATH, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
import torch
import numpy as np

dae = State_Autoencoder(1, 1).cuda().to(DEVICE)
optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
PLT_INTERVAL = 50000
SAVE_INTERVAL = 100000

# LOAD IN
#print(WEIGHTS_PATH)
dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_denoising/dae_{699}_{100000}.pth')))
dae.eval()

encoded_dataset = None
original_dataset = None

#for i in range((MOVING_MNIST_DATASET.shape[0])):
for i in range(1000):
    if i % 1000 == 0:
        print(i)

    with torch.no_grad():
        state = reg_transform(MOVING_MNIST_DATASET[i]).to(DEVICE).float().unsqueeze(0).permute(2,0,3,1)
        computed_state = dae.encoder(state).unsqueeze(0).cpu()

        if encoded_dataset == None:
            encoded_dataset = computed_state
            original_dataset = torch.tensor(MOVING_MNIST_DATASET[i]).unsqueeze(0)
        else:
            encoded_dataset = torch.cat((encoded_dataset, computed_state), 0)
            original_dataset = torch.cat((original_dataset, torch.tensor(MOVING_MNIST_DATASET[i]).unsqueeze(0)), 0)

np.savez(str(DATASETS_PATH) + '/mnist_encoded_seq.npz', encoded=encoded_dataset.numpy(), original=original_dataset.numpy())
