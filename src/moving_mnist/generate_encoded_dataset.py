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
NUM_FRAMES = 2

# LOAD IN
#print(WEIGHTS_PATH)
dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_training/dae_{17}_{0}.pth')))
dae.eval()

with torch.no_grad():
    encoded_dataset = None
    original_dataset = None

    for i in range((MOVING_MNIST_DATASET.shape[0])):
    #for i in range(100):
        if i % 1000 == 0:
            print(f"{i}/{MOVING_MNIST_DATASET.shape[0]}")

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

    # PRECACHE CURR AND NEXT STATES FOR SAE
    preloaded_curr_state = None
    preloaded_next_state = None

    for i in range(len(encoded_dataset)):
        if i % 1000 == 0:
            print(f"{i}/{len(encoded_dataset)}")

        g = 0 
        ep_loss = 0
        while g < encoded_dataset.shape[1]-2:
                
            #exit()

            #state = torch.tensor(encoded_dataset["encoded"][(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(DEVICE)
            #state = torch.flatten(state, 2).unsqueeze(3)
            current_state = torch.reshape(encoded_dataset[i][g:g+NUM_FRAMES], (1, encoded_dataset[i][g:g+NUM_FRAMES].shape[1] * encoded_dataset[i][g:g+NUM_FRAMES].shape[0], encoded_dataset[i][g:g+NUM_FRAMES].shape[2], encoded_dataset[i][g:g+NUM_FRAMES].shape[3]))
            next_state = torch.reshape(encoded_dataset[i][g+1:g+NUM_FRAMES+1], (1, encoded_dataset[i][g+1:g+NUM_FRAMES+1].shape[1] * encoded_dataset[i][g+1:g+NUM_FRAMES+1].shape[0], encoded_dataset[i][g+1:g+NUM_FRAMES+1].shape[2], encoded_dataset[i][g+1:g+NUM_FRAMES+1].shape[3]))
            
            if preloaded_curr_state == None:
                preloaded_curr_state = current_state.cpu()
            else:
                preloaded_curr_state = torch.cat((preloaded_curr_state.cpu(), current_state.cpu()), 0)

            if preloaded_next_state == None:
                preloaded_next_state = next_state.cpu()
            else:
                preloaded_next_state = torch.cat((preloaded_next_state.cpu(), next_state.cpu()), 0)

            g += NUM_FRAMES

    # SAVING PRELOADED DATA
    np.savez(str(DATASETS_PATH) + "/mnist_preloaded_encoded.npz", curr_state=preloaded_curr_state, next_state=preloaded_next_state)
