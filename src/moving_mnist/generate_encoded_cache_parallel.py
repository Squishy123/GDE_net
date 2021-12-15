from pathlib import Path
from dae_utils import MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, DATASETS_PATH, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
from components.replay_memory import ReplayMemory
import torch
import numpy as np

import sys, getopt

def main(argv):
    #dae = State_Autoencoder(1, 1).cuda().to(DEVICE)
    #optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

    BATCH_SIZE = 1000
    TOTAL_EPOCHS = 500
    PLT_INTERVAL = 50000
    SAVE_INTERVAL = 100000
    NUM_FRAMES = 5
    OUTPUT_FILE="mnist_preloaded_encoded.npz"

    try:
        opts, args = getopt.getopt(argv,"hf:o:",["numframes=","ofile="])
    except getopt.GetoptError:
        print('test.py -f <num_frames> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -f <numframes> -o <outputfile>')
            print("ERROR")
            sys.exit()
        elif opt in ("-f", "--numframes"):
            NUM_FRAMES = int(arg)
        elif opt in ("-o", "--ofile"):
            OUTPUT_FILE = "/" + str(arg)

    # LOAD IN
    #print(WEIGHTS_PATH)
    #dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_training/dae_{115}_{0}.pth')))
    #dae.eval()
    data = np.load(str(DATASETS_PATH) + "/mnist_encoded_seq.npz")

    encoded_dataset = torch.tensor(data["encoded"]).cpu()
    #original_dataset = torch.tensor(data["original"]).cpu()

    # PRECACHE CURR AND NEXT STATES FOR SAE
    preloaded_curr_state_ALL = None
    preloaded_next_state_ALL = None

    for i in range(len(encoded_dataset)//BATCH_SIZE):
        print(f"{i}/{len(encoded_dataset)//BATCH_SIZE}")

        data = encoded_dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        preloaded_curr_state=data[:,:-1]
        preloaded_curr_state = preloaded_curr_state.unfold(1, NUM_FRAMES, 1)
        preloaded_curr_state=torch.reshape(preloaded_curr_state, (preloaded_curr_state.shape[0], preloaded_curr_state.shape[1], NUM_FRAMES*64, 10, 10))

        preloaded_next_state=data[:,1:]
        preloaded_next_state = preloaded_next_state.unfold(1, NUM_FRAMES, 1)
        preloaded_next_state=torch.reshape(preloaded_next_state, (preloaded_next_state.shape[0], preloaded_next_state.shape[1], NUM_FRAMES*64, 10, 10))

        '''
        if preloaded_curr_state_ALL == None:
            preloaded_curr_state_ALL = preloaded_curr_state
        else:
            preloaded_curr_state_ALL = torch.cat((preloaded_curr_state_ALL, preloaded_curr_state), 0)

        if preloaded_next_state_ALL == None:
            preloaded_next_state_ALL = preloaded_next_state
        else:
            preloaded_next_state_ALL = torch.cat((preloaded_next_state_ALL, preloaded_next_state), 0)
        '''
        #del preloaded_next_state
        #del preloaded_curr_state
        #del data

        #print(preloaded_curr_state_ALL.shape)

        # SAVING PRELOADED DATA
        np.savez(str(DATASETS_PATH) + "/" + f"CACHE_{i}_" + OUTPUT_FILE, curr_state=preloaded_curr_state, next_state=preloaded_next_state)


if __name__ == "__main__":
    main(sys.argv[1:])