from pathlib import Path
from dae_utils import MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, DATASETS_PATH, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
import torch
import numpy as np

import sys, getopt

def main(argv):
    dae = State_Autoencoder(1, 1).cuda().to(DEVICE)
    optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

    BATCH_SIZE = 1000
    TOTAL_EPOCHS = 500
    PLT_INTERVAL = 50000
    SAVE_INTERVAL = 100000
    NUM_FRAMES = 1
    OUTPUT_FILE="/mnist_preloaded_encoded.npz"

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
    dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_training/dae_{115}_{0}.pth')))
    dae.eval()

    with torch.no_grad():
        data = np.load(str(DATASETS_PATH) + "/mnist_encoded_seq.npz")
        encoded_dataset = torch.tensor(data["encoded"])
        original_dataset = torch.tensor(data["original"])

        # PRECACHE CURR AND NEXT STATES FOR SAE
        preloaded_curr_state = None
        preloaded_next_state = None

        for i in range(100):
            if i % 1000 == 0:
                print(f"{i}/{len(encoded_dataset)}")

            g = 0 
            ep_loss = 0
            while g + NUM_FRAMES < encoded_dataset.shape[1]-1:
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

                g += 1

        # SAVING PRELOADED DATA
        np.savez(str(DATASETS_PATH) + OUTPUT_FILE, curr_state=preloaded_curr_state, next_state=preloaded_next_state)


if __name__ == "__main__":
    main(sys.argv[1:])