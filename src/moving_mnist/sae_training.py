# SEQUENTIAL AUTO ENCODER TRAINING

from dae_utils import DATASETS_PATH, MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, MOVING_MNIST_DATASET_ENCODED, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
from components.sequential_autoencoder import Sequential_Autoencoder
import torch
import numpy as np

# HYPER-PARAMETERS
BATCH_SIZE = 100
TOTAL_EPOCHS = 3
PLT_INTERVAL = 1000
SAVE_INTERVAL = 1000
NUM_FRAMES = 1

dae = State_Autoencoder(1, 1).cuda().to(DEVICE)

# LOAD IN
dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_training/dae_{115}_{0}.pth')))
dae.eval()

#print(MOVING_MNIST_DATASET_ENCODED["encoded"].shape)
#print(MOVING_MNIST_DATASET_ENCODED["original"].shape)

'''
fig2, (a, b, c) = plt.subplots(1, 3)
with torch.no_grad():
    #print(MOVING_MNIST_DATASET_ENCODED[5].shape)
    ind = dae(reg_transform(MOVING_MNIST_DATASET_ENCODED["original"][0][0]).unsqueeze(0).to(DEVICE)).squeeze(0).squeeze(0).cpu().numpy()
    test_state = dae.decoder(torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][0]).to(DEVICE)).cpu().numpy()
    print(test_state[0].shape)
    a.imshow(ind)
    b.imshow(MOVING_MNIST_DATASET_ENCODED["original"][0][0])
    c.imshow(test_state[0].squeeze(0))
    plt.show()

exit()
'''

sae = Sequential_Autoencoder(NUM_FRAMES).to(DEVICE)
optim = torch.optim.Adam(sae.parameters(), lr=1e-3)

# DENOISING
fig1, (ax1) = plt.subplots(1, constrained_layout=True)
ax1.set_title('SAE TRAINING - LOSS OVER EPISODES')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Loss')

# LOADING PRELOADED DATA
data = np.load(str(DATASETS_PATH) + "/mnist_preloaded_encoded_1.npz")
preloaded_curr_state = torch.tensor(data["curr_state"])
preloaded_next_state = torch.tensor(data["next_state"])


for e in range(TOTAL_EPOCHS):
    epoch_loss = 0
    ep = 0
    print(f"TRAINING EPOCH: {e}")
    for i in range(len(preloaded_curr_state)//BATCH_SIZE):
        
        current_state = preloaded_curr_state[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(DEVICE)
        next_state = preloaded_next_state[i*BATCH_SIZE:(i+1)*BATCH_SIZE].to(DEVICE)
        
        optim.zero_grad()

        computed_state = sae(current_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, next_state)

        predicted_loss.backward()

        for param in sae.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()


        if ep % PLT_INTERVAL == 0:
            print(f"LOSS: {predicted_loss.item()}")
            ax1.scatter((e*len(preloaded_curr_state))+ep, predicted_loss.item(), color="blue")
            fig1.savefig((str(RESULTS_PATH) + '/sae_training/sae_loss.png'))

        epoch_loss += predicted_loss.item()
    
        if ep % SAVE_INTERVAL == 0:
            torch.save(sae.state_dict(), (str(WEIGHTS_PATH) + f'/sae_training/sae_{e}_{ep}.pth'))

            with torch.no_grad():
                ix = np.random.randint(0, len(MOVING_MNIST_DATASET_ENCODED["original"]))
                idx = np.random.randint(0, 19)

                fig2, (a, b, c) = plt.subplots(1, 3)
                a.imshow(MOVING_MNIST_DATASET_ENCODED["original"][ix][idx])
                a.set_title("Actual Current State")

                b.imshow(MOVING_MNIST_DATASET_ENCODED["original"][ix][idx+1])
                b.set_title("Actual Next State")

                #current_state = torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][ix][idx:idx+NUM_FRAMES]).to(DEVICE)
                #next_state = torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][ix][idx+1:idx+NUM_FRAMES+1]).to(DEVICE)
                c_s = torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][ix][idx:idx+NUM_FRAMES])
                n_s = torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][ix][idx+1:idx+NUM_FRAMES+1])
                current_state = torch.reshape(c_s, (1, c_s.shape[1] * c_s.shape[0], c_s.shape[2], c_s.shape[3])).to(DEVICE)
                next_state = torch.reshape(n_s, (1, n_s.shape[1] * n_s.shape[0], n_s.shape[2], n_s.shape[3])).to(DEVICE)
            
                n_out = sae(current_state)


                if NUM_FRAMES > 1:
                    f = (n_out.shape[1]+1)//NUM_FRAMES
                    n_out = n_out[0][n_out.shape[1]-f:n_out.shape[1]].unsqueeze(0)

                n_out = dae.decoder(n_out).cpu().squeeze(0).squeeze(0).numpy()

                c.imshow(n_out)
                c.set_title("Predicted Next State")

                fig2.savefig((str(RESULTS_PATH) + f'/sae_training/sae_reconstruction_{e}_{ep}.png'))
                plt.close(fig2)

        ep += BATCH_SIZE

    print(f"EPOCH LOSS: {epoch_loss}")