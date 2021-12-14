# SEQUENTIAL AUTO ENCODER TRAINING

from dae_utils import MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, MOVING_MNIST_DATASET_ENCODED, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
from components.sequential_autoencoder import Sequential_Autoencoder
import torch
import numpy as np

dae = State_Autoencoder(1, 1).cuda().to(DEVICE)

# HYPER-PARAMETERS
BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
PLT_INTERVAL = 50000
SAVE_INTERVAL = 100000

# LOAD IN
dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_denoising/dae_{699}_{100000}.pth')))
dae.eval()

print(MOVING_MNIST_DATASET_ENCODED["encoded"].shape)
print(MOVING_MNIST_DATASET_ENCODED["original"].shape)

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

exit()
'''

sae = Sequential_Autoencoder().to(DEVICE)
optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

# DENOISING
fig1, (ax1) = plt.subplots(1, constrained_layout=True)
ax1.set_title('SAE TRAINING - LOSS OVER EPISODES')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Loss')

for e in range(TOTAL_EPOCHS):
    epoch_loss = 0
    ep = 0
    print(f"TRAINING EPOCH: {e}")
    for i in range((len(MOVING_MNIST_DATASET_ENCODED["encoded"])//BATCH_SIZE)):
        state = torch.tensor(MOVING_MNIST_DATASET_ENCODED["encoded"][(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(DEVICE)
        state = torch.flatten(state, 2).unsqueeze(3)

        print(state.shape)

        optim.zero_grad()

        computed_state = sae(state)
        print(computed_state.shape)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, state)

        predicted_loss.backward()

        for param in dae.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        if ep % PLT_INTERVAL == 0:
            print(f"LOSS: {predicted_loss.item()}")
            ax1.scatter((e*len(MOVING_MNIST_DATASET_ENCODED["encoded"]))+ep, predicted_loss.item(), color="blue")
            fig1.savefig((str(RESULTS_PATH) + '/sae_training/sae_loss.png'))

        epoch_loss += predicted_loss.item()
    
        if ep % SAVE_INTERVAL == 0:
            torch.save(dae.state_dict(), (str(WEIGHTS_PATH) + f'/sae_training/sae_{e}_{ep}.pth'))

            with torch.no_grad():
                idx = np.random.randint(0, 100)

                fig2, (a, b) = plt.subplots(1, 2)
                a.imshow(MOVING_MNIST_DATASET_ENCODED["original"][idx].squeeze(0))
                a.set_title("Actual Image")


                state = reg_transform(MOVING_MNIST_DATASET_ENCODED["encoded"][idx]).to(DEVICE).float().unsqueeze(0)
                b.imshow(dae(state).squeeze(0).squeeze(0).squeeze(0).cpu().numpy())
                b.set_title("SAE Prediction")
                fig2.savefig((str(RESULTS_PATH) + f'/sae_training/sae_reconstruction_{e}_{ep}.png'))
                plt.close(fig2)

        ep += BATCH_SIZE

    print(f"EPOCH LOSS: {epoch_loss}")