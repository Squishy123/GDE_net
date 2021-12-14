# DENOISING AUTO ENCODER TRAINING

from dae_utils import MOVING_MNIST_DATASET, MOVING_MNIST_DATASET_FLAT, reg_transform, noisy_transform, DEVICE, WEIGHTS_PATH, RESULTS_PATH

import matplotlib.pyplot as plt
from components.state_autoencoder import State_Autoencoder
import torch
import numpy as np

dae = State_Autoencoder(1, 1).cuda().to(DEVICE)
optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

# HYPER-PARAMETERS
BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
PLT_INTERVAL = 50000
SAVE_INTERVAL = 100000

# LOAD IN
dae.load_state_dict(torch.load((str(WEIGHTS_PATH) + f'/dae_denoising/dae_{799}_{100000}.pth')))
dae.eval()

# DENOISING
fig1, (ax1) = plt.subplots(1, constrained_layout=True)
ax1.set_title('DAE NOISY-TRAINING - LOSS OVER EPISODES')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Loss')

for e in range(TOTAL_EPOCHS):
    epoch_loss = 0
    ep = 0
    print(f"TRAINING EPOCH: {e}")
    for i in range((len(MOVING_MNIST_DATASET_FLAT)//BATCH_SIZE)):
        state = reg_transform(MOVING_MNIST_DATASET_FLAT[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(DEVICE).float().unsqueeze(0).permute(2,0,1,3)
        noise_state = state = noisy_transform(MOVING_MNIST_DATASET_FLAT[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(DEVICE).float().unsqueeze(0).permute(2,0,1,3)

        optim.zero_grad()

        computed_state = dae(noise_state)
        predicted_loss = torch.nn.functional.mse_loss(computed_state, state)

        predicted_loss.backward()

        for param in dae.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        if ep % PLT_INTERVAL == 0:
            print(f"LOSS: {predicted_loss.item()}")
            ax1.scatter((e*len(MOVING_MNIST_DATASET_FLAT))+ep, predicted_loss.item(), color="blue")
            fig1.savefig((str(RESULTS_PATH) + '/dae_denoising/dae_loss.png'))

        epoch_loss += predicted_loss.item()
    
        if ep % SAVE_INTERVAL == 0:
            torch.save(dae.state_dict(), (str(WEIGHTS_PATH) + f'/dae_denoising/dae_{e}_{ep}.pth'))

            with torch.no_grad():
                idx = np.random.randint(0, 100)

                fig2, (a, b, c) = plt.subplots(1, 3)
                state = MOVING_MNIST_DATASET_FLAT[idx]
                noise_state = noisy_transform(MOVING_MNIST_DATASET_FLAT[idx]).to(DEVICE).float().unsqueeze(0)

                a.imshow(state)
                a.set_title("Original Image")

                b.imshow(noise_state.squeeze(0).squeeze(0).cpu().numpy())
                b.set_title("Noisy Image")

                c.imshow(dae(noise_state).squeeze(0).squeeze(0).squeeze(0).cpu().numpy())
                c.set_title("DAE Prediction")

                fig2.savefig((str(RESULTS_PATH) + f'/dae_denoising/dae_reconstruction_{e}_{ep}.png'))
                plt.close(fig2)

        ep += BATCH_SIZE

    print(f"EPOCH LOSS: {epoch_loss}")