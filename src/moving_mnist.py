from pathlib import Path
import numpy as np
import torch

from torchvision import transforms

import matplotlib.pyplot as plt

from components.state_autoencoder import State_Autoencoder

results_path = Path((Path(__file__).parent / '../results/movingMNIST/').resolve())
results_path.mkdir(parents=True, exist_ok=True)
weights_path = Path((Path(__file__).parent / '../weights/movingMNIST/').resolve())
weights_path.mkdir(parents=True, exist_ok=True)


movingMNIST = np.load((Path(__file__).parent / '../datasets/mnist_test_seq.npy').resolve())
movingMNIST = np.transpose(movingMNIST, (1, 0, 2, 3))

flattened_data = movingMNIST.reshape((movingMNIST.shape[0]*movingMNIST.shape[1], movingMNIST.shape[2], movingMNIST.shape[3]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noisy = tensor + torch.randn(tensor.size()) * self.std + self.mean
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

dae = State_Autoencoder(1, 1).cuda().to(device)
optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

BATCH_SIZE = 1000
TOTAL_EPOCHS = 10
PLT_INTERVAL = 50000
SAVE_INTERVAL = 100000


fig1, (ax1) = plt.subplots(1, constrained_layout=True)
ax1.set_title('DAE PRE-TRAINING - LOSS OVER EPISODES')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Loss')


# PRETRAINING
for e in range(TOTAL_EPOCHS):
    epoch_loss = 0
    ep = 0
    print(f"TRAINING EPOCH: {e}")
    for i in range((len(flattened_data)//BATCH_SIZE)):
        #print(f"TRAINING EXAMPLES: {i*BATCH_SIZE}-{(i+1)*BATCH_SIZE}")
        state = reg_transform(flattened_data[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(device).float().unsqueeze(0).permute(2,0,1,3)

        computed_state = dae(state)
        predicted_loss = torch.nn.functional.mse_loss(computed_state, state)

        predicted_loss.backward()

        for param in dae.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()
        optim.zero_grad()

        if ep % PLT_INTERVAL == 0:
            print(f"LOSS: {predicted_loss.item()}")
            ax1.scatter((e*len(flattened_data))+ep, predicted_loss.item(), color="blue")
            fig1.savefig((str(results_path) + '/dae_pretraining/dae_loss.png'))

        epoch_loss += predicted_loss.item()
    
        if ep % SAVE_INTERVAL == 0:
            torch.save(dae.state_dict(), (str(weights_path) + f'/dae_pretraining/dae_{e}_{ep}.pth'))

            with torch.no_grad():
                idx = np.random.randint(0, 100)

                fig2, (a, b) = plt.subplots(1, 2)
                a.imshow(flattened_data[idx])
                a.set_title("Actual Image")


                state = reg_transform(flattened_data[idx]).to(device).float().unsqueeze(0)
                b.imshow(dae(state).squeeze(0).squeeze(0).squeeze(0).cpu().numpy())
                b.set_title("DAE Prediction")
                fig2.savefig((str(results_path) + f'/dae_pretraining/dae_reconstruction_{e}_{ep}.png'))

        ep += BATCH_SIZE

    print(f"EPOCH LOSS: {epoch_loss}")

dae = State_Autoencoder(1, 1).cuda().to(device)
optim = torch.optim.Adam(dae.parameters(), lr=1e-3)

BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
PLT_INTERVAL = 50000
SAVE_INTERVAL = 100000

# LOAD IN
print(weights_path)
dae.load_state_dict(torch.load((str(weights_path) + f'/dae_pretraining/dae_{9}_{20000}.pth')))
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
    for i in range((len(flattened_data)//BATCH_SIZE)):
        #print(f"TRAINING EXAMPLES: {i*BATCH_SIZE}-{(i+1)*BATCH_SIZE}")
        state = reg_transform(flattened_data[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(device).float().unsqueeze(0).permute(2,0,1,3)
        noise_state = state = noisy_transform(flattened_data[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]).to(device).float().unsqueeze(0).permute(2,0,1,3)

        computed_state = dae(noise_state)
        predicted_loss = torch.nn.functional.mse_loss(computed_state, state)

        predicted_loss.backward()

        for param in dae.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()
        optim.zero_grad()

        if ep % PLT_INTERVAL == 0:
            print(f"LOSS: {predicted_loss.item()}")
            ax1.scatter((e*len(flattened_data))+ep, predicted_loss.item(), color="blue")
            fig1.savefig((str(results_path) + '/dae_denoising/dae_loss.png'))

        epoch_loss += predicted_loss.item()
    
        if ep % SAVE_INTERVAL == 0:
            torch.save(dae.state_dict(), (str(weights_path) + f'/dae_denoising/dae_{e}_{ep}.pth'))

            with torch.no_grad():
                idx = np.random.randint(0, 100)

                fig2, (a, b, c) = plt.subplots(1, 3)
                state = flattened_data[idx]
                noise_state = noisy_transform(flattened_data[idx]).to(device).float().unsqueeze(0)

                a.imshow(state)
                a.set_title("Original Image")

                b.imshow(noise_state.squeeze(0).squeeze(0).cpu().numpy())
                b.set_title("Noisy Image")

                c.imshow(dae(noise_state).squeeze(0).squeeze(0).squeeze(0).cpu().numpy())
                c.set_title("DAE Prediction")

                fig2.savefig((str(results_path) + f'/dae_denoising/dae_reconstruction_{e}_{ep}.png'))

        ep += BATCH_SIZE

    print(f"EPOCH LOSS: {epoch_loss}")