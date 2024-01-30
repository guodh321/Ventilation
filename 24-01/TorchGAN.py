#!/usr/bin/env python
# coding: utf-8

# ## Initial settings

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{2}"


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

import sys
sys.path.append(".")
import tools as t


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[3]:


from tensorflow import keras
import tensorflow as tf


# In[4]:


root_path = "/home/dg321/gitTest/PRI/irp/Ventilation/24-01"


# ## Load data

# In[5]:


rawdata = pd.read_csv(root_path + "/AHU Data 20230410.csv")


# In[6]:


print(rawdata.shape)
print(rawdata.columns)
print(rawdata.head())


# In[7]:


print(rawdata["Timestamp"][0])
print(rawdata["Timestamp"][862])


# In[8]:


rawdata['Timestamp'] = pd.to_datetime(rawdata['Timestamp'])


# In[9]:


# Convert timestamp to seconds
timestamp_seconds = (rawdata['Timestamp'] - rawdata['Timestamp'].min()).dt.total_seconds()

# Calculate time sin and time cos
time_sin = np.sin(2 * np.pi * timestamp_seconds / (24 * 60 * 60))
time_cos = np.cos(2 * np.pi * timestamp_seconds / (24 * 60 * 60))

rawdata.insert(1, 'time_sin', time_sin)
rawdata.insert(2, 'time_cos', time_cos)


# In[10]:


print(rawdata.info())


# In[11]:


column_names_list = rawdata.columns.tolist()


# ## Preprocessing

# In[12]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

data_reducedcols = rawdata.iloc[:, 1:-2]

# Standardize the dataset
scaler = MinMaxScaler()

# Scale the data
scaled_data = scaler.fit_transform(data_reducedcols)

# Save the scaler object
joblib.dump(scaler, root_path + '/scaler.pkl')


# In[13]:


# Choose the first 80% samples as training data
train_samples = int(len(scaled_data) * 0.8)
data_train = scaled_data[:train_samples]
data_test = scaled_data[train_samples:]

train_samples_2 = int(len(data_train) * 0.9)
data_val = data_train[train_samples_2:]
train_data = data_train[:train_samples_2]


# ## TorchGAN

# In[14]:


try:
    import torchgan

    print(f"Existing TorchGAN {torchgan.__version__} installation found")
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchgan"])
    import torchgan

    print(f"Installed TorchGAN {torchgan.__version__}")


# In[15]:


# General Imports
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

# Pytorch and Torchvision Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Torchgan Imports
import torchgan.models as models
import torchgan.losses as losses
from torchgan.trainer import Trainer


# In[16]:


# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random Seed: ", manualSeed)


# ### Create dataset and dataloader

# In[17]:


print(data_train.shape)


# In[18]:


all_values = data_train[:, :8]
ncoeffs = all_values.shape[1]
print(ncoeffs)
ntimes = 8
BATCH_SIZE = 64
step = 1


# In[19]:


train_ct = t.concat_timesteps(all_values, ntimes, step)
val_ct = t.concat_timesteps(data_val, ntimes, step)
test_ct = t.concat_timesteps(data_test, ntimes, step)


# In[20]:


print("Type of train_ct: ", type(train_ct))
print("Shape of train_ct: ", train_ct.shape)


# In[21]:


from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Resize, ToPILImage, ToTensor
import torch

# create dataset
# Convert numpy arrays to PyTorch tensors
train_ct_torch = torch.from_numpy(train_ct).float()
val_ct_torch = torch.from_numpy(val_ct).float()


class ResizedDataset(Dataset):
    def __init__(self, dataset, output_size):
        self.dataset = dataset
        self.output_size = output_size
        self.transform = transforms.Compose([
            ToPILImage(),
            Resize(output_size),
            ToTensor()
        ])

    def __getitem__(self, index):
        img, = self.dataset[index]
        return self.transform(img)

    def __len__(self):
        return len(self.dataset)

# Create your original dataset
train_dataset = TensorDataset(train_ct_torch)

# Create the resized dataset
resized_dataset = ResizedDataset(train_dataset, (32, 32))


# In[22]:


# Create DataLoaders
loader = data.DataLoader(resized_dataset, batch_size=64, shuffle=True)


# In[23]:


# dataset = dsets.MNIST(
#     root="./mnist",
#     train=True,
#     transform=transforms.Compose(
#         [
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,), std=(0.5,)),
#         ]
#     ),
#     download=True,
# )

# # loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# loader = data.DataLoader(dataset, batch_size=32, shuffle=True)


# In[24]:


class AdversarialAutoencoderGenerator(models.Generator):
    def __init__(
        self,
        encoding_dims,
        input_size,
        input_channels,
        step_channels=32,
        nonlinearity=nn.LeakyReLU(0.2),
    ):
        super(AdversarialAutoencoderGenerator, self).__init__(encoding_dims)
        encoder = [
            nn.Sequential(
                nn.Conv2d(input_channels, step_channels, 5, 2, 2), nonlinearity
            )
        ]
        size = input_size // 2
        channels = step_channels
        while size > 1:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 4, 5, 4, 2),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size = size // 4
        self.encoder = nn.Sequential(*encoder)
        self.encoder_fc = nn.Linear(
            channels, encoding_dims
        )  # Can add a Tanh nonlinearity if training is unstable as noise prior is Gaussian
        self.decoder_fc = nn.Linear(encoding_dims, step_channels)
        decoder = []
        size = 1
        channels = step_channels
        while size < input_size // 2:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels * 4, 5, 4, 2, 3),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size *= 4
        decoder.append(nn.ConvTranspose2d(channels, input_channels, 5, 2, 2, 1))
        self.decoder = nn.Sequential(*decoder)

    def sample(self, noise):
        noise = self.decoder_fc(noise)
        noise = noise.view(-1, noise.size(1), 1, 1)
        return self.decoder(noise)

    def forward(self, x):
        if self.training:
            encoding = self.encoder(x)
            encoding = self.encoder_fc(
                encoding.view(
                    -1, encoding.size(1) * encoding.size(2) * encoding.size(3)
                )
            )
            return self.sample(encoding), encoding
        else:
            return self.sample(x)


# In[25]:


class AdversarialAutoencoderDiscriminator(models.Discriminator):
    def __init__(self, input_dims, nonlinearity=nn.LeakyReLU(0.2)):
        super(AdversarialAutoencoderDiscriminator, self).__init__(input_dims)
        model = [nn.Sequential(nn.Linear(input_dims, input_dims // 2), nonlinearity)]
        size = input_dims // 2
        while size > 16:
            model.append(
                nn.Sequential(
                    nn.Linear(size, size // 2), nn.BatchNorm1d(size // 2), nonlinearity
                )
            )
            size = size // 2
        model.append(nn.Linear(size, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# In[26]:


class AdversarialAutoencoderGeneratorLoss(losses.GeneratorLoss):
    def forward(self, real_inputs, gen_inputs, dgz):
        loss = 0.999 * F.mse_loss(gen_inputs, real_inputs)
        target = torch.ones_like(dgz)
        loss += 0.001 * F.binary_cross_entropy_with_logits(dgz, target)
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        recon, encodings = generator(real_inputs)
        optimizer_generator.zero_grad()
        dgz = discriminator(encodings)
        loss = self.forward(real_inputs, recon, dgz)
        loss.backward()
        optimizer_generator.step()
        return loss.item()


class AdversarialAutoencoderDiscriminatorLoss(losses.DiscriminatorLoss):
    def forward(self, dx, dgz):
        target_real = torch.ones_like(dx)
        target_fake = torch.zeros_like(dx)
        loss = 0.5 * F.binary_cross_entropy_with_logits(dx, target_real)
        loss += 0.5 * F.binary_cross_entropy_with_logits(dgz, target_fake)
        return loss

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        batch_size,
        labels=None,
    ):
        _, encodings = generator(real_inputs)
        noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
        optimizer_discriminator.zero_grad()
        dx = discriminator(noise)
        dgz = discriminator(encodings)
        loss = self.forward(dx, dgz)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()


losses = [
    AdversarialAutoencoderGeneratorLoss(),
    AdversarialAutoencoderDiscriminatorLoss(),
]


# In[27]:


network = {
    "generator": {
        "name": AdversarialAutoencoderGenerator,
        "args": {"encoding_dims": 128, "input_size": 32, "input_channels": 1},
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
    "discriminator": {
        "name": AdversarialAutoencoderDiscriminator,
        "args": {"input_dims": 128,},
        "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
    },
}


# In[28]:


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # Use deterministic cudnn algorithms
    torch.backends.cudnn.deterministic = True
    epochs = 10000
else:
    device = torch.device("cpu")
    epochs = 5

print("Device: {}".format(device))
print("Epochs: {}".format(epochs))


# In[59]:


trainer = Trainer(network, losses, sample_size=64, epochs=epochs, device=device)


# In[131]:


trainer(loader)

