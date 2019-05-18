#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
#from vae import VAE
from vaecnn import VAECNN
#from util import train,test


# In[2]:


from matplotlib.pyplot import imshow
import numpy as np
import PIL
from PIL import Image


# In[3]:


transform = transforms.Compose([
     transforms.ToPILImage(),
     transforms.RandomCrop(80),
     transforms.Resize((96,96)),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(hue=.05, saturation=.05),
     transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
     transforms.ToTensor()
])


# In[4]:


def tripletLoss(anchor, positive, negative, size_average=True, margin = 0.5):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + margin)
        return losses.mean() if size_average else losses.sum()


# In[5]:


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar,z):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') #.view(-1, 784)
    BCE = F.mse_loss(recon_x, x, size_average=False)/100
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / 100
    
    anchor = z[:z.shape[0]/2]
    positive = z[z.shape[0]/2:]
    negative = torch.cat((anchor[-1:], anchor[:-1]))
#     print(anchor.shape)
#     print(positive.shape)
#     print(negative.shape)
    TL = tripletLoss(anchor, positive, negative, size_average=True) 
    
    print('BCE',BCE)
    print('KLD',KLD)
    print('TL',TL)
    
    return  BCE + 0.01 * (KLD) + 5 * TL


# In[6]:


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data_T = torch.stack([transform(d) for d in data])
        data = torch.cat( (data,data_T), 0 )
        data = data.to(device)
        #print(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar,z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar,z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar,z).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# In[7]:


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# In[8]:


args = Namespace(
    batch_size = 100,
    epochs = 20,
    cuda = True,
    seed = 2019,
    log_interval = 1)


# In[9]:


torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")


# In[10]:


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#trainDataDir = '/scratch/um367/DL/data/sample_4/train'
#valDataDir = '/scratch/um367/DL/data/sample_4/val'

trainDataDir = '/scratch/um367/DL/data/ssl_data_96/unsupervised'
valDataDir = '/scratch/um367/DL/data/ssl_data_96/supervised/val'

# trainDataDir = '/scratch/um367/DL/data/sampledata/supervised/train'
# valDataDir = '/scratch/um367/DL/data/sampledata/supervised/val'


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(trainDataDir, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valDataDir, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


# In[11]:


print(len(train_loader.dataset))


# In[12]:


model = VAECNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[13]:


## load parameters
# to load
checkpoint = torch.load('vaeSiamTest.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])


# In[ ]:


for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'vaeSiamTest.pth.tar')
#     with torch.no_grad():
#         sample = torch.randn(64, 20).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#         'results/sample_' + str(epoch) + '.png')


# In[ ]:





# In[ ]:




