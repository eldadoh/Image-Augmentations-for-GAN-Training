# %% Imports

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import util
import fid_score
from tqdm import tqdm

cudnn.benchmark = True

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#loading the dataset
dataset = dset.CIFAR10(root="./data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
nc=3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=0)

#checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %% Configs
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# %% Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
#load weights to test the model
#netG.load_state_dict(torch.load('weights/netG_epoch_24.pth',map_location=torch.device('cpu')))
print(netG)

# %% Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
#load weights to test the model
#netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
print(netD)

# %% optimizer & Loss Config

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(128, nz, 1, 1, device=device)
fixed_noise2 = torch.randn(49920, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

niter = 25
g_loss = []
d_loss = []


# %% Train Loop
aug_type_vec   = ['colorNoise']
lambda_vec     = np.arange(0.1, 1.1, 0.1).round(2)
for aug_type in aug_type_vec:
    for lam in lambda_vec:
        print(aug_type, lam)
        util.createDir(aug_type,lam)
        util.train_loop(lam,aug_type,niter,dataloader,netG,netD,criterion,device,real_label,fake_label,nz,optimizerG,optimizerD,fixed_noise)
print('finish Train Loop')

# %% create folders
## aug_type_vec   = ['translationX']
# lambda_vec     = np.arange(0.6, 1.1, 0.1).round(2)
for aug_type in aug_type_vec:
    for lam in lambda_vec:
        print(aug_type, lam)
        folder_name = aug_type + '_' + str(lam)  # .replace('.','p')
        dirType = '/weights/'
        pathMkdir = os.getcwd() + dirType + folder_name + '/fake_samples/'
        # mkdir
        print(pathMkdir)
        try:
            os.mkdir(pathMkdir)
        except OSError:
            print("Creation of the directory %s failed" % pathMkdir)
        else:
            print("Successfully created the directory %s " % pathMkdir)
        #end mkdir

        path = os.getcwd() + dirType + folder_name
        netG = util.load_weights(netG, path=path+'/netG_epoch_23.pth', device=device)
        torch.manual_seed(42)
        for i in tqdm(range(20000)):
            fixed_noise2 = torch.randn(1, nz, 1, 1, device=device)
            fake         = netG(fixed_noise2)
            vutils.save_image(fake[0, :, :, :].detach(), path + '/fake_samples/fake_sample_%03d.png' % (i),
                              normalize=True)
print('finish create folders')

# create Real Data folder
# for i, data in enumerate(dataloader, 0):
#    real = data[0]
#    for j in range(real.shape[0]):
#        vutils.save_image(real[j, :, :, :].detach(), 'output/real_samples/real_sample_%03d.png' % (i*real.shape[0]+j), normalize=True)

# %% prepare FID
paths = ['output/real_samples', 'output/translationX/fake_samples/']
batch_size = 32
cuda = (device == 'cuda')
dims = 2048
model = fid_score.getInceptionModel(batch_size, cuda, dims)
m1, s1 = fid_score.getRealm1s1(paths, batch_size, cuda, dims, model)

print('finish prepare FID')

# %% calc FID per dir

# aug_type_vec   = ['translationX']
# lambda_vec     = np.arange(0.1, 1.1, 0.1).round(2)
matplotlib.use('Agg')
for aug_type in aug_type_vec:
    fid_vec = []
    for lam in lambda_vec:
        print(aug_type, lam)
        folder_name = aug_type + '_' + str(lam)  # .replace('.','p')
        paths = ['output/real_samples', 'weights/'+folder_name+'/fake_samples/']
        fid   = fid_score.calculate_fid_given_paths(paths, batch_size, cuda, dims, model, m1, s1)
        file = open("output/"+folder_name+"/result_FID.txt", "w")
        file.write(str(fid))
        file.close()
        fid_vec.append(fid)

    p = plt.plot(lambda_vec, np.array(fid_vec))
    plt.title(aug_type)
    plt.xlabel("lambda_aug")
    plt.ylabel("fid score")
    plt.savefig("output/"+aug_type+"_result_FID.png")
print("FINISH calc FID per dir")

# %% read FID per dir

# matplotlib.use('Agg')
# for aug_type in aug_type_vec:
#     fid_vec = []
#     for lam in lambda_vec:
#         print(aug_type, lam)
#         folder_name = aug_type + '_' + str(lam)  # .replace('.','p')
#         file = open("output/"+folder_name+"/result_FID.txt", "r")
#         fid = file.read()
#         file.close()
#         fid_vec.append(float(fid))
#         print(aug_type,fid)
#     p = plt.plot(lambda_vec, np.array(fid_vec))
#     plt.title(aug_type)
#     plt.xlabel("lambda_aug")
#     plt.ylabel("fid score")
#     plt.savefig("output/" + aug_type + "_result_FID.png")
# print("FINISH calc FID per dir")