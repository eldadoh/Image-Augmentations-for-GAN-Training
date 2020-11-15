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
import Augmentation as aug

def load_weights(netG, path ='weights/netG_epoch_24.pth',device='cpu'):
    netG.load_state_dict(torch.load(path,map_location=torch.device(device)))
    return netG


def createDir(aug_type, lam):
    folder_name = aug_type + '_' + str(lam)  # .replace('.','p')
    dirTypes = ['/output/' , '/weights/']
    for dirType in dirTypes:
        path = os.getcwd() + dirType + folder_name
        print(path)
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

def train_loop(lam,aug_type,niter,dataloader,netG,netD,criterion,device,real_label,fake_label,nz,optimizerG,optimizerD,fixed_noise):
    folder_name = aug_type+'_'+str(lam)
    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            real_cpu = aug.Augmentation(real_cpu,lam,aug_type,device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake = aug.Augmentation(fake, lam, aug_type,device=device)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
            epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # save the output
            if i % 100 == 0:
                print('saving the output')
                vutils.save_image(real_cpu,  'output/'+folder_name+'/real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), 'output/'+folder_name+'/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

        # Check pointing for every epoch
        torch.save(netG.state_dict(), 'weights/'+folder_name+'/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weights/'+folder_name+'/netD_epoch_%d.pth' % (epoch))

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
        #     epoch, niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


