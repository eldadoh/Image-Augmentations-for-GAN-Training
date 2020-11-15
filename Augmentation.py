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

# import the necessary packages
# from __future__ import print_function
import numpy as np
# import argparse
import cv2
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


# import torch
# # import kornia
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# def torchRotate(data: torch.tensor ,alpha: float = 45.0):
#     # create transformation (rotation)
#     alpha: float = 45.0  # in degrees
#     angle: torch.tensor = torch.ones(1) * alpha
#
#     # define the rotation center
#     center: torch.tensor = torch.ones(1, 2)
#     center[..., 0] = data.shape[3] / 2  # x
#     center[..., 1] = data.shape[2] / 2  # y
#
#     # define the scale factor
#     scale: torch.tensor = torch.ones(1, 2)
#
#     # compute the transformation matrix
#     M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
#
#     # apply the transformation to original image
#     _, _, h, w = data.shape
#     data_warped: torch.tensor = kornia.warp_affine(data.float(), M, dsize=(h, w))
#     return data_warped

def Augmentation(I,lam,aug_type='translationX',device='cuda'):
    # I shape: n
    I_out = torch.zeros_like(I,device=device)
    if aug_type == 'translationX':
        W = I.shape[3]
        for i in range(I.shape[0]):
            a = np.random.uniform(-lam, lam,1).item()
            num_cols = round(abs(a)*W)
            if a>=0:
                I_out[i,:,:,list(np.arange(0,num_cols))] = I[i,:,:,list(np.arange(0,num_cols))[::-1]]
                I_out[i,:,:,num_cols:] = I[i,:,:,0:(W-num_cols)]
            else:
                I_out[i,:,:,(W-num_cols):]  = I[i,:,:,list(np.arange(W-num_cols,W))[::-1]]
                I_out[i,:,:,0:(W-num_cols)] = I[i,:,:,num_cols:]
    elif aug_type == 'translationY':
        H = I.shape[2]
        # with torch.no_grad():
        for i in range(I.shape[0]):
            a = np.random.uniform(-lam, lam, 1).item()
            num_rows = round(abs(a) * H)
            if a >= 0:
                I_out[i, :, list(np.arange(0, num_rows)), :] = I[i, :, list(np.arange(0, num_rows))[::-1],:]
                I_out[i, :, num_rows:, :] = I[i, :, 0:(H - num_rows), :]
            else:
                I_out[i, :, (H - num_rows): , :] = I[i, :, list(np.arange(H - num_rows, H))[::-1], :]
                I_out[i, :, 0:(H - num_rows), :] = I[i, :, num_rows:, :]
    elif aug_type == 'colorNoise': # colorNoise
        I_out = I.clone()
        for i in range(I.shape[0]):
            for c in range(3):
                a = torch.tensor(np.random.normal(0, lam, 1).item())
                # a= torch.tensor(np.random.randint(1,int(lam*10),1))
                I_out[i, c, :, :] += a
            # I_out[i, :, :, :] **= I_out[i, :, :, :].pow(a)
            # I_out[i, :, :, :] -= 1

            # I_out[i,:,:,:] = (I[i,:,:,:]+a).clamp(-1,1)

        #     I_out[i, :, :, :]=I_out[i, :, :, :]**2
        # I_out.requires_grad = True

        # with torch.no_grad():
        #     I[1,:,:,:] = torch.tensor(torch.zeros_like(I).numpy())[1,:,:,:]

        # with torch.no_grad():
        # for i in range(I.shape[0]):
        #     I_out[i,:,:,:] =
        #     I_out = Image[i, :, :, :]
            # a = abs(np.random.normal(1, lam, 1).item())+1e-10
            # image = I[i,:,:,:]
            # for c in range(3):
                # min, max = torch.min(I[i,c,:,:]), torch.max(I[i,c,:,:])
                # imageGrey = (image[c,:,:] - min) / (max - min)
                # imageGrey = imageGrey**a
                # imageGrey = imageGrey*(max - min) + min
                # I_out[i,c,:,:] = I[i,c,:,:].pow(0.9)
                # I_out[i,:,:,:] = torch.tensor(adjust_gamma(image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy(), gamma=a),dtype=torch.float).permute(2, 0, 1).mul_(1/255)
    else:
        raise NotImplementedError()
    return I_out.to(device).clamp(-1,1)


