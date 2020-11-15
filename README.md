# Image-Augmentations-for-GAN-Training

1. Structure of the code:
	folders hyrarchy:
	./src/					- main source code (explanation next)
	./src/data/				- CIFAR10 dataset place holder (automatically to be downloaded from web)
	./src/FID/				- FID calculation files
	./src/weigts/			- weigts place holder (automatically to be created when run)
	./src/outputs/			- outputs place holder (automatically to be created when run)

	
	1.1 ./src/FID/ : FID Calculaion files.
		1.1.1 fid_score.py - fid score calculation, operates using calculate_fid_given_paths function
							 main functions:
											def getInceptionModel			- prepare for FID calc
											def getRealm1s1					- prepare for FID calc
											def calculate_fid_given_paths	- FID calc
		1.1.2 inception.py - generation of InceptionV3 model used as part of fid score calculation
		
	1.2 ./src/FID/Augmentation.py : augmentation file. 
									implementation of translationX, translationY, colorNoise
									def Augmentation(I,lam,aug_type='translationX',device='cuda'):

	1.3 ./src/FID/util.py : utility functions.
									def load_weights(netG, path ='weights/netG_epoch_24.pth',device='cpu'):
									def createDir(aug_type, lam)
									def train_loop(..) - the training loop over batches and epoches for training of Generator and Discriminator 
														 with augmentation.
	1.4 ./src/FID/test.py : experiment file process:
							# %% Imports
							# %% Configs.
							# %% Generator
							# %% Discriminator
							# %% optimizer & Loss Config
							# %% Train Loop (over aug_types , lambda vals): aug_type_vec   = ['colorNoise','gammaCor','translationY'], lambda_vec = np.arange(0.1, 1.1, 0.1).round(2)
							# %% create folders - load wights (for aug type and lmbda val) and produce directory with results.
							# %% prepare FID - prepare Inception model and m1, s1 value based on real data
							# %% calc FID per dir 
	1.5 py-sbatch.sh, jupyter-lab.sh  - file to create job process to run in the CS servers.
	1.6 gan_cifar.py, gan_cifar.ipynb - files for manully further research. not relevant for reproduce resutls. 
	
		
2. steps to reproduce the results:
	2.1 run in tesy.py lines: 165, decide augmentation and lambda values. default -  all of them (be carefull it's long!)
	2.2 procedure to run test.py on the servers: 
		- conda activate cs236781-hw
		- cd FINAL/DCGAN_CIFAR10/
		- srun -c 2 --gres=gpu:1 --pty python -m test.py
	2.3 if you want to check the net manualy in the jupyter lab, you can use the notebook gan_cifar.ipynb
		- srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
			
