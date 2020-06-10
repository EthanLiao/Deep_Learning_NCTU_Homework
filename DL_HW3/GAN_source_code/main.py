from __future__ import print_function

#%matplotlib inline

import argparse

import glob,os

import random

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.backends.cudnn as cudnn

import torch.optim as optim

import torch.utils.data as Data

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import matplotlib.image as pltim

import argparse

from IPython.display import HTML

from Model import *

from PIL import Image





# Set random seed for reproducibility
manualSeed = 999

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)

random.seed(manualSeed)
torch.manual_seed(manualSeed)

train_data_num = 10000
beta1 = 0.5
ngpu = 1
nz = 100
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



def common_arg_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ImageFolder method
    # parser.add_argument('--dataroot', default='./img_align_celeba/', type=str)

    parser.add_argument('--dataroot', default='./img_align_celeba/1/*.jpg', type=str)

    parser.add_argument('--batch_size', default=200, type=int)

    parser.add_argument('--image_size', default=64, type=int)

    parser.add_argument('--num_epochs', default=50, type=int)

    parser.add_argument('--lr', default=0.0001, type=float)

    return parser.parse_args()


def weight_init(md):
    classname = md.__class__.__name__
    if classname.find('Conv') != -1 :
        nn.init.normal_(md.weight.data, 0.0, 0.02) # initialize the Discriminator
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(md.weight.data, 1.0, 0.02)  # initialize the Bacthnorm unit
        nn.init.constant_(md.bias.data, 0)





def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs, batch_size, lr):
    # Each epoch, we have to go through every data in dataset
    # For calculate loss, you need to create label for your data
    # construc the label for identifing real or fake data
    lable_real = 1
    lable_fake = 0
    sample_size = 64
    gen_loss_li = []  ; disc_loss_li = [] ; img_list = []
    sample_noise = torch.randn(sample_size, nz, 1, 1, device=device)
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            ##############
            # (1) update the discriminator by loss : log(D(x)) + log(1-D(G(Z)))
            # use true and fake data to train discriminator
            ##############
            # initialize gradient for network
            discriminator.zero_grad()
            # send the data into device for computation
            real_data = data[0].to(device=device, dtype=torch.float)
            batch_size = real_data.size(0)  # batch size is 20 by default
            # construct a vector to indicate that the label is a real data
            label = torch.full((batch_size,), lable_real, device=device)
            # Send data to discriminator and calculate the loss and gradient
            # view(-1) will expand all the output vector
            output = discriminator(real_data).view(-1)
            # training D(x) value
            D_x = output.mean().item()
            # calculate loss for discriminator
            loss_disc = criterion(output, label)
            loss_disc.backward()



            ## train with fake data
            ## Using Fake data, other steps are the same.
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate a batch fake data by using generator
            fake = generator(noise)
            # Send data to discriminator
            # notice : the detach() will stop gradient decent from discriminator
            output = discriminator(fake.detach()).view(-1)
            # train D(G(Z)) value
            D_G_Z = output.mean().item()
            # construct a vector to indicate that the label is a real data
            # use the original label to construct the label vector
            label.fill_(lable_fake)
            # calculate the loss and gradient
            loss_gen = criterion(output, label)
            # Update your network
            loss_gen.backward()


            optimizer_d.step()
            disc_total_loss = loss_gen + loss_disc

            ##############
            # (1) update the generator by loss : max log(D(G(z)))
            # use fake data to train discriminator
            ##############
            generator.zero_grad()
            label.fill_(lable_real)
            # generator aims to make the fake data approach to real data
            # we should minimize loss between fake data and true data
            output = discriminator(fake).view(-1)
            # train D(G(Z)) value for traing generator
            D_G_Z_GEN = output.mean().item()

            gen_loss = criterion(output,label)
            # only make generator backword
            gen_loss.backward()

            optimizer_g.step()



            if i % 50 == 0:
                # Record your loss every iteration for visualization
                print("[%d] , Disc_Loss : %.2f, Gen_Loss : %.2f " %(epoch, disc_total_loss.item(), gen_loss.item()))
            disc_loss_li.append(disc_total_loss.item())
            gen_loss_li.append(gen_loss.item())

    # Remember to save all things you need after all batches finished!!!
    # You can also use this function to save models and samples after fixed number of iteration
    sample_fake = generator(sample_noise).detach().cpu()
    img_batch = sample_fake.data.numpy().reshape((-1,64,64,3))
    img = img_batch[0].reshape(64,64,3)
    img = np.abs(img)
    plt.figure();plt.axis('off');plt.imshow(img);pltim.imsave('./result/fake_img_'+str(epoch)+'.png',img)
    plt.figure();plt.plot(disc_loss_li);plt.title('discriminator loss');plt.savefig('./result/disc_loss_batch_{:f}_lr_{:f}.png'.format(batch_size,lr))
    plt.figure();plt.plot(gen_loss_li);plt.title('generator loss');plt.savefig('./result/gen_loss_batch_{:f}_lr_{:f}.png'.format(batch_size,lr))
    torch.save(discriminator.state_dict(),'./result/disc_training_model_batch_{:f}_lr_{:f}'.format(batch_size,lr))
    torch.save(generator.state_dict(),'./result/gen_training_model_batch_{:f}_lr_{:f}'.format(batch_size,lr))
    # visualization in ipynb
    img_list.append(vutils.make_grid(sample_fake, normalize=False))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.abs(np.transpose(i,(1,2,0))), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

def main(args):

    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # ImgageFolder method
    # img = dset.ImageFolder(os.path.join(args.dataroot),
    #                         transform = transforms.Compose([
    #                         transforms.Resize(args.image_size),
    #                         transforms.CenterCrop(52),
    #                         transforms.ToTensor()
    #                         ])
    #                         )

    # customized reading file method
    filename = args.dataroot
    img_size = args.image_size
    size = (img_size,img_size)
    shuf_arr = np.arange(train_data_num)
    np.random.shuffle(shuf_arr)
    data_arr = np.zeros((train_data_num,*size,3), dtype=float)
    for i, file in enumerate(glob.glob(filename)):
        im = Image.open(file)
        im = im.convert("RGB")
        im = im.resize(size)
        data_arr[shuf_arr[i], :, :, :] = im
        if i == train_data_num -1 :
            break

    data_arr = data_arr / 225 # normalize the data

    # remember to preprocess the image by using functions in pytorch
    datatensor = torch.as_tensor(data_arr, device=device)
    dataset = Data.TensorDataset(datatensor)


    # Create the dataloader
    dataloader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size)
    # print(dataloader)

    # Create the generator and the discriminator
    # Send them to your device
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)
    # Initialize them
    generator.apply(weight_init)
    discriminator.apply(weight_init)

    # Setup optimizers for both G and D and setup criterion at the same time

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()


    # # Start training~~
    train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, args.num_epochs, args.batch_size, args.lr)





if __name__ == '__main__':
    args = common_arg_parser()
    main(args)
