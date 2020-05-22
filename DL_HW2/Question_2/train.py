from PIL import Image
import numpy as np
import glob,os,math
from VAE import VAE
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Preprocessing data
filename = "/home/mint/Desktop/Data_Set/Deep_Learning_NCTU_Homework/DL_HW2/Question_2/data/*.png"
size = (32,32)
train_data_num = 12000
shuf_arr = np.arange(train_data_num)
np.random.shuffle(shuf_arr)
data_arr = np.zeros((train_data_num,32,32,3),dtype = float)

for i,file in enumerate(glob.glob(filename)):
    im = Image.open(file)       # useful method to open image
    im = im.convert('RGB')
    im = im.resize(size)
    data_arr[shuf_arr[i],:,:,:] = im
    if i == train_data_num-1:
        break

data_arr = data_arr / 225 #normalize the data

# Training parameters
EPOCHS = 10000
BATCH = 50
LR = 0.0001

def loss_fun(x_dec, x, latent_x):
    x_dec = torch.reshape(x_dec,(32*32*3,-1))
    x = torch.reshape(x,(32*32*3,-1))
    MSE = torch.sum((x_dec-x) ** 2,0)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1.0 + latent_x - latent_x.pow(2) - latent_x.exp(), axis=1)
    return torch.mean(MSE + KLD*70)

vae = VAE().cuda()
optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

# For batch normalization , training data should be wrapped into wrapper
train_data_tensor = torch.Tensor(data_arr).cuda()
train_data = Data.TensorDataset(train_data_tensor,train_data_tensor)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH)
loss_arr = np.empty((0))

for epoch in range(EPOCHS) :
    for step, (b_x,b_y) in enumerate(train_loader):
        img, latent_x = vae(b_x.cuda())
        img = img.cuda();latent_x = latent_x.cuda()
        loss = loss_fun(img, b_x, latent_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EPOCHS == 0 :
            loss_arr = np.append(loss_arr,loss_fun(img,b_x,latent_x).cpu().data.numpy())
            print("EPOCH : ", epoch, "LOSS : {:.3f}".format(loss_arr[epoch]))
            if (loss_arr[epoch] <=256) or (epoch>=260) :
                break
    else:
        continue
    break

plt.figure();plt.plot(list(loss_arr));plt.title('losss');plt.savefig('./result/loss_batch_{:f}_lr_{:f}_amt_{:f}.png'.format(BATCH,LR,train_data_num))
torch.save(vae.state_dict(),'./result/training_model_batch_{:f}_lr_{:f}_amt_{:f}'.format(BATCH,LR,train_data_num))
