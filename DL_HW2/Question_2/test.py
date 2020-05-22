from PIL import Image
import numpy as np
import glob,os,math
from VAE import VAE
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.image as pltim

filename = "/home/mint/Desktop/Data_Set/Deep_Learning_NCTU_Homework/DL_HW2/Question_2/data/test/*.png"
write_dir = "./result/fake"
model_dir = './result/KLX100_model'

def predict(test_image_x):
    device = torch.device("cpu")
    vae = VAE()
    vae.load_state_dict(torch.load(model_dir,map_location = device))
    # transfer data into tensor vector form
    test_tensor_x = torch.Tensor(test_image_x)
    # prediction
    with torch.no_grad():
        img, latent_x = vae(test_tensor_x)
        img = img.cpu().data.numpy()
        # # calculate accuracy
        # pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        return img, latent_x

def synthesize(test_image_x):
    device = torch.device("cpu")
    vae = VAE()
    vae.load_state_dict(torch.load(model_dir,map_location = device))
    # transfer data into tensor vector form
    test_tensor_x = torch.Tensor(test_image_x)
    # prediction
    with torch.no_grad():
        dec = vae.latent2dec(test_tensor_x)
        img = vae.decoder(dec)
        img = img.cpu().data.numpy().reshape(32,32,3)
        # # calculate accuracy
        # pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        return img

size = (32,32)
test_data_num = 10
data_arr = np.zeros((test_data_num,32,32,3),dtype = float)

for i,file in enumerate(glob.glob(filename)):
    im = Image.open(file)       # useful method to open image
    im = im.convert('RGB')
    im = im.resize(size)
    data_arr[i,:,:,:] = im
    if i == test_data_num-1:
        break

data_arr = data_arr/225 #normalize the data
generate_img,latent_var = predict(data_arr[0].reshape((3,32,32)))
test_img = generate_img.reshape(32,32,3)


plt.figure(figsize=(2,2));plt.axis('off');plt.imshow(test_img);
pltim.imsave(write_dir+'/test.png',test_img)


# step 4 synthesized some image latent variable
img = synthesize(latent_var * 0.2+0.5)
plt.figure(figsize=(2,2));plt.axis('off');plt.imshow(img);
pltim.imsave(write_dir+'/test_synthsized.png',img)

# step 5 based on two images interpolation of two latent codes z
inter_value = [i*0.1 for i in range(11)]
generate_img_1,latent_var_1 = predict(data_arr[0].reshape((3,32,32)))
generate_img_2,latent_var_2 = predict(data_arr[4].reshape((3,32,32)))
for idx,inter in enumerate(inter_value):
    img = synthesize(latent_var_1 * inter+latent_var_2 * (1-inter) )
    plt.figure(figsize=(2,2));plt.axis('off');plt.imshow(img);
    pltim.imsave(write_dir+'/interploated_img_'+str(idx)+'.png',img)
