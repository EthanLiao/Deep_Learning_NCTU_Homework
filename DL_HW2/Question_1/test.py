import pygal
from pygal_maps_world.maps import World
from pygal.maps.world import COUNTRIES
import pandas as pd
import numpy as np
from GRU import GRU
from LSTM import LSTM
import torch
from torchvision import models

L = 7
filename = "/home/mint/Desktop/Data_Set/Deep_Learning_NCTU_Homework/DL_HW2/Question_1/covid_19.csv"
gru_model_dir = './result/GRU_training_model_batch_20.000000_lr_0.000100_L_7.000000'
lstm_model_dir = "./result/LSTM_training_model_batch_20.000000_lr_0.000100_L_7.000000"

def gru_predict(test_x):
    device = torch.device("cpu")
    gru=GRU(in_dim=L, hid_dim=2*L, layers=5, out_state = 2)
    gru.load_state_dict(torch.load(gru_model_dir,map_location = device))
    # transfer data into tensor vector form
    test_tensor_x = torch.Tensor(test_x)
    # prediction
    with torch.no_grad():
        out = gru(test_tensor_x)
        out = out.cpu().data.numpy()
        # # calculate accuracy
        # pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        return out


def lstm_predict(test_x):
    device = torch.device("cpu")
    lstm = LSTM(in_dim=L, hid_dim=2*L, layers=5, out_state = 2)
    lstm.load_state_dict(torch.load(lstm_model_dir,map_location = device))
    # transfer data into tensor vector form
    test_tensor_x = torch.Tensor(test_x)
    # prediction
    with torch.no_grad():
        out = lstm(test_tensor_x)
        out = out.cpu().data.numpy()
        # # calculate accuracy
        # pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        return out


# generate the abbreviation list for csv
use_cols_list = ["Country/Region"]
csv_list = pd.read_csv(filename, skiprows=[0, 1], usecols=use_cols_list).T
csv_list = csv_list.values.tolist()
csv_list = csv_list[0] #eliminate the list of list
abbrev_list = []

for cont in csv_list:
    for key, values in COUNTRIES.items():
        try:
            if values[0:5] == cont[0:5] :
                abbrev_list.append(key)
        except IndexError:
            if values[0:4] == cont[0:4]:
                abbrev_list.append(key)


# modify some value in abbrev_list
abbrev_list[163] = 'tw'
abbrev_list[159] = 'tz'
# collect the trend data
use_cols_list = [*range(3, 85)]
seq = pd.read_csv(filename, skiprows=[0, 1], usecols=use_cols_list)

seq_value = seq.values
for i in range(len(seq)):
    seq_num = seq_value
    seq_mat = np.array(seq_num)

# use GRU to extract the last sequence then predict

acsending_list = []
descending_list = []
for i,c in enumerate(seq_mat):
    try :
        last_seq = np.array(c[len(c)-L:]).reshape(-1,1,L)
        flag = np.argmax(gru_predict(last_seq),axis=1)
        if flag == 1:
            acsending_list.append(abbrev_list[i])
        else:
            descending_list.append(abbrev_list[i])
    except IndexError:
        break

word_map_chart = pygal.maps.world.World()
word_map_chart.title = 'Covid-19 Trend'
word_map_chart.add('acsending',acsending_list)
word_map_chart.add('descending',descending_list)
word_map_chart.render_to_png('GRU_Distribution.png')

# use LSTM to extract the last sequence then predict
acsending_list = []
descending_list = []
for i,c in enumerate(seq_mat):
    try :
        last_seq = np.array(c[len(c)-L:]).reshape(-1,1,L)
        flag = np.argmax(gru_predict(last_seq),axis=1)
        if flag == 1:
            acsending_list.append(abbrev_list[i])
        else:
            descending_list.append(abbrev_list[i])
    except IndexError:
        break

word_map_chart = pygal.maps.world.World()
word_map_chart.title = 'Covid-19 Trend'
word_map_chart.add('acsending',acsending_list)
word_map_chart.add('descending',descending_list)
word_map_chart.render_to_png('LSTM_Distribution.png')
